package main

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// WorkerNode represents a live GPU or TPU inference worker in the Swizzle fleet.
type WorkerNode struct {
	ID               string    `json:"id"`
	EndpointURL      string    `json:"endpoint_url"`
	Hardware         string    `json:"hardware"`
	Region           string    `json:"region"`
	SupportedModels  []string  `json:"supported_models"`
	Healthy          bool      `json:"healthy"`
	KVCacheUsagePct  float64   `json:"kv_cache_usage_pct"`
	ActiveRequests   int64     `json:"active_requests"`
	TotalRouted      uint64    `json:"total_routed"`
	LastHeartbeat    time.Time `json:"last_heartbeat"`
	CircuitOpenUntil time.Time `json:"circuit_open_until,omitempty"`
}

// radixNode represents a content-addressed KV block in the distributed Radix Prefix Tree.
type radixNode struct {
	blockHash  string
	workerHits map[string]time.Time // workerID -> expiration time in HBM
	children   map[string]*radixNode
}

// RadixPrefixRouter implements content-addressed Longest-Prefix-Match (LPM) routing
// so multi-turn agent prompts consistently land on the exact GPU worker holding their KV cache.
type RadixPrefixRouter struct {
	mu          sync.RWMutex
	root        *radixNode
	workers     map[string]*WorkerNode
	clusterSalt []byte
	rrCounter   atomic.Uint64
}

func NewRadixPrefixRouter() *RadixPrefixRouter {
	salt := os.Getenv("SWIZZLE_CLUSTER_SALT")
	if salt == "" {
		salt = "swizzle-enterprise-zdr-kv-salt-v1"
	}
	r := &RadixPrefixRouter{
		root: &radixNode{
			workerHits: make(map[string]time.Time),
			children:   make(map[string]*radixNode),
		},
		workers:     make(map[string]*WorkerNode),
		clusterSalt: []byte(salt),
	}
	r.seedDefaultWorkers()
	return r
}

func (r *RadixPrefixRouter) seedDefaultWorkers() {
	upstreamEnv := os.Getenv("UPSTREAM_URL")
	if upstreamEnv == "" {
		upstreamEnv = "https://employers-willow-integrate-penetration.trycloudflare.com/v1"
	}
	urls := strings.Split(upstreamEnv, ",")
	for i, rawURL := range urls {
		u := strings.TrimSpace(rawURL)
		if u == "" {
			continue
		}
		workerID := "a100-sxm4-colab-01"
		if i > 0 {
			workerID = "gpu-worker-0" + string(rune('1'+i))
		}
		r.workers[workerID] = &WorkerNode{
			ID:          workerID,
			EndpointURL: strings.TrimRight(u, "/"),
			Hardware:    "NVIDIA A100-SXM4-40GB (CUDA XOR-Swizzled PagedAttention)",
			Region:      "global-edge-cf",
			SupportedModels: []string{
				"Qwen/Qwen2.5-Coder-7B-Instruct",
				"swizzle/qwen3.6-35b-a3b-fp8",
				"swizzle/gemma-4-31b-it",
				"swizzle/deepseek-v4.1-flash-1017",
				"swizzle/deepseek-v4.1-flash-abliterated",
			},
			Healthy:         true,
			KVCacheUsagePct: 18.4,
			LastHeartbeat:   time.Now().UTC(),
		}
	}
	r.workers["edge-robotics-pool-01"] = &WorkerNode{
		ID:          "edge-robotics-pool-01",
		EndpointURL: "local://swizzle-robotics-edge",
		Hardware:    "Student Laptop 2-8GB GPU + Colab Free T4 Elastic Pool",
		Region:      "edge-local-teleop",
		SupportedModels: []string{
			"swizzle/smolvla-450m-int8",
			"swizzle/pi0-fast-int4",
			"swizzle/openvla-7b-int4",
			"swizzle/dino-wm-latent",
			"swizzle/vjepa2-ac-int4",
		},
		Healthy:         true,
		KVCacheUsagePct: 12.0,
		LastHeartbeat:   time.Now().UTC(),
	}
}

// RegisterOrHeartbeat registers a new GPU/TPU worker or refreshes its heartbeat telemetry.
func (r *RadixPrefixRouter) RegisterOrHeartbeat(w WorkerNode) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if w.ID == "" {
		w.ID = "worker-" + hex.EncodeToString([]byte(w.EndpointURL))[:8]
	}
	w.EndpointURL = strings.TrimRight(w.EndpointURL, "/")
	w.Healthy = true
	w.LastHeartbeat = time.Now().UTC()

	if existing, ok := r.workers[w.ID]; ok {
		w.TotalRouted = existing.TotalRouted
	}
	r.workers[w.ID] = &w
}

// computeBlockHash generates a tenant-isolated cryptographic hash for a 128-char prompt chunk.
// Prevents cross-tenant KV cache timing side-channel attacks (Zero-Data-Retention compliance).
func (r *RadixPrefixRouter) computeBlockHash(tenantID, modelID, chunk string) string {
	mac := hmac.New(sha256.New, r.clusterSalt)
	mac.Write([]byte(tenantID))
	mac.Write([]byte("|"))
	mac.Write([]byte(modelID))
	mac.Write([]byte("|"))
	mac.Write([]byte(chunk))
	return hex.EncodeToString(mac.Sum(nil)[:12])
}

// RouteAndRecord evaluates the Radix Tree for the longest matching KV prefix and selects the optimal worker.
func (r *RadixPrefixRouter) RouteAndRecord(tenantID, modelID, prompt string) (selected *WorkerNode, matchedBlocks int, totalBlocks int) {
	const blockChars = 128
	if len(prompt) > 0 {
		totalBlocks = (len(prompt) + blockChars - 1) / blockChars
	} else {
		totalBlocks = 1
	}

	now := time.Now().UTC()
	r.mu.Lock()
	defer r.mu.Unlock()

	// Collect healthy workers whose circuit breaker is closed
	var healthyWorkers []*WorkerNode
	for _, w := range r.workers {
		if w.Healthy && now.After(w.CircuitOpenUntil) {
			healthyWorkers = append(healthyWorkers, w)
		}
	}

	// Walk the Radix Prefix Tree to score worker affinity by cached block hits
	workerAffinity := make(map[string]int)
	curr := r.root
	var pathNodes []*radixNode

	for i := 0; i < totalBlocks; i++ {
		start := i * blockChars
		end := (i + 1) * blockChars
		if end > len(prompt) {
			end = len(prompt)
		}
		chunk := ""
		if len(prompt) > start {
			chunk = prompt[start:end]
		}
		bHash := r.computeBlockHash(tenantID, modelID, chunk)

		next, exists := curr.children[bHash]
		if !exists {
			next = &radixNode{
				blockHash:  bHash,
				workerHits: make(map[string]time.Time),
				children:   make(map[string]*radixNode),
			}
			curr.children[bHash] = next
		}
		for wID, exp := range next.workerHits {
			if now.Before(exp) {
				workerAffinity[wID]++
			} else {
				delete(next.workerHits, wID)
			}
		}
		pathNodes = append(pathNodes, next)
		curr = next
	}

	// Pick worker with highest prefix match count (Longest-Prefix-Match Affinity)
	bestHits := -1
	for _, w := range healthyWorkers {
		hits := workerAffinity[w.ID]
		// Penalize overloaded workers (>90% KV cache usage or >32 concurrent in-flight requests)
		if w.KVCacheUsagePct > 92.0 || w.ActiveRequests > 32 {
			continue
		}
		if hits > bestHits {
			bestHits = hits
			selected = w
		}
	}

	// Fallback to round-robin if no healthy worker stood out
	if selected == nil && len(healthyWorkers) > 0 {
		idx := int(r.rrCounter.Add(1) % uint64(len(healthyWorkers)))
		selected = healthyWorkers[idx]
		bestHits = workerAffinity[selected.ID]
	}

	if bestHits < 0 {
		bestHits = 0
	}
	matchedBlocks = bestHits

	// Record new KV cache residency on the chosen worker (15-min sliding HBM TTL)
	targetWorkerID := "sim-worker-01"
	if selected != nil {
		targetWorkerID = selected.ID
		selected.TotalRouted++
	}
	ttl := now.Add(15 * time.Minute)
	for _, node := range pathNodes {
		node.workerHits[targetWorkerID] = ttl
	}

	return selected, matchedBlocks, totalBlocks
}

// MarkWorkerFailure trips a 15-second circuit breaker if an upstream worker fails or times out.
func (r *RadixPrefixRouter) MarkWorkerFailure(workerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if w, ok := r.workers[workerID]; ok {
		w.CircuitOpenUntil = time.Now().UTC().Add(15 * time.Second)
	}
}

// MarkWorkerSuccess clears any circuit breaker state on successful response.
func (r *RadixPrefixRouter) MarkWorkerSuccess(workerID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if w, ok := r.workers[workerID]; ok {
		w.CircuitOpenUntil = time.Time{}
		w.Healthy = true
		w.LastHeartbeat = time.Now().UTC()
	}
}

// ListWorkers returns a snapshot of all registered GPU/TPU workers.
func (r *RadixPrefixRouter) ListWorkers() []WorkerNode {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]WorkerNode, 0, len(r.workers))
	for _, w := range r.workers {
		out = append(out, *w)
	}
	return out
}

var prefixRouter = NewRadixPrefixRouter()

func handleWorkers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method == http.MethodPost {
		var node WorkerNode
		if err := json.NewDecoder(r.Body).Decode(&node); err != nil || node.EndpointURL == "" {
			http.Error(w, `{"error":{"message":"invalid worker registration payload"}}`, http.StatusBadRequest)
			return
		}
		prefixRouter.RegisterOrHeartbeat(node)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":  "registered",
			"workers": prefixRouter.ListWorkers(),
		})
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]any{
		"object":  "worker_pool",
		"routing": "Content-Addressed Radix-Tree Longest-Prefix Match (HMAC-SHA256 Tenant Isolated)",
		"workers": prefixRouter.ListWorkers(),
	})
}

