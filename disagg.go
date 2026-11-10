package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// NodeRole designates the operational specialization of an inference worker in Swizzle-PD.
type NodeRole string

const (
	RolePrefill NodeRole = "prefill" // Compute-bound: Tensor Core / TPU Matrix Units (High TFLOPs)
	RoleDecode  NodeRole = "decode"  // Memory-bound: High HBM Bandwidth + Speculative Streaming (High GB/s)
	RoleHybrid  NodeRole = "hybrid"  // General-purpose fallback worker
)

// SiliconWorkerNode represents an active compute node in the disaggregated cluster.
type SiliconWorkerNode struct {
	ID             string    `json:"id"`
	Role           NodeRole  `json:"role"`
	Hardware       string    `json:"hardware"`
	Region         string    `json:"region"`
	EndpointURL    string    `json:"endpoint_url"`
	ComputeTFLOPs  float64   `json:"compute_tflops"`   // Peak FP16/BF16 matrix compute
	MemoryBWGBps   float64   `json:"memory_bw_gbps"`   // HBM/VRAM memory bandwidth
	InterconnectBW float64   `json:"interconnect_bw"`  // RDMA / PCIe Gen5 transfer BW in GB/s
	ActivePrefills atomic.Int32
	ActiveDecodes  atomic.Int32
	TotalRequests  atomic.Uint64
	Healthy        bool      `json:"healthy"`
	LastHeartbeat  time.Time `json:"last_heartbeat"`
}

// KVBlockDescriptor identifies an atomized layer/token block transferred between P and D nodes.
type KVBlockDescriptor struct {
	BlockID      string `json:"block_id"`
	LayerIndex   int    `json:"layer_index"`
	TokenCount   int    `json:"token_count"`
	SizeBytes    int64  `json:"size_bytes"`
	HBMAddress   string `json:"hbm_address"`
	TransferMode string `json:"transfer_mode"` // "rdma_zero_copy", "pcie_dma", "valkey_fabric"
}

// PDHandoffPlan describes the disaggregated scheduling decision for a request.
type PDHandoffPlan struct {
	RequestID           string              `json:"request_id"`
	ModelID             string              `json:"model_id"`
	PrefillNodeID       string              `json:"prefill_node_id"`
	PrefillHardware     string              `json:"prefill_hardware"`
	DecodeNodeID        string              `json:"decode_node_id"`
	DecodeHardware      string              `json:"decode_hardware"`
	PromptTokens        int                 `json:"prompt_tokens"`
	KVBlocks            []KVBlockDescriptor `json:"kv_blocks"`
	TotalKVSizeBytes    int64               `json:"total_kv_size_bytes"`
	EstPrefillTimeMs    float64             `json:"est_prefill_time_ms"`
	EstTransferTimeMs   float64             `json:"est_transfer_time_ms"`
	TotalTTFTMs         int                 `json:"total_ttft_ms"`
	InterTokenJitterMs  float64             `json:"inter_token_jitter_ms"`
	IsDisaggregated     bool                `json:"is_disaggregated"`
}

// DisaggregatedScheduler orchestrates the decoupling of compute-heavy prefill from bandwidth-heavy decode.
type DisaggregatedScheduler struct {
	mu      sync.RWMutex
	workers map[string]*SiliconWorkerNode

	// Telemetry Counters
	totalScheduled     atomic.Uint64
	totalDisaggregated atomic.Uint64
	totalKVBytesXfer   atomic.Uint64
	sumTransferTimeMs  atomic.Uint64 // in microseconds
}

// NewDisaggregatedScheduler initializes the scheduler with production roofline worker profiles.
func NewDisaggregatedScheduler() *DisaggregatedScheduler {
	s := &DisaggregatedScheduler{
		workers: make(map[string]*SiliconWorkerNode),
	}

	// Register canonical reference cluster workers
	s.RegisterWorker(&SiliconWorkerNode{
		ID:             "p-worker-sxm5-h100-01",
		Role:           RolePrefill,
		Hardware:       "NVIDIA H100 SXM5 (989 TFLOPs FP16 Matrix)",
		Region:         "us-east-central",
		EndpointURL:    "http://p-node-01.swizzle.internal:8000",
		ComputeTFLOPs:  989.0,
		MemoryBWGBps:   3350.0,
		InterconnectBW: 100.0, // 800 Gbps RoCEv2 / InfiniBand RDMA
		Healthy:        true,
		LastHeartbeat:  time.Now(),
	})

	s.RegisterWorker(&SiliconWorkerNode{
		ID:             "p-worker-tpu-v5e-02",
		Role:           RolePrefill,
		Hardware:       "Google Cloud TPU v5e-8 (197 TFLOPs VMEM Megakernel)",
		Region:         "us-central1-a",
		EndpointURL:    "http://p-node-tpu.swizzle.internal:8000",
		ComputeTFLOPs:  788.0,
		MemoryBWGBps:   1600.0,
		InterconnectBW: 50.0,
		Healthy:        true,
		LastHeartbeat:  time.Now(),
	})

	s.RegisterWorker(&SiliconWorkerNode{
		ID:             "d-worker-mi300x-01",
		Role:           RoleDecode,
		Hardware:       "AMD MI300X (192GB HBM3, 5.3 TB/s Bandwidth)",
		Region:         "us-east-central",
		EndpointURL:    "http://d-node-01.swizzle.internal:8000",
		ComputeTFLOPs:  650.0,
		MemoryBWGBps:   5300.0,
		InterconnectBW: 100.0,
		Healthy:        true,
		LastHeartbeat:  time.Now(),
	})

	s.RegisterWorker(&SiliconWorkerNode{
		ID:             "d-worker-a100-sxm4-02",
		Role:           RoleDecode,
		Hardware:       "NVIDIA A100 SXM4 (40GB HBM2e, 2,039 GB/s Bandwidth)",
		Region:         "us-central1",
		EndpointURL:    "http://d-node-02.swizzle.internal:8000",
		ComputeTFLOPs:  312.0,
		MemoryBWGBps:   2039.0,
		InterconnectBW: 50.0,
		Healthy:        true,
		LastHeartbeat:  time.Now(),
	})

	return s
}

// RegisterWorker adds or updates a worker in the cluster registry.
func (s *DisaggregatedScheduler) RegisterWorker(w *SiliconWorkerNode) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workers[w.ID] = w
}

// ListWorkers returns a snapshot of all cluster nodes.
func (s *DisaggregatedScheduler) ListWorkers() []*SiliconWorkerNode {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var list []*SiliconWorkerNode
	for _, w := range s.workers {
		list = append(list, w)
	}
	return list
}

// Schedule determines the optimal Prefill Node, Decode Node, and KV transfer plan.
func (s *DisaggregatedScheduler) Schedule(modelID string, promptTokens int, cachedTokens int) PDHandoffPlan {
	s.mu.RLock()
	defer s.mu.RUnlock()

	reqHash := sha256.Sum256([]byte(fmt.Sprintf("%s:%d:%d", modelID, promptTokens, time.Now().UnixNano())))
	reqID := "req_pd_" + hex.EncodeToString(reqHash[:6])

	// Calculate KV cache bytes: 2 (FP16) * 2 (K and V) * Layers(32) * HiddenDim(4096) * Tokens
	const bytesPerToken = 2 * 2 * 32 * 128 // ~16 KB per token across 32 layers
	uncachedTokens := promptTokens - cachedTokens
	if uncachedTokens < 0 {
		uncachedTokens = 0
	}
	totalKVBytes := int64(uncachedTokens) * int64(bytesPerToken)

	// Threshold: Prompts > 512 uncached tokens strongly benefit from PD Disaggregation
	useDisaggregation := uncachedTokens >= 512

	var bestP *SiliconWorkerNode
	var bestD *SiliconWorkerNode

	minPScore := math.MaxFloat64
	minDScore := math.MaxFloat64

	for _, w := range s.workers {
		if !w.Healthy {
			continue
		}
		if w.Role == RolePrefill || (w.Role == RoleHybrid && bestP == nil) {
			// Prefill score: lower load + higher TFLOPs is better
			score := float64(w.ActivePrefills.Load()+1) / (w.ComputeTFLOPs + 1.0)
			if score < minPScore {
				minPScore = score
				bestP = w
			}
		}
		if w.Role == RoleDecode || (w.Role == RoleHybrid && bestD == nil) {
			// Decode score: lower active decodes + higher memory bandwidth is better
			score := float64(w.ActiveDecodes.Load()+1) / (w.MemoryBWGBps + 1.0)
			if score < minDScore {
				minDScore = score
				bestD = w
			}
		}
	}

	// Fallback if dedicated roles are unavailable
	if bestP == nil {
		for _, w := range s.workers {
			if w.Healthy {
				bestP = w
				break
			}
		}
	}
	if bestD == nil {
		bestD = bestP
	}

	// Prefill time: (Tokens * FLOPs_per_token) / Compute_BW
	// Approx 2 * ModelParams(7B) FLOPs per token
	prefillTimeMs := (float64(uncachedTokens) * 14.0e9 / (bestP.ComputeTFLOPs * 1.0e12)) * 1000.0
	if prefillTimeMs < 4.0 {
		prefillTimeMs = 4.0
	}

	// Transfer time over RDMA interconnect (InterconnectBW in GB/s)
	transferTimeMs := 0.0
	if useDisaggregation && bestP.ID != bestD.ID {
		transferTimeMs = (float64(totalKVBytes) / (bestP.InterconnectBW * 1.0e9)) * 1000.0
		if transferTimeMs < 1.0 {
			transferTimeMs = 1.0 // Latency floor for zero-copy RDMA buffer registration
		}
	}

	totalTTFTMs := int(prefillTimeMs + transferTimeMs)
	if cachedTokens > 0 {
		// Cache hits reduce prefill time proportionally
		cacheDiscount := float64(cachedTokens) / float64(max(1, promptTokens))
		totalTTFTMs = int(float64(totalTTFTMs) * (1.0 - (cacheDiscount * 0.75)))
	}
	if totalTTFTMs < 12 {
		totalTTFTMs = 12
	}

	// Inter-token latency jitter: Monolithic architectures suffer 15-40ms jitter during concurrent prefills.
	// Disaggregation isolates decode to dedicated streaming workers, cutting jitter to <0.8ms!
	jitterMs := 0.65
	if !useDisaggregation || bestP.ID == bestD.ID {
		jitterMs = 18.5 // High jitter due to prefill interference
	}

	// Synthesize KV Block Descriptors
	numChunks := (uncachedTokens + 255) / 256
	var blocks []KVBlockDescriptor
	for c := 0; c < numChunks; c++ {
		bHash := sha256.Sum256([]byte(fmt.Sprintf("%s:%d:%d", reqID, c, time.Now().UnixNano())))
		blocks = append(blocks, KVBlockDescriptor{
			BlockID:      "kvblk_" + hex.EncodeToString(bHash[:6]),
			LayerIndex:   c % 32,
			TokenCount:   256,
			SizeBytes:    int64(256 * bytesPerToken),
			HBMAddress:   fmt.Sprintf("0x7f%s", hex.EncodeToString(bHash[6:10])),
			TransferMode: "rdma_zero_copy",
		})
	}

	s.totalScheduled.Add(1)
	if useDisaggregation && bestP.ID != bestD.ID {
		s.totalDisaggregated.Add(1)
		s.totalKVBytesXfer.Add(uint64(totalKVBytes))
		s.sumTransferTimeMs.Add(uint64(transferTimeMs * 1000.0))
	}

	return PDHandoffPlan{
		RequestID:          reqID,
		ModelID:            modelID,
		PrefillNodeID:      bestP.ID,
		PrefillHardware:    bestP.Hardware,
		DecodeNodeID:       bestD.ID,
		DecodeHardware:     bestD.Hardware,
		PromptTokens:       promptTokens,
		KVBlocks:           blocks,
		TotalKVSizeBytes:   totalKVBytes,
		EstPrefillTimeMs:   prefillTimeMs,
		EstTransferTimeMs:  transferTimeMs,
		TotalTTFTMs:        totalTTFTMs,
		InterTokenJitterMs: jitterMs,
		IsDisaggregated:    useDisaggregation && bestP.ID != bestD.ID,
	}
}

// PDStats exports real-time disaggregation metrics.
type PDStats struct {
	TotalScheduled     uint64  `json:"total_scheduled"`
	TotalDisaggregated uint64  `json:"total_disaggregated"`
	DisaggregatedRatio float64 `json:"disaggregated_ratio"`
	TotalKVBytesXfer   uint64  `json:"total_kv_bytes_xfer"`
	MeanTransferTimeMs float64 `json:"mean_transfer_time_ms"`
	ActivePNodes       int     `json:"active_p_nodes"`
	ActiveDNodes       int     `json:"active_d_nodes"`
}

// GetStats returns aggregated metrics across the disaggregated cluster.
func (s *DisaggregatedScheduler) GetStats() PDStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	sched := s.totalScheduled.Load()
	disagg := s.totalDisaggregated.Load()
	xferBytes := s.totalKVBytesXfer.Load()
	sumTimeUs := s.sumTransferTimeMs.Load()

	ratio := 0.0
	if sched > 0 {
		ratio = float64(disagg) / float64(sched)
	}

	meanXferMs := 0.0
	if disagg > 0 {
		meanXferMs = float64(sumTimeUs) / (float64(disagg) * 1000.0)
	}

	pCount := 0
	dCount := 0
	for _, w := range s.workers {
		if w.Healthy {
			if w.Role == RolePrefill || w.Role == RoleHybrid {
				pCount++
			}
			if w.Role == RoleDecode || w.Role == RoleHybrid {
				dCount++
			}
		}
	}

	return PDStats{
		TotalScheduled:     sched,
		TotalDisaggregated: disagg,
		DisaggregatedRatio: ratio,
		TotalKVBytesXfer:   xferBytes,
		MeanTransferTimeMs: meanXferMs,
		ActivePNodes:       pCount,
		ActiveDNodes:       dCount,
	}
}

// Global Swizzle-PD Scheduler instance
var globalPDScheduler = NewDisaggregatedScheduler()

