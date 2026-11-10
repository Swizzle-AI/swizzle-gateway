package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestPrefixCacheAndTTFTDrop(t *testing.T) {
	_ = os.Setenv("SWIZZLE_DISABLE_UPSTREAM", "1")
	defer os.Unsetenv("SWIZZLE_DISABLE_UPSTREAM")

	ledger = NewPrefixCacheLedger()
	prefixRouter = NewRadixPrefixRouter()

	repoContext := strings.Repeat("func flashAttentionSwizzleMMA(q, k, v *float16) { /* 128-byte aligned shared memory tile */ }\n", 20)
	payload := map[string]any{
		"model": "swizzle/qwen3.6-35b-a3b-fp8",
		"messages": []map[string]string{
			{"role": "system", "content": repoContext},
			{"role": "user", "content": "Optimize bank conflict stride."},
		},
	}
	body, _ := json.Marshal(payload)

	// Turn 1: Cold cache
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rec1 := httptest.NewRecorder()
	handleChatCompletions(rec1, req1)

	if rec1.Code != http.StatusOK {
		t.Fatalf("expected 200 OK on turn 1, got %d", rec1.Code)
	}
	ttft1 := rec1.Header().Get("X-Swizzle-TTFT-Ms")
	hit1 := rec1.Header().Get("X-Swizzle-Cache-Hit-Tokens")
	if hit1 != "0" {
		t.Fatalf("expected 0 cached tokens on cold turn 1, got %s", hit1)
	}

	// Turn 2: Warm prefix cache (same repo context)
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	rec2 := httptest.NewRecorder()
	handleChatCompletions(rec2, req2)

	ttft2 := rec2.Header().Get("X-Swizzle-TTFT-Ms")
	hit2 := rec2.Header().Get("X-Swizzle-Cache-Hit-Tokens")
	if hit2 == "0" {
		t.Fatalf("expected non-zero cached tokens on warm turn 2, got %s", hit2)
	}
	t.Logf("Turn 1 (Cold): TTFT=%sms, CachedTokens=%s | Turn 2 (Warm): TTFT=%sms, CachedTokens=%s", ttft1, hit1, ttft2, hit2)
}

func TestRadixPrefixRouterMultiWorkerAffinity(t *testing.T) {
	router := NewRadixPrefixRouter()
	router.RegisterOrHeartbeat(WorkerNode{
		ID:              "h100-node-alpha",
		EndpointURL:     "http://10.0.1.10:8000/v1",
		Hardware:        "NVIDIA H100 SXM5",
		KVCacheUsagePct: 22.0,
	})
	router.RegisterOrHeartbeat(WorkerNode{
		ID:              "h100-node-beta",
		EndpointURL:     "http://10.0.1.11:8000/v1",
		Hardware:        "NVIDIA H100 SXM5",
		KVCacheUsagePct: 19.0,
	})

	longPrefix := strings.Repeat("module KernelSwizzle where\n-- 128-byte shared memory bank conflict free layout\n", 12)

	// Turn 1: Record prefix on initial worker
	w1, hits1, total1 := router.RouteAndRecord("org_acme", "swizzle/qwen3.6-35b-a3b-fp8", longPrefix+"Turn 1 query")
	if w1 == nil {
		t.Fatal("expected non-nil worker on turn 1")
	}

	// Turn 2: Same prefix + extended query MUST route to the exact same worker due to Radix LPM affinity
	w2, hits2, _ := router.RouteAndRecord("org_acme", "swizzle/qwen3.6-35b-a3b-fp8", longPrefix+"Turn 2 follow-up")
	if w2 == nil || w2.ID != w1.ID {
		t.Fatalf("expected Radix LPM affinity to keep tenant on worker %s, got %v", w1.ID, w2)
	}
	if hits2 <= hits1 {
		t.Fatalf("expected warm turn 2 block hits (%d) > cold turn 1 block hits (%d) out of %d blocks", hits2, hits1, total1)
	}
	t.Logf("Radix LPM Affinity verified: Worker=%s | Cold Hits=%d/%d -> Warm Hits=%d/%d", w2.ID, hits1, total1, hits2, total1)
}

func TestDLPSecretScrubberAndRestore(t *testing.T) {
	vault := NewSecretVault()
	rawPrompt := "Connect to AWS using AKIAIOSFODNN7EXAMPLE and GitHub token ghp_1234567890abcdefghijklmnopqrstuvwxyz"
	scrubbed, count := vault.Scrub(rawPrompt)
	if count != 2 {
		t.Fatalf("expected 2 secrets redacted, got %d", count)
	}
	if strings.Contains(scrubbed, "AKIAIOSFODNN7EXAMPLE") || strings.Contains(scrubbed, "ghp_1234567890") {
		t.Fatalf("expected secrets to be scrubbed from prompt, got: %s", scrubbed)
	}

	restored := vault.Restore("Configured key: <SWIZZLE_REDACTED_SECRET_1>")
	if !strings.Contains(restored, "AKIAIOSFODNN7EXAMPLE") {
		t.Fatalf("expected secret placeholder to be restored in output, got: %s", restored)
	}
}
