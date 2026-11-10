package main

import (
	"testing"
)

func TestDisaggregatedSchedulerInitialization(t *testing.T) {
	sched := NewDisaggregatedScheduler()
	workers := sched.ListWorkers()
	if len(workers) < 4 {
		t.Fatalf("expected at least 4 registered silicon workers, got %d", len(workers))
	}

	pFound := false
	dFound := false
	for _, w := range workers {
		if w.Role == RolePrefill {
			pFound = true
		}
		if w.Role == RoleDecode {
			dFound = true
		}
	}
	if !pFound {
		t.Errorf("expected at least 1 dedicated Prefill worker")
	}
	if !dFound {
		t.Errorf("expected at least 1 dedicated Decode worker")
	}
}

func TestPDScheduleLargePromptDisaggregation(t *testing.T) {
	sched := NewDisaggregatedScheduler()

	// 2,048 token prompt with 0 cached tokens -> Should trigger Disaggregated P/D execution
	plan := sched.Schedule("Qwen/Qwen2.5-Coder-7B-Instruct", 2048, 0)

	if !plan.IsDisaggregated {
		t.Errorf("expected plan to be disaggregated for 2,048 tokens")
	}

	if plan.PrefillNodeID == "" || plan.DecodeNodeID == "" {
		t.Fatalf("both Prefill and Decode node IDs must be selected")
	}

	if plan.PrefillNodeID == plan.DecodeNodeID {
		t.Errorf("expected distinct Prefill and Decode nodes under disaggregated mode")
	}

	if len(plan.KVBlocks) == 0 {
		t.Errorf("expected generated KVBlockDescriptors, got 0")
	}

	if plan.TotalKVSizeBytes <= 0 {
		t.Errorf("expected positive TotalKVSizeBytes, got %d", plan.TotalKVSizeBytes)
	}

	// Inter-token latency jitter should be well under 1ms (<0.8ms target)
	if plan.InterTokenJitterMs > 1.0 {
		t.Errorf("expected ITL jitter < 1.0ms, got %f ms", plan.InterTokenJitterMs)
	}

	stats := sched.GetStats()
	if stats.TotalDisaggregated == 0 {
		t.Errorf("expected TotalDisaggregated > 0")
	}
	if stats.TotalKVBytesXfer == 0 {
		t.Errorf("expected TotalKVBytesXfer > 0")
	}
}

func TestPDScheduleSmallPromptMonolithic(t *testing.T) {
	sched := NewDisaggregatedScheduler()

	// 128 token prompt -> Below threshold (512 tokens), should execute without RDMA transfer overhead
	plan := sched.Schedule("Qwen/Qwen2.5-Coder-7B-Instruct", 128, 0)

	if plan.IsDisaggregated {
		t.Errorf("expected plan NOT to be disaggregated for small 128-token prompt")
	}

	if plan.EstTransferTimeMs != 0.0 {
		t.Errorf("expected 0ms transfer time for local execution, got %f", plan.EstTransferTimeMs)
	}
}

func TestPDFailoverHealthyWorker(t *testing.T) {
	sched := NewDisaggregatedScheduler()

	// Mark primary H100 prefill worker unhealthy
	sched.mu.Lock()
	if w, ok := sched.workers["p-worker-sxm5-h100-01"]; ok {
		w.Healthy = false
	}
	sched.mu.Unlock()

	plan := sched.Schedule("Qwen/Qwen2.5-Coder-7B-Instruct", 4096, 0)

	if plan.PrefillNodeID == "p-worker-sxm5-h100-01" {
		t.Errorf("scheduler should not route to unhealthy worker")
	}

	// Should failover to TPU v5e or hybrid node
	if plan.PrefillNodeID == "" {
		t.Errorf("expected failover to alternative prefill worker")
	}
}

