package main

import (
	"testing"
)

func TestPatchCacheEngine_SpatioTemporalReuse(t *testing.T) {
	engine := NewPatchCacheEngine()
	total := 196 // 14x14 ViT patch grid for 224x224 image

	// Frame 1: Cold start (0% reuse)
	hashes1 := make([]uint64, total)
	intensities1 := make([]float64, total)
	for i := 0; i < total; i++ {
		hashes1[i] = uint64(1000 + i)
		intensities1[i] = 0.50
	}

	m1 := engine.EvaluateFramePatches("so100-session-1", 14, 14, hashes1, intensities1, 0.02)
	if m1.ReusedPatches != 0 {
		t.Fatalf("expected 0 reused patches on cold frame, got %d", m1.ReusedPatches)
	}
	if m1.RecomputedPatches != total {
		t.Fatalf("expected %d recomputed patches, got %d", total, m1.RecomputedPatches)
	}

	// Frame 2: Robot gripper moves across 24 foreground patches; 172 background patches remain static
	hashes2 := make([]uint64, total)
	intensities2 := make([]float64, total)
	copy(hashes2, hashes1)
	copy(intensities2, intensities1)
	for i := 0; i < 24; i++ {
		hashes2[i] = uint64(9000 + i)
		intensities2[i] = 0.85
	}

	m2 := engine.EvaluateFramePatches("so100-session-1", 14, 14, hashes2, intensities2, 0.02)
	if m2.ReusedPatches != 172 {
		t.Fatalf("expected 172 reused patches on frame 2, got %d", m2.ReusedPatches)
	}
	if m2.PatchHitRatio < 0.87 {
		t.Fatalf("expected >87%% patch hit ratio, got %.4f", m2.PatchHitRatio)
	}
	if m2.EstimatedPrefillSpeedup < 3.5 {
		t.Fatalf("expected >3.5x prefill speedup, got %.2fx", m2.EstimatedPrefillSpeedup)
	}
}

