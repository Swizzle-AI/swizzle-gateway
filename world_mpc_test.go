package main

import (
	"testing"
)

func TestWorldMPCEngine_CEMConvergence(t *testing.T) {
	engine := NewWorldMPCEngine()
	req := WorldMPCPlanRequest{
		SessionID:       "dino-wm-pusht-0",
		Model:           "swizzle/dino-wm-latent",
		CurrentLatent:   []float64{0.10, -0.20, 0.05, 0.0, 0.0, 0.0, 1.0},
		GoalLatent:      []float64{0.55, 0.25, 0.40, 0.10, -0.10, 0.0, 1.0},
		NumTrajectories: 128,
		Horizon:         5,
		CEMIterations:   3,
		EliteFraction:   0.125,
	}

	resp := engine.PlanLatentMPC(req)
	if len(resp.OptimalActionSequence) != 5 {
		t.Fatalf("expected horizon 5 optimal action sequence, got %d", len(resp.OptimalActionSequence))
	}
	if resp.FinalPredictedDistance >= resp.InitialLatentDistance {
		t.Fatalf("expected CEM planner to reduce latent distance: initial=%.4f final=%.4f",
			resp.InitialLatentDistance, resp.FinalPredictedDistance)
	}
	if resp.PlanningSpeedup < 7.0 {
		t.Fatalf("expected >= 7.0x batched planning speedup, got %.2fx", resp.PlanningSpeedup)
	}
}

func TestRecommendModelsForStudentVRAM(t *testing.T) {
	models4GB := RecommendModelsForStudentVRAM(4.0, false)
	if len(models4GB) < 5 {
		t.Fatalf("expected at least 5 models runnable within 4GB VRAM, got %d", len(models4GB))
	}
	for _, m := range models4GB {
		if m.MinVRAMGB > 4.0 {
			t.Fatalf("model %s requires %.2f GB > 4.0 GB budget", m.ID, m.MinVRAMGB)
		}
	}
}

