package main

import (
	"testing"
)

func TestTeleopBufferEngine_TemporalEnsembling(t *testing.T) {
	engine := NewTeleopBufferEngine()
	session := "so100-teleop-0"

	chunk0 := []Action7DoF{
		{0.10, 0.0, 0.20, 0, 0, 0, 1},
		{0.20, 0.0, 0.30, 0, 0, 0, 1},
		{0.30, 0.0, 0.40, 0, 0, 0, 1},
		{0.40, 0.0, 0.50, 0, 0, 0, 1},
	}
	m0 := engine.PushAndEnsembleChunk(session, 0, chunk0, 0.05)
	if m0.OverlappingPredictions != 1 {
		t.Fatalf("expected 1 overlapping prediction at step 0, got %d", m0.OverlappingPredictions)
	}

	chunk1 := []Action7DoF{
		{0.22, 0.0, 0.32, 0, 0, 0, 1},
		{0.32, 0.0, 0.42, 0, 0, 0, 1},
		{0.42, 0.0, 0.52, 0, 0, 0, 1},
		{0.52, 0.0, 0.62, 0, 0, 0, 1},
	}
	m1 := engine.PushAndEnsembleChunk(session, 1, chunk1, 0.05)
	if m1.OverlappingPredictions != 2 {
		t.Fatalf("expected 2 overlapping predictions at step 1, got %d", m1.OverlappingPredictions)
	}
	if len(m1.Upsampled50HzSubsteps) != 4 {
		t.Fatalf("expected 4 50Hz spline substeps, got %d", len(m1.Upsampled50HzSubsteps))
	}
	// Ensembled X at step 1 must blend 0.20 (from chunk0[1]) and 0.22 (from chunk1[0])
	if m1.EnsembledAction[0] <= 0.20 || m1.EnsembledAction[0] >= 0.22 {
		t.Fatalf("expected blended action between 0.20 and 0.22, got %.4f", m1.EnsembledAction[0])
	}
}

