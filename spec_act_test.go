package main

import (
	"math"
	"testing"
)

func TestSpecActEngine_QuantizeRoundtrip(t *testing.T) {
	vals := []float64{-1.0, -0.5, 0.0, 0.25, 1.0}
	for _, v := range vals {
		tok := QuantizeActionToTokenID(v)
		if tok < ActionVocabBaseToken || tok >= ActionVocabBaseToken+ActionVocabBins {
			t.Fatalf("token ID %d out of action vocab bounds", tok)
		}
		deq := DequantizeTokenIDToAction(tok)
		if math.Abs(deq-v) > 0.01 {
			t.Fatalf("roundtrip drift too large: input=%.4f deq=%.4f", v, deq)
		}
	}
}

func TestSpecActEngine_KinematicExtrapolation(t *testing.T) {
	engine := NewSpecActEngine()
	session := "aloha-arm-test"

	// Simulate a smooth reaching trajectory across 4 steps
	traj := []Action7DoF{
		{0.10, -0.05, 0.20, 0.0, 0.0, 0.0, 1.0},
		{0.11, -0.05, 0.21, 0.0, 0.0, 0.0, 1.0},
		{0.12, -0.05, 0.22, 0.0, 0.0, 0.0, 1.0},
		{0.13, -0.05, 0.23, 0.0, 0.0, 0.0, 1.0},
	}

	var last SpecActResult
	for _, step := range traj {
		last = engine.DraftAndVerify7DoF(session, step, 2)
	}

	if last.AcceptedDoFTokens < 6 {
		t.Fatalf("expected >= 6 accepted DoF tokens on smooth trajectory, got %d", last.AcceptedDoFTokens)
	}
	if last.ActionDecodingSpeedup < 2.5 {
		t.Fatalf("expected >= 2.5x action decoding speedup, got %.2fx", last.ActionDecodingSpeedup)
	}
}

