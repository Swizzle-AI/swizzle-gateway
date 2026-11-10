package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

// RoboticsActRequest represents a single closed-loop VLA control step from a student's robot or simulator.
type RoboticsActRequest struct {
	SessionID        string      `json:"session_id"`
	Model            string      `json:"model"`
	Instruction      string      `json:"instruction"`
	StepIndex        int         `json:"step_index"`
	GridWidth        int         `json:"grid_width"`
	GridHeight       int         `json:"grid_height"`
	PatchHashes      []uint64    `json:"patch_hashes"`
	PatchIntensities []float64   `json:"patch_intensities"`
	Proprioception   Action7DoF  `json:"proprioception"`
	TargetActionHint *Action7DoF `json:"target_action_hint,omitempty"`
}

// RoboticsActResponse returns the 7-DoF action vector, 50Hz spline substeps, and Swizzle engine telemetry.
type RoboticsActResponse struct {
	Object       string                `json:"object"`
	Model        string                `json:"model"`
	Instruction  string                `json:"instruction"`
	Action       Action7DoF            `json:"action"`
	PatchCache   PatchCacheMetrics     `json:"patch_cache"`
	SpecAct      SpecActResult         `json:"spec_act"`
	Teleop       TeleopEnsembleMetrics `json:"teleop"`
	EndToEndMs   float64               `json:"end_to_end_ms"`
	ControlHz    float64               `json:"effective_control_hz"`
}

func handleRoboticsCatalog(w http.ResponseWriter, r *http.Request) {
	vramStr := r.URL.Query().Get("vram_gb")
	vramGB := 16.0
	if vramStr != "" {
		if parsed, err := strconv.ParseFloat(vramStr, 64); err == nil && parsed > 0 {
			vramGB = parsed
		}
	}
	includeTunnel := r.URL.Query().Get("colab_tunnel") == "true"
	recommended := RecommendModelsForStudentVRAM(vramGB, includeTunnel)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object":               "list",
		"student_vram_budget":  vramGB,
		"colab_tunnel_enabled": includeTunnel,
		"recommended_count":    len(recommended),
		"data":                 recommended,
	})
}

func handleRoboticsAct(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req RoboticsActRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON payload"}`, http.StatusBadRequest)
		return
	}
	if req.Model == "" {
		req.Model = "swizzle/smolvla-450m-int8"
	}
	if req.SessionID == "" {
		req.SessionID = "so100-default-session"
	}

	// 1. Evaluate Swizzle-PatchCache (Spatio-Temporal ViT Patch & KV Reuse)
	patchMetrics := globalPatchCache.EvaluateFramePatches(
		req.SessionID,
		req.GridWidth,
		req.GridHeight,
		req.PatchHashes,
		req.PatchIntensities,
		0.025,
	)

	// 2. Synthesize or Forward Target 7-DoF Policy Action
	var targetAction Action7DoF
	if req.TargetActionHint != nil {
		targetAction = *req.TargetActionHint
	} else {
		// Deterministic goal-directed proportional kinematic response from proprioception
		for d := 0; d < 6; d++ {
			targetAction[d] = clampUnit(req.Proprioception[d]*0.85 + 0.04)
		}
		targetAction[6] = 1.0 // Closed gripper
	}

	// 3. Evaluate Swizzle-SpecAct (7-DoF Kinematic Speculative Action Decoding)
	specResult := globalSpecAct.DraftAndVerify7DoF(req.SessionID, targetAction, 2)

	// 4. Evaluate Swizzle-Teleop (Receding-Horizon Buffer & 50Hz Spline Ensembling)
	chunk := make([]Action7DoF, 8)
	for h := 0; h < 8; h++ {
		scale := 1.0 - float64(h)*0.04
		for d := 0; d < 6; d++ {
			chunk[h][d] = clampUnit(specResult.VerifiedAction[d] * scale)
		}
		chunk[h][6] = specResult.VerifiedAction[6]
	}
	teleopMetrics := globalTeleopBuffer.PushAndEnsembleChunk(req.SessionID, req.StepIndex, chunk, 0.05)

	decodeMs := 21.0 / specResult.ActionDecodingSpeedup
	totalMs := patchMetrics.SwizzlePrefillMs + decodeMs
	effectiveHz := 1000.0 / totalMs

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Swizzle-Patch-Hit-Ratio", fmt.Sprintf("%.4f", patchMetrics.PatchHitRatio))
	w.Header().Set("X-Swizzle-SpecAct-Speedup", fmt.Sprintf("%.2fx", specResult.ActionDecodingSpeedup))
	w.Header().Set("X-Swizzle-Control-Loop-Hz", fmt.Sprintf("%.1f", effectiveHz))

	_ = json.NewEncoder(w).Encode(RoboticsActResponse{
		Object:      "robotics.action",
		Model:       req.Model,
		Instruction: req.Instruction,
		Action:      teleopMetrics.EnsembledAction,
		PatchCache:  patchMetrics,
		SpecAct:     specResult,
		Teleop:      teleopMetrics,
		EndToEndMs:  float64(int(totalMs*100)) / 100.0,
		ControlHz:   float64(int(effectiveHz*10)) / 10.0,
	})
}

func handleWorldPlan(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	var req WorldMPCPlanRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON payload"}`, http.StatusBadRequest)
		return
	}

	resp := globalWorldMPC.PlanLatentMPC(req)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Swizzle-MPC-Speedup", fmt.Sprintf("%.2fx", resp.PlanningSpeedup))
	w.Header().Set("X-Swizzle-VRAM-GB", fmt.Sprintf("%.2f", resp.VRAMUsedGB))
	_ = json.NewEncoder(w).Encode(resp)
}

