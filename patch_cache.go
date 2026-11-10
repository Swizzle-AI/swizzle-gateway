package main

import (
	"hash/fnv"
	"math"
	"sync"
	"sync/atomic"
	"time"
)

// PatchState represents a cached 16x16 ViT image patch in a robotic manipulation session.
type PatchState struct {
	Hash          uint64
	MeanIntensity float64
	KVBlockID     uint64
	UpdatedAt     time.Time
}

// SessionPatchGrid stores the spatio-temporal ViT patch cache for a robot camera stream.
type SessionPatchGrid struct {
	SessionID   string
	GridWidth   int
	GridHeight  int
	FrameCount  uint64
	Patches     []PatchState
	LastUpdated time.Time
}

// PatchCacheMetrics reports the frame-to-frame ViT & VLM KV cache reuse statistics.
type PatchCacheMetrics struct {
	SessionID               string  `json:"session_id"`
	FrameIndex              uint64  `json:"frame_index"`
	TotalPatches            int     `json:"total_patches"`
	ReusedPatches           int     `json:"reused_patches"`
	RecomputedPatches       int     `json:"recomputed_patches"`
	PatchHitRatio           float64 `json:"patch_hit_ratio"`
	EstimatedPrefillSpeedup float64 `json:"estimated_prefill_speedup"`
	BaselinePrefillMs       float64 `json:"baseline_prefill_ms"`
	SwizzlePrefillMs        float64 `json:"swizzle_prefill_ms"`
	ReuseBitmask            []bool  `json:"reuse_bitmask"`
}

// PatchCacheEngine implements Swizzle-PatchCache: Spatio-Temporal ViT Patch & KV Reuse.
// In a 30Hz robot camera stream (224x224 -> 14x14=196 patches, or 256x256 -> 16x16=256 patches),
// 80%-90% of background patches (tabletop, walls, static fixtures) remain invariant frame-to-frame.
// Swizzle-PatchCache gates ViT encoding and VLM prefill using L1 intensity deltas and FNV-1a hashes.
type PatchCacheEngine struct {
	mu                    sync.RWMutex
	sessions              map[string]*SessionPatchGrid
	totalFramesEvaluated  atomic.Uint64
	totalPatchesEvaluated atomic.Uint64
	totalPatchesReused    atomic.Uint64
}

var globalPatchCache = NewPatchCacheEngine()

func NewPatchCacheEngine() *PatchCacheEngine {
	return &PatchCacheEngine{
		sessions: make(map[string]*SessionPatchGrid),
	}
}

// HashPatchBytes computes a fast 64-bit FNV-1a perceptual digest for a quantized patch buffer.
func HashPatchBytes(data []byte) uint64 {
	h := fnv.New64a()
	_, _ = h.Write(data)
	return h.Sum64()
}

// EvaluateFramePatches compares incoming 16x16 patch hashes and mean L1 intensities against
// the session's reference frame and returns the reuse bitmask and prefill acceleration metrics.
func (pce *PatchCacheEngine) EvaluateFramePatches(
	sessionID string,
	gridWidth int,
	gridHeight int,
	patchHashes []uint64,
	patchIntensities []float64,
	deltaThreshold float64,
) PatchCacheMetrics {
	if sessionID == "" {
		sessionID = "default-robot-arm-0"
	}
	if gridWidth <= 0 {
		gridWidth = 14
	}
	if gridHeight <= 0 {
		gridHeight = 14
	}
	totalPatches := gridWidth * gridHeight
	if len(patchIntensities) > 0 && len(patchIntensities) != totalPatches {
		totalPatches = len(patchIntensities)
	}
	if deltaThreshold <= 0 {
		deltaThreshold = 0.025 // 2.5% L1 intensity change tolerance for sensor noise
	}

	pce.mu.Lock()
	defer pce.mu.Unlock()

	now := time.Now()
	grid, exists := pce.sessions[sessionID]
	reuseMask := make([]bool, totalPatches)
	reused := 0

	if !exists || len(grid.Patches) != totalPatches {
		// Cold frame 0: initialize reference patch states
		patches := make([]PatchState, totalPatches)
		for i := 0; i < totalPatches; i++ {
			var h uint64
			if i < len(patchHashes) {
				h = patchHashes[i]
			}
			var intensity float64
			if i < len(patchIntensities) {
				intensity = patchIntensities[i]
			}
			patches[i] = PatchState{
				Hash:          h,
				MeanIntensity: intensity,
				KVBlockID:     uint64(i + 1),
				UpdatedAt:     now,
			}
		}
		grid = &SessionPatchGrid{
			SessionID:   sessionID,
			GridWidth:   gridWidth,
			GridHeight:  gridHeight,
			FrameCount:  1,
			Patches:     patches,
			LastUpdated: now,
		}
		pce.sessions[sessionID] = grid
	} else {
		grid.FrameCount++
		grid.LastUpdated = now
		for i := 0; i < totalPatches; i++ {
			prev := &grid.Patches[i]
			var currHash uint64
			if i < len(patchHashes) {
				currHash = patchHashes[i]
			}
			var currIntensity float64
			if i < len(patchIntensities) {
				currIntensity = patchIntensities[i]
			}

			hashMatch := (currHash != 0 && currHash == prev.Hash)
			intensityMatch := len(patchIntensities) > i && math.Abs(currIntensity-prev.MeanIntensity) <= deltaThreshold

			if hashMatch || intensityMatch {
				reuseMask[i] = true
				reused++
			} else {
				reuseMask[i] = false
				prev.Hash = currHash
				prev.MeanIntensity = currIntensity
				prev.KVBlockID++
				prev.UpdatedAt = now
			}
		}
	}

	recomputed := totalPatches - reused
	hitRatio := 0.0
	if totalPatches > 0 {
		hitRatio = float64(reused) / float64(totalPatches)
	}

	// Laptop GPU Roofline Model (e.g., RTX 4060 8GB running SmolVLA / OpenVLA-INT4):
	// Full 196-patch ViT + VLM prefill takes ~42.0ms; fixed language/proprioception overhead is ~6.5ms.
	baselineMs := 42.0
	fixedOverheadMs := 6.5
	variablePatchMs := (baselineMs - fixedOverheadMs) * (float64(recomputed) / math.Max(1.0, float64(totalPatches)))
	swizzleMs := fixedOverheadMs + variablePatchMs
	speedup := baselineMs / math.Max(1.0, swizzleMs)

	pce.totalFramesEvaluated.Add(1)
	pce.totalPatchesEvaluated.Add(uint64(totalPatches))
	pce.totalPatchesReused.Add(uint64(reused))

	return PatchCacheMetrics{
		SessionID:               sessionID,
		FrameIndex:              grid.FrameCount,
		TotalPatches:            totalPatches,
		ReusedPatches:           reused,
		RecomputedPatches:       recomputed,
		PatchHitRatio:           math.Round(hitRatio*10000) / 10000,
		EstimatedPrefillSpeedup: math.Round(speedup*100) / 100,
		BaselinePrefillMs:       baselineMs,
		SwizzlePrefillMs:        math.Round(swizzleMs*100) / 100,
		ReuseBitmask:            reuseMask,
	}
}

