package main

import (
	"strings"
	"sync"
	"sync/atomic"
)

// TrieNode represents a node in the speculative prompt-lookup prefix tree.
type TrieNode struct {
	children map[string]*TrieNode
	count    int
	isLeaf   bool
}

func newTrieNode() *TrieNode {
	return &TrieNode{
		children: make(map[string]*TrieNode),
	}
}

// SpeculativeEngineConfig configures the Swizzle-Draft speculative decoding engine.
type SpeculativeEngineConfig struct {
	DraftLength         int     // gamma: number of candidate tokens to draft (default: 4)
	MinGram             int     // minimum n-gram match required to trigger draft (default: 2)
	MaxGram             int     // maximum context window to consider for n-gram match (default: 6)
	MinConfidenceRatio  float64 // minimum transition probability to propose a branch (default: 0.65)
	MaxTrieNodes        int     // maximum in-memory nodes before LRU pruning (default: 100,000)
}

// DefaultSpeculativeConfig returns production-tuned parameters for coding agents.
func DefaultSpeculativeConfig() SpeculativeEngineConfig {
	return SpeculativeEngineConfig{
		DraftLength:        4,
		MinGram:            2,
		MaxGram:            6,
		MinConfidenceRatio: 0.60,
		MaxTrieNodes:       100000,
	}
}

// SpeculativeMetrics tracks real-time statistical performance of speculative decoding.
type SpeculativeMetrics struct {
	TotalDraftsProposed  atomic.Uint64
	TotalDraftsAccepted  atomic.Uint64
	TotalDraftsRejected  atomic.Uint64
	TokensGeneratedTotal atomic.Uint64
	FastForwardTokens    atomic.Uint64 // tokens accepted without target forward pass
}

// SpeculativeDraftEngine implements high-throughput Prompt-Lookup Speculative Decoding (PLD).
// It constructs an indexed n-gram suffix trie from prompt, system instructions, and AST context
// to propose speculative candidate branches for verification at zero GPU memory overhead.
type SpeculativeDraftEngine struct {
	mu      sync.RWMutex
	root    *TrieNode
	nodeCnt int
	cfg     SpeculativeEngineConfig
	metrics SpeculativeMetrics
}

// NewSpeculativeDraftEngine initializes an engine with given or default configuration.
func NewSpeculativeDraftEngine(cfg ...SpeculativeEngineConfig) *SpeculativeDraftEngine {
	c := DefaultSpeculativeConfig()
	if len(cfg) > 0 {
		c = cfg[0]
	}
	return &SpeculativeDraftEngine{
		root: newTrieNode(),
		cfg:  c,
	}
}

// Tokenize splits code and text into lexical units (words, operators, whitespace blocks).
func Tokenize(text string) []string {
	if len(text) == 0 {
		return nil
	}
	var tokens []string
	var cur strings.Builder

	isSep := func(r rune) bool {
		return r == ' ' || r == '\t' || r == '\n' || r == '\r' ||
			r == '(' || r == ')' || r == '{' || r == '}' ||
			r == '[' || r == ']' || r == ';' || r == ':' ||
			r == ',' || r == '.' || r == '"' || r == '\'' ||
			r == '=' || r == '+' || r == '-' || r == '*' ||
			r == '/' || r == '<' || r == '>' || r == '&' ||
			r == '|' || r == '!'
	}

	for _, r := range text {
		if isSep(r) {
			if cur.Len() > 0 {
				tokens = append(tokens, cur.String())
				cur.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}
	return tokens
}

// IngestContext populates the n-gram tree with tokens from the prompt and historical context.
func (e *SpeculativeDraftEngine) IngestContext(text string) int {
	tokens := Tokenize(text)
	if len(tokens) < e.cfg.MinGram {
		return 0
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	inserted := 0
	// Build sliding-window n-grams from minGram up to maxGram + draftLen
	windowSize := e.cfg.MaxGram + e.cfg.DraftLength
	for i := 0; i < len(tokens); i++ {
		end := i + windowSize
		if end > len(tokens) {
			end = len(tokens)
		}
		if end-i < e.cfg.MinGram {
			continue
		}
		e.insertNGram(tokens[i:end])
		inserted++
	}
	return inserted
}

func (e *SpeculativeDraftEngine) insertNGram(seq []string) {
	curr := e.root
	for _, tok := range seq {
		next, exists := curr.children[tok]
		if !exists {
			if e.nodeCnt >= e.cfg.MaxTrieNodes {
				return // node budget limit reached
			}
			next = newTrieNode()
			curr.children[tok] = next
			e.nodeCnt++
		}
		next.count++
		curr = next
	}
	curr.isLeaf = true
}

// ProposeDraft searches the active context trie for the longest matching suffix
// and extracts the most probable forward trajectory of up to gamma tokens.
func (e *SpeculativeDraftEngine) ProposeDraft(recentTokens []string) []string {
	if len(recentTokens) < e.cfg.MinGram {
		return nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	// Try longest suffix match first, down to minGram
	maxGram := e.cfg.MaxGram
	if len(recentTokens) < maxGram {
		maxGram = len(recentTokens)
	}

	for k := maxGram; k >= e.cfg.MinGram; k-- {
		suffix := recentTokens[len(recentTokens)-k:]
		curr := e.root
		matched := true

		for _, tok := range suffix {
			next, exists := curr.children[tok]
			if !exists {
				matched = false
				break
			}
			curr = next
		}

		if matched && len(curr.children) > 0 {
			// Extract greedy or highest-frequency continuation branch
			var draft []string
			traversal := curr
			for step := 0; step < e.cfg.DraftLength; step++ {
				var bestTok string
				var maxCnt int
				totalCnt := 0

				for tok, child := range traversal.children {
					totalCnt += child.count
					if child.count > maxCnt {
						maxCnt = child.count
						bestTok = tok
					}
				}

				if bestTok == "" || totalCnt == 0 {
					break
				}

				confidence := float64(maxCnt) / float64(totalCnt)
				if confidence < e.cfg.MinConfidenceRatio && step > 0 {
					break // Stop if confidence degrades below threshold
				}

				draft = append(draft, bestTok)
				traversal = traversal.children[bestTok]
				if len(traversal.children) == 0 {
					break
				}
			}

			if len(draft) > 0 {
				e.metrics.TotalDraftsProposed.Add(uint64(len(draft)))
				return draft
			}
		}
	}

	return nil
}

// VerificationResult contains the result of the speculative verification step.
type VerificationResult struct {
	AcceptedTokens []string
	RejectedIndex  int      // -1 if all accepted
	BonusToken     string   // target model's newly sampled token at rejection boundary
	SpeedupFactor  float64  // effective forward pass speedup
}

// Verify evaluates proposed draft tokens against actual target model outputs.
func (e *SpeculativeDraftEngine) Verify(proposed []string, targetActual []string) VerificationResult {
	res := VerificationResult{
		RejectedIndex: -1,
	}

	if len(proposed) == 0 {
		if len(targetActual) > 0 {
			res.AcceptedTokens = []string{targetActual[0]}
			res.SpeedupFactor = 1.0
		}
		return res
	}

	matchedCount := 0
	for i := 0; i < len(proposed); i++ {
		if i >= len(targetActual) {
			res.RejectedIndex = i
			break
		}
		if proposed[i] == targetActual[i] {
			res.AcceptedTokens = append(res.AcceptedTokens, proposed[i])
			matchedCount++
		} else {
			res.RejectedIndex = i
			res.BonusToken = targetActual[i]
			break
		}
	}

	e.metrics.TotalDraftsAccepted.Add(uint64(matchedCount))
	if res.RejectedIndex != -1 {
		rejectedCount := len(proposed) - matchedCount
		e.metrics.TotalDraftsRejected.Add(uint64(rejectedCount))
	}

	e.metrics.TokensGeneratedTotal.Add(uint64(matchedCount + 1))
	if matchedCount > 0 {
		e.metrics.FastForwardTokens.Add(uint64(matchedCount))
	}

	// Speedup calculation: (AcceptedTokens + 1 Target Step) / (1 Target Step + epsilon)
	res.SpeedupFactor = float64(matchedCount+1) / 1.0
	return res
}

// SpeculativeStats returns a snapshot of runtime metrics for telemetry and HTTP headers.
type SpeculativeStats struct {
	DraftsProposed uint64  `json:"drafts_proposed"`
	DraftsAccepted uint64  `json:"drafts_accepted"`
	DraftsRejected uint64  `json:"drafts_rejected"`
	AcceptanceRate float64 `json:"acceptance_rate"`
	EffectiveSpeedup float64 `json:"effective_speedup"`
}

// GetStats computes real-time acceptance rate and speedup ratio.
func (e *SpeculativeDraftEngine) GetStats() SpeculativeStats {
	prop := e.metrics.TotalDraftsProposed.Load()
	acc := e.metrics.TotalDraftsAccepted.Load()
	rej := e.metrics.TotalDraftsRejected.Load()

	rate := 0.0
	if prop > 0 {
		rate = float64(acc) / float64(prop)
	}

	// Mean speedup = 1 + gamma * alpha
	speedup := 1.0 + (float64(e.cfg.DraftLength) * rate)

	return SpeculativeStats{
		DraftsProposed:   prop,
		DraftsAccepted:   acc,
		DraftsRejected:   rej,
		AcceptanceRate:   rate,
		EffectiveSpeedup: speedup,
	}
}

// Global speculative draft engine instance
var globalSpeculativeEngine = NewSpeculativeDraftEngine()
