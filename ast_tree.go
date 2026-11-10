package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

// ASTBlockKind identifies the syntactic nature of a code block.
type ASTBlockKind string

const (
	BlockImport   ASTBlockKind = "import"
	BlockType     ASTBlockKind = "type"
	BlockFunction ASTBlockKind = "function"
	BlockClass    ASTBlockKind = "class"
	BlockGlobal   ASTBlockKind = "global"
	BlockComment  ASTBlockKind = "comment"
	BlockGeneral  ASTBlockKind = "general"
)

// ASTBlock represents an atomized semantic block of source code.
type ASTBlock struct {
	ID        string       `json:"id"`
	Kind      ASTBlockKind `json:"kind"`
	Name      string       `json:"name"`
	Content   string       `json:"content"`
	TokenEst  int          `json:"token_est"`
	Hash      string       `json:"hash"`      // Content-addressed SHA-256 hash
	IsMutable bool         `json:"is_mutable"` // true if actively edited
}

// ComputeHash computes a truncated content-addressed hash of the AST block.
func (b *ASTBlock) ComputeHash() string {
	// Canonicalize: trim trailing spaces and normalize line endings
	norm := strings.TrimSpace(strings.ReplaceAll(b.Content, "\r\n", "\n"))
	sum := sha256.Sum256([]byte(string(b.Kind) + ":" + b.Name + ":" + norm))
	b.Hash = hex.EncodeToString(sum[:12])
	b.TokenEst = (len(norm) / 4) + 2
	return b.Hash
}

// ASTTree represents a decomposed syntactic tree of source code blocks.
type ASTTree struct {
	Language string     `json:"language"`
	Blocks   []ASTBlock `json:"blocks"`
}

// TotalTokens estimates the token count of the entire AST tree.
func (t *ASTTree) TotalTokens() int {
	total := 0
	for _, b := range t.Blocks {
		total += b.TokenEst
	}
	return total
}

// ASTParser splits code into syntactic blocks using lexical boundary heuristics.
type ASTParser struct{}

// NewASTParser creates a parser instance.
func NewASTParser() *ASTParser {
	return &ASTParser{}
}

// Parse decomposes code into ASTBlocks by recognizing top-level declarations.
func (p *ASTParser) Parse(code, filename string) ASTTree {
	ext := strings.ToLower(filename)
	lang := "general"
	if strings.HasSuffix(ext, ".go") {
		lang = "go"
	} else if strings.HasSuffix(ext, ".py") {
		lang = "python"
	} else if strings.HasSuffix(ext, ".ts") || strings.HasSuffix(ext, ".tsx") || strings.HasSuffix(ext, ".js") {
		lang = "typescript"
	}

	lines := strings.Split(strings.ReplaceAll(code, "\r\n", "\n"), "\n")
	var blocks []ASTBlock
	var curBlock strings.Builder
	curKind := BlockGeneral
	curName := "preamble"

	flush := func() {
		content := curBlock.String()
		if strings.TrimSpace(content) != "" {
			b := ASTBlock{
				Kind:    curKind,
				Name:    curName,
				Content: content,
			}
			b.ComputeHash()
			b.ID = fmt.Sprintf("%s_%s_%s", b.Kind, b.Name, b.Hash[:6])
			blocks = append(blocks, b)
		}
		curBlock.Reset()
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		// Top-level declaration detection
		isNewBlock := false
		newKind := BlockGeneral
		newName := ""

		if strings.HasPrefix(trimmed, "import ") || strings.HasPrefix(trimmed, "package ") || strings.HasPrefix(trimmed, "from ") {
			if curKind != BlockImport {
				isNewBlock = true
				newKind = BlockImport
				newName = "imports"
			}
		} else if strings.HasPrefix(trimmed, "type ") || strings.HasPrefix(trimmed, "interface ") || strings.HasPrefix(trimmed, "struct ") {
			isNewBlock = true
			newKind = BlockType
			parts := strings.Fields(trimmed)
			if len(parts) > 1 {
				newName = parts[1]
			}
		} else if strings.HasPrefix(trimmed, "func ") || strings.HasPrefix(trimmed, "def ") || strings.HasPrefix(trimmed, "function ") ||
			(strings.Contains(trimmed, "const ") && strings.Contains(trimmed, " = (") && strings.Contains(trimmed, ") =>")) {
			isNewBlock = true
			newKind = BlockFunction
			parts := strings.Fields(trimmed)
			if len(parts) > 1 {
				newName = strings.Split(parts[1], "(")[0]
				newName = strings.TrimPrefix(newName, "*")
			}
		} else if strings.HasPrefix(trimmed, "class ") {
			isNewBlock = true
			newKind = BlockClass
			parts := strings.Fields(trimmed)
			if len(parts) > 1 {
				newName = strings.Split(parts[1], "(")[0]
				newName = strings.Split(newName, ":")[0]
			}
		}

		if isNewBlock && curBlock.Len() > 0 {
			flush()
			curKind = newKind
			curName = newName
		}

		curBlock.WriteString(line)
		curBlock.WriteString("\n")
	}
	flush()

	return ASTTree{
		Language: lang,
		Blocks:   blocks,
	}
}

// ASTDiffSummary reports the syntactic differences between two versions of code.
type ASTDiffSummary struct {
	TotalBlocks      int     `json:"total_blocks"`
	UnchangedBlocks  int     `json:"unchanged_blocks"`
	ModifiedBlocks   int     `json:"modified_blocks"`
	AddedBlocks      int     `json:"added_blocks"`
	DeletedBlocks    int     `json:"deleted_blocks"`
	TotalTokens      int     `json:"total_tokens"`
	ReusedTokens     int     `json:"reused_tokens"`
	PrefillTokens    int     `json:"prefill_tokens"`
	KVSavingsRatio   float64 `json:"kv_savings_ratio"`   // ReusedTokens / TotalTokens
	LinearCacheRatio float64 `json:"linear_cache_ratio"` // Standard linear cache hit ratio
}

// ASTDiffEngine computes semantic diffs and constructs canonical prefix trees.
type ASTDiffEngine struct {
	mu           sync.RWMutex
	kvBlockStore map[string]int // maps ASTBlock.Hash to TokenEst
	parser       *ASTParser

	// Metrics
	metrics struct {
		TotalDiffEvaluations atomic.Uint64
		TotalReusedTokens    atomic.Uint64
		TotalPrefillTokens   atomic.Uint64
	}
}

// NewASTDiffEngine initializes an AST-Diff aware engine.
func NewASTDiffEngine() *ASTDiffEngine {
	return &ASTDiffEngine{
		kvBlockStore: make(map[string]int),
		parser:       NewASTParser(),
	}
}

// CompareTrees calculates semantic block reuse and compares Swizzle-Tree vs linear caching.
func (e *ASTDiffEngine) CompareTrees(prevTree, currTree ASTTree) ASTDiffSummary {
	prevMap := make(map[string]ASTBlock)
	for _, b := range prevTree.Blocks {
		prevMap[b.Name] = b
	}

	totalTokens := 0
	reusedTokens := 0
	unchanged := 0
	modified := 0
	added := 0

	// Also simulate standard linear prefix cache hit for comparison
	firstDivergenceIdx := -1
	linearReusedTokens := 0

	for i, b := range currTree.Blocks {
		totalTokens += b.TokenEst

		// Linear prefix caching simulation
		if firstDivergenceIdx == -1 {
			if i < len(prevTree.Blocks) && prevTree.Blocks[i].Hash == b.Hash {
				linearReusedTokens += b.TokenEst
			} else {
				firstDivergenceIdx = i
			}
		}

		// Swizzle-Tree non-linear AST matching
		if prevBlock, exists := prevMap[b.Name]; exists {
			if prevBlock.Hash == b.Hash {
				unchanged++
				reusedTokens += b.TokenEst
			} else {
				modified++
			}
		} else {
			added++
		}
	}

	deleted := 0
	currMap := make(map[string]bool)
	for _, b := range currTree.Blocks {
		currMap[b.Name] = true
	}
	for name := range prevMap {
		if !currMap[name] {
			deleted++
		}
	}

	prefillTokens := totalTokens - reusedTokens
	if prefillTokens < 0 {
		prefillTokens = 0
	}

	kvRatio := 0.0
	if totalTokens > 0 {
		kvRatio = float64(reusedTokens) / float64(totalTokens)
	}

	linearRatio := 0.0
	if totalTokens > 0 {
		linearRatio = float64(linearReusedTokens) / float64(totalTokens)
	}

	e.metrics.TotalDiffEvaluations.Add(1)
	e.metrics.TotalReusedTokens.Add(uint64(reusedTokens))
	e.metrics.TotalPrefillTokens.Add(uint64(prefillTokens))

	// Register warm blocks in store
	e.mu.Lock()
	for _, b := range currTree.Blocks {
		e.kvBlockStore[b.Hash] = b.TokenEst
	}
	e.mu.Unlock()

	return ASTDiffSummary{
		TotalBlocks:      len(currTree.Blocks),
		UnchangedBlocks:  unchanged,
		ModifiedBlocks:   modified,
		AddedBlocks:      added,
		DeletedBlocks:    deleted,
		TotalTokens:      totalTokens,
		ReusedTokens:     reusedTokens,
		PrefillTokens:    prefillTokens,
		KVSavingsRatio:   kvRatio,
		LinearCacheRatio: linearRatio,
	}
}

// CanonicalReorder clusters unchanged, cached AST blocks at the front of the prompt,
// pushing the modified/added blocks to the suffix. This converts non-linear code edits
// into a contiguous warm prefix, guaranteeing maximum KV cache hits on any standard worker!
func (e *ASTDiffEngine) CanonicalReorder(prevTree, currTree ASTTree) (reorderedPrompt string, summary ASTDiffSummary) {
	summary = e.CompareTrees(prevTree, currTree)

	prevMap := make(map[string]string)
	for _, b := range prevTree.Blocks {
		prevMap[b.Name] = b.Hash
	}

	var warmPrefix strings.Builder
	var activeSuffix strings.Builder

	for _, b := range currTree.Blocks {
		prevHash, exists := prevMap[b.Name]
		if exists && prevHash == b.Hash {
			// Unchanged block -> Place into contiguous warm prefix
			warmPrefix.WriteString(b.Content)
			if !strings.HasSuffix(b.Content, "\n") {
				warmPrefix.WriteString("\n")
			}
		} else {
			// Modified or Added block -> Place into active suffix
			activeSuffix.WriteString(b.Content)
			if !strings.HasSuffix(b.Content, "\n") {
				activeSuffix.WriteString("\n")
			}
		}
	}

	var full strings.Builder
	full.WriteString(warmPrefix.String())
	full.WriteString(activeSuffix.String())

	return full.String(), summary
}

// Global AST Diff Engine instance
var globalASTDiffEngine = NewASTDiffEngine()

