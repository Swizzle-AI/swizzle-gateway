package main

import (
	"strings"
	"testing"
)

func TestASTParserGo(t *testing.T) {
	parser := NewASTParser()
	code := `package server

import (
	"net/http"
	"time"
)

type Config struct {
	Port int
	Host string
}

func StartServer(cfg Config) error {
	return http.ListenAndServe(":8080", nil)
}

func HealthCheck(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}
`
	tree := parser.Parse(code, "server.go")
	if len(tree.Blocks) < 4 {
		t.Fatalf("expected at least 4 AST blocks, got %d", len(tree.Blocks))
	}

	foundFunc := false
	foundType := false
	foundImport := false

	for _, b := range tree.Blocks {
		if b.Kind == BlockFunction && b.Name == "StartServer" {
			foundFunc = true
		}
		if b.Kind == BlockType && b.Name == "Config" {
			foundType = true
		}
		if b.Kind == BlockImport {
			foundImport = true
		}
	}

	if !foundFunc {
		t.Errorf("expected to parse StartServer function block")
	}
	if !foundType {
		t.Errorf("expected to parse Config type block")
	}
	if !foundImport {
		t.Errorf("expected to parse import block")
	}
}

func TestASTDiffEngineNonLinearKVReuse(t *testing.T) {
	parser := NewASTParser()
	engine := NewASTDiffEngine()

	// Turn 1 Code: 4 independent functions
	turn1Code := `package service

import "fmt"

func CalculateMetrics(x int) int {
	return x * 2
}

func FormatOutput(res int) string {
	return fmt.Sprintf("result: %d", res)
}

func LogTelemetry(msg string) {
	fmt.Println("[TELEMETRY]", msg)
}

func CleanDatabase() error {
	return nil
}
`

	// Turn 2 Code: Agent modified ONLY CalculateMetrics (the first function)
	turn2Code := `package service

import "fmt"

func CalculateMetrics(x int) int {
	// Added optimization
	return (x * 2) + 42
}

func FormatOutput(res int) string {
	return fmt.Sprintf("result: %d", res)
}

func LogTelemetry(msg string) {
	fmt.Println("[TELEMETRY]", msg)
}

func CleanDatabase() error {
	return nil
}
`

	tree1 := parser.Parse(turn1Code, "service.go")
	tree2 := parser.Parse(turn2Code, "service.go")

	reorderedPrompt, summary := engine.CanonicalReorder(tree1, tree2)

	// In linear caching, modifying the first function invalidates all downstream functions
	if summary.LinearCacheRatio > 0.35 {
		t.Errorf("expected linear caching to drop below 35%%, got %.2f", summary.LinearCacheRatio)
	}

	// In Swizzle-Tree, FormatOutput, LogTelemetry, CleanDatabase, and imports remain cached
	if summary.KVSavingsRatio < 0.65 {
		t.Errorf("expected Swizzle-Tree KV savings to exceed 65%%, got %.2f", summary.KVSavingsRatio)
	}

	if summary.ModifiedBlocks != 1 {
		t.Errorf("expected exactly 1 modified block, got %d", summary.ModifiedBlocks)
	}

	if summary.UnchangedBlocks < 3 {
		t.Errorf("expected at least 3 unchanged blocks, got %d", summary.UnchangedBlocks)
	}

	// Verify CanonicalReorder placed unchanged functions BEFORE the modified CalculateMetrics
	calcIdx := strings.Index(reorderedPrompt, "func CalculateMetrics")
	formatIdx := strings.Index(reorderedPrompt, "func FormatOutput")
	if calcIdx == -1 || formatIdx == -1 {
		t.Fatalf("both functions must be present in reordered prompt")
	}

	if formatIdx > calcIdx {
		t.Errorf("CanonicalReorder should place warm unchanged block FormatOutput before modified block CalculateMetrics")
	}
}

