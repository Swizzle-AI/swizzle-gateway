package main

import (
	"testing"
)

func TestTokenize(t *testing.T) {
	code := `func Add(a, b int) int { return a + b; }`
	toks := Tokenize(code)
	if len(toks) == 0 {
		t.Fatalf("expected tokens, got none")
	}

	expectedPrefix := []string{"func", " ", "Add", "(", "a", ",", " ", "b"}
	for i, exp := range expectedPrefix {
		if toks[i] != exp {
			t.Errorf("token %d mismatch: got %q, want %q", i, toks[i], exp)
		}
	}
}

func TestSpeculativeDraftProposal(t *testing.T) {
	engine := NewSpeculativeDraftEngine(DefaultSpeculativeConfig())

	// Ingest a standard coding agent context with boilerplate function signatures
	contextCode := `
package main

import "net/http"

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
}
`
	inserted := engine.IngestContext(contextCode)
	if inserted == 0 {
		t.Fatalf("expected context to be indexed, got 0")
	}

	// Now simulate output generation matching the prefix: "func handleChatCompletions("
	recentTokens := []string{"func", " ", "handleChatCompletions", "("}
	draft := engine.ProposeDraft(recentTokens)

	if len(draft) == 0 {
		t.Fatalf("expected draft tokens to be proposed, got empty draft")
	}

	// Expected proposed continuation begins with "w"
	if draft[0] != "w" {
		t.Errorf("expected first draft token to be 'w', got %q", draft[0])
	}

	// Verify against exact actual target model continuation
	targetActual := []string{"w", " ", "http", ".", "ResponseWriter"}
	verif := engine.Verify(draft, targetActual)

	if len(verif.AcceptedTokens) == 0 {
		t.Errorf("expected at least 1 accepted token, got 0")
	}

	if verif.SpeedupFactor < 1.0 {
		t.Errorf("speedup factor should be >= 1.0, got %f", verif.SpeedupFactor)
	}

	stats := engine.GetStats()
	if stats.DraftsProposed == 0 {
		t.Errorf("expected DraftsProposed > 0")
	}
	if stats.DraftsAccepted == 0 {
		t.Errorf("expected DraftsAccepted > 0")
	}
	if stats.AcceptanceRate <= 0.0 {
		t.Errorf("expected positive acceptance rate, got %f", stats.AcceptanceRate)
	}
}

func TestSpeculativeVerificationRejection(t *testing.T) {
	engine := NewSpeculativeDraftEngine(DefaultSpeculativeConfig())
	prompt := `const API_ENDPOINT = "https://api.swizzle.cloud/v1/chat/completions";`
	engine.IngestContext(prompt)

	// Proposed draft based on prompt
	proposed := []string{"https", ":", "/", "/", "api", ".", "swizzle"}
	// Actual target model decided to output something different at index 4: "internal" instead of "api"
	actual := []string{"https", ":", "/", "/", "internal", ".", "swizzle"}

	verif := engine.Verify(proposed, actual)

	if len(verif.AcceptedTokens) != 4 {
		t.Fatalf("expected 4 accepted tokens, got %d", len(verif.AcceptedTokens))
	}

	if verif.RejectedIndex != 4 {
		t.Errorf("expected rejected index 4, got %d", verif.RejectedIndex)
	}

	if verif.BonusToken != "internal" {
		t.Errorf("expected bonus token 'internal', got %q", verif.BonusToken)
	}
}

