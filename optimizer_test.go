package main

import (
	"strings"
	"testing"
)

func TestOptimizePrompt(t *testing.T) {
	samplePrompt := `
/* Copyright 2026 Example Corp.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software distributed under the License
   is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND. */
package main



import "fmt"   



/* Copyright 2026 Example Corp.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software distributed under the License
   is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND. */
func main() {    
	fmt.Println("Testing Swizzle token compression")
}
`
	optimized, rep := OptimizePrompt(samplePrompt)

	if rep.TokensSaved <= 0 {
		t.Errorf("Expected tokens to be saved, got %d", rep.TokensSaved)
	}

	if strings.Contains(optimized, "\n\n\n") {
		t.Errorf("Expected multi-newlines to be collapsed")
	}

	if !strings.Contains(optimized, "License Collapsed") {
		t.Errorf("Expected duplicate license to be deduplicated")
	}

	t.Logf("Original: %d toks, Optimized: %d toks, Saved: %d (Ratio: %.2f)",
		rep.OriginalTokens, rep.OptimizedTokens, rep.TokensSaved, rep.CompressionRatio)
}
