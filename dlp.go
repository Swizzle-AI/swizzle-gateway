package main

import (
	"fmt"
	"regexp"
	"strings"
)

var secretPatterns = []*regexp.Regexp{
	// AWS Access Key ID
	regexp.MustCompile(`AKIA[0-9A-Z]{16}`),
	// GitHub Personal Access Tokens (Classic & Fine-grained)
	regexp.MustCompile(`ghp_[0-9a-zA-Z]{36}`),
	regexp.MustCompile(`github_pat_[0-9a-zA-Z_]{40,}`),
	// Slack Bot / User Tokens
	regexp.MustCompile(`xox[baprs]-[0-9a-zA-Z\-]{10,}`),
	// Private Key PEM Blocks
	regexp.MustCompile(`(?s)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----`),
}

// SecretVault holds ephemeral in-memory secret mappings for a single request lifecycle.
// Never persisted to disk or telemetry logs (Zero-Data-Retention compliant).
type SecretVault struct {
	placeholders map[string]string
	counter      int
}

func NewSecretVault() *SecretVault {
	return &SecretVault{
		placeholders: make(map[string]string),
	}
}

// Scrub inspects text for high-entropy credentials and replaces them with deterministic placeholders.
func (v *SecretVault) Scrub(input string) (string, int) {
	if len(input) == 0 {
		return input, 0
	}
	redactedCount := 0
	output := input

	for _, pat := range secretPatterns {
		output = pat.ReplaceAllStringFunc(output, func(match string) string {
			// Check if already mapped in this request
			for ph, orig := range v.placeholders {
				if orig == match {
					redactedCount++
					return ph
				}
			}
			v.counter++
			ph := fmt.Sprintf("<SWIZZLE_REDACTED_SECRET_%d>", v.counter)
			v.placeholders[ph] = match
			redactedCount++
			return ph
		})
	}
	return output, redactedCount
}

// Restore replaces any placeholders in the model's completion output back with the original local secret.
func (v *SecretVault) Restore(completion string) string {
	if len(v.placeholders) == 0 || len(completion) == 0 {
		return completion
	}
	out := completion
	for ph, orig := range v.placeholders {
		out = strings.ReplaceAll(out, ph, orig)
	}
	return out
}

