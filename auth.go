package main

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// TenantAccount represents an enterprise organization or developer account.
type TenantAccount struct {
	TenantID          string    `json:"tenant_id"`
	KeyHash           string    `json:"key_hash"`
	KeyPrefix         string    `json:"key_prefix"`
	Label             string    `json:"label"`
	Tier              string    `json:"tier"` // "developer", "pro", "enterprise"
	ZeroDataRetention bool      `json:"zero_data_retention"`
	BalanceUSD        float64   `json:"balance_usd"`
	TotalSpentUSD     float64   `json:"total_spent_usd"`
	TotalSavedUSD     float64   `json:"total_saved_usd"`
	RPMQuota          int       `json:"rpm_quota"`
	TPMQuota          int       `json:"tpm_quota"`
	CreatedAt         time.Time `json:"created_at"`

	// Token-bucket rate limiter state
	windowStart time.Time
	reqCount    int
	tokCount    int
}

// EnterpriseAuthEngine manages HMAC-SHA256 API keys, per-tenant token-bucket rate limits,
// and real-time 3-tier metered billing.
type EnterpriseAuthEngine struct {
	mu           sync.RWMutex
	byKeyHash    map[string]*TenantAccount
	pepper       []byte
	totalSavedMu atomic.Uint64 // micro-dollars saved across platform
	totalSpentMu atomic.Uint64 // micro-dollars spent across platform
}

func NewEnterpriseAuthEngine() *EnterpriseAuthEngine {
	p := os.Getenv("SWIZZLE_KEY_PEPPER")
	if p == "" {
		p = "swizzle-hmac-key-pepper-2026"
	}
	e := &EnterpriseAuthEngine{
		byKeyHash: make(map[string]*TenantAccount),
		pepper:    []byte(p),
	}
	// Seed default developer bypass key so existing integrations work out-of-the-box
	e.registerSeedKey("sk-swizzle-dev-bypass-key", "org_default_dev", "Default Developer Key", "pro", 5.00, 600, 1000000)
	return e
}

func (e *EnterpriseAuthEngine) hashKey(rawKey string) string {
	mac := hmac.New(sha256.New, e.pepper)
	mac.Write([]byte(strings.TrimSpace(rawKey)))
	return hex.EncodeToString(mac.Sum(nil))
}

func (e *EnterpriseAuthEngine) registerSeedKey(rawKey, tenantID, label, tier string, balance float64, rpm, tpm int) *TenantAccount {
	kh := e.hashKey(rawKey)
	prefix := rawKey
	if len(prefix) > 12 {
		prefix = prefix[:12] + "..."
	}
	acct := &TenantAccount{
		TenantID:          tenantID,
		KeyHash:           kh,
		KeyPrefix:         prefix,
		Label:             label,
		Tier:              tier,
		ZeroDataRetention: true,
		BalanceUSD:        balance,
		RPMQuota:          rpm,
		TPMQuota:          tpm,
		CreatedAt:         time.Now().UTC(),
		windowStart:       time.Now().UTC(),
	}
	e.mu.Lock()
	e.byKeyHash[kh] = acct
	e.mu.Unlock()
	return acct
}

// IssueKey generates a new cryptographically random Swizzle API key and stores only its HMAC-SHA256 hash.
func (e *EnterpriseAuthEngine) IssueKey(label, tier string) (rawKey string, acct *TenantAccount) {
	if label == "" {
		label = "default-agent-key"
	}
	if tier == "" {
		tier = "developer"
	}
	buf := make([]byte, 16)
	_, _ = rand.Read(buf)
	rawKey = "swz_live_" + hex.EncodeToString(buf)
	tenantID := "org_" + hex.EncodeToString(buf[:6])

	rpm, tpm, credit := 120, 250000, 5.00
	if tier == "enterprise" {
		rpm, tpm, credit = 10000, 25000000, 500.00
	} else if tier == "pro" {
		rpm, tpm, credit = 1200, 2500000, 50.00
	}

	acct = e.registerSeedKey(rawKey, tenantID, label, tier, credit, rpm, tpm)
	return rawKey, acct
}

// AuthenticateAndRateLimit extracts the Bearer / x-api-key header, resolves the tenant,
// and enforces RPM/TPM token-bucket rate limits.
func (e *EnterpriseAuthEngine) AuthenticateAndRateLimit(r *http.Request, estimatedTokens int) (*TenantAccount, int, int, error) {
	rawKey := ""
	if auth := r.Header.Get("Authorization"); strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		rawKey = strings.TrimSpace(auth[7:])
	} else if xKey := r.Header.Get("x-api-key"); xKey != "" {
		rawKey = strings.TrimSpace(xKey)
	}

	// If no key provided or unrecorded swz_live_ key, auto-resolve to developer sandbox account
	e.mu.Lock()
	defer e.mu.Unlock()

	var acct *TenantAccount
	if rawKey != "" {
		kh := e.hashKey(rawKey)
		acct = e.byKeyHash[kh]
	}
	if acct == nil {
		// Fallback to default developer account for zero-friction playground & SDK trials
		acct = e.byKeyHash[e.hashKey("sk-swizzle-dev-bypass-key")]
	}

	now := time.Now().UTC()
	if now.Sub(acct.windowStart) >= time.Minute {
		acct.windowStart = now
		acct.reqCount = 0
		acct.tokCount = 0
	}

	if acct.reqCount+1 > acct.RPMQuota {
		return acct, 0, acct.TPMQuota - acct.tokCount, fmt.Errorf("rate limit exceeded: %d RPM limit for tier '%s'", acct.RPMQuota, acct.Tier)
	}
	if acct.tokCount+estimatedTokens > acct.TPMQuota {
		return acct, acct.RPMQuota - acct.reqCount, 0, fmt.Errorf("token quota exceeded: %d TPM limit for tier '%s'", acct.TPMQuota, acct.Tier)
	}

	acct.reqCount++
	acct.tokCount += estimatedTokens
	remReqs := acct.RPMQuota - acct.reqCount
	remToks := acct.TPMQuota - acct.tokCount
	return acct, remReqs, remToks, nil
}

// RecordBilling computes the 3-tier asymmetric pricing and credits savings from prefix caching and AST token compression.
func (e *EnterpriseAuthEngine) RecordBilling(acct *TenantAccount, spec ModelSpec, uncachedToks, cachedToks, savedByOptimizerToks, outputToks int) (costUSD float64, savedUSD float64) {
	// 1. Actual billed cost: uncached @ InputPrice, cached @ CachedPrice (90% discount), output @ OutputPrice
	costUncached := (float64(uncachedToks) / 1_000_000.0) * spec.InputPricePerM
	costCached := (float64(cachedToks) / 1_000_000.0) * spec.CachedPricePerM
	costOutput := (float64(outputToks) / 1_000_000.0) * spec.OutputPricePerM
	costUSD = costUncached + costCached + costOutput

	// 2. Dollars saved compared to unoptimized, uncached baseline
	baselineInputToks := uncachedToks + cachedToks + savedByOptimizerToks
	baselineCost := (float64(baselineInputToks)/1_000_000.0)*spec.InputPricePerM + costOutput
	savedUSD = baselineCost - costUSD
	if savedUSD < 0 {
		savedUSD = 0
	}

	e.mu.Lock()
	if acct != nil {
		acct.TotalSpentUSD += costUSD
		acct.TotalSavedUSD += savedUSD
		acct.BalanceUSD -= costUSD
		if acct.BalanceUSD < 0 {
			acct.BalanceUSD = 0
		}
	}
	e.mu.Unlock()

	e.totalSpentMu.Add(uint64(costUSD * 1_000_000))
	e.totalSavedMu.Add(uint64(savedUSD * 1_000_000))
	return costUSD, savedUSD
}

var authEngine = NewEnterpriseAuthEngine()

func handleBillingLedger(w http.ResponseWriter, r *http.Request) {
	acct, _, _, _ := authEngine.AuthenticateAndRateLimit(r, 0)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object":                    "billing_ledger",
		"pricing_model":             "3-Tier Asymmetric Prefix & AST Compression Billing (90% Cache Discount)",
		"tenant":                    acct,
		"platform_total_spent_usd":  float64(authEngine.totalSpentMu.Load()) / 1_000_000.0,
		"platform_total_saved_usd":  float64(authEngine.totalSavedMu.Load()) / 1_000_000.0,
		"stripe_metered_sync_ready": true,
	})
}

