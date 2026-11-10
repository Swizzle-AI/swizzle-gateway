package main

import (
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
)

var (
	metricTokensSavedByOpt atomic.Uint64
	metricSecretsScrubbed  atomic.Uint64
)

// handlePrometheusMetrics exports hardware-roofline, KV-cache, and token-optimizer telemetry
// in standard Prometheus exposition format for Grafana dashboards and KEDA autoscalers.
func handlePrometheusMetrics(w http.ResponseWriter, r *http.Request) {
	totalReq := ledger.totalRequests.Load()
	totalIn := ledger.totalInputToks.Load()
	cachedIn := ledger.cachedInputToks.Load()
	savedOpt := metricTokensSavedByOpt.Load()
	scrubbed := metricSecretsScrubbed.Load()
	savedUSD := float64(authEngine.totalSavedMu.Load()) / 1_000_000.0
	spentUSD := float64(authEngine.totalSpentMu.Load()) / 1_000_000.0

	hitRatio := 0.0
	if totalIn > 0 {
		hitRatio = float64(cachedIn) / float64(totalIn)
	}

	specStats := globalSpeculativeEngine.GetStats()

	var sb strings.Builder
	sb.WriteString("# HELP swizzle_requests_total Total inference requests processed by Swizzle Edge Gateway\n")
	sb.WriteString("# TYPE swizzle_requests_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_requests_total %d\n\n", totalReq))

	sb.WriteString("# HELP swizzle_input_tokens_total Total prompt tokens evaluated\n")
	sb.WriteString("# TYPE swizzle_input_tokens_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_input_tokens_total %d\n\n", totalIn))

	sb.WriteString("# HELP swizzle_cached_input_tokens_total Prompt tokens served directly from GPU HBM / Valkey KV prefix cache\n")
	sb.WriteString("# TYPE swizzle_cached_input_tokens_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_cached_input_tokens_total %d\n\n", cachedIn))

	sb.WriteString("# HELP swizzle_tokens_saved_by_optimizer_total Prompt tokens eliminated via lossless AST & boilerplate compression\n")
	sb.WriteString("# TYPE swizzle_tokens_saved_by_optimizer_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_tokens_saved_by_optimizer_total %d\n\n", savedOpt))

	sb.WriteString("# HELP swizzle_secrets_scrubbed_total High-entropy credentials redacted at the edge via ZDR DLP scrubber\n")
	sb.WriteString("# TYPE swizzle_secrets_scrubbed_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_secrets_scrubbed_total %d\n\n", scrubbed))

	sb.WriteString("# HELP swizzle_speculative_drafts_proposed_total Speculative tokens proposed via Swizzle-Draft N-gram trie\n")
	sb.WriteString("# TYPE swizzle_speculative_drafts_proposed_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_speculative_drafts_proposed_total %d\n\n", specStats.DraftsProposed))

	sb.WriteString("# HELP swizzle_speculative_drafts_accepted_total Speculative tokens accepted without extra forward passes\n")
	sb.WriteString("# TYPE swizzle_speculative_drafts_accepted_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_speculative_drafts_accepted_total %d\n\n", specStats.DraftsAccepted))

	sb.WriteString("# HELP swizzle_speculative_acceptance_ratio Ratio of speculative tokens accepted (0.0 to 1.0)\n")
	sb.WriteString("# TYPE swizzle_speculative_acceptance_ratio gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_speculative_acceptance_ratio %.4f\n\n", specStats.AcceptanceRate))

	sb.WriteString("# HELP swizzle_speculative_effective_speedup Generation speedup factor achieved via speculative decoding\n")
	sb.WriteString("# TYPE swizzle_speculative_effective_speedup gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_speculative_effective_speedup %.2f\n\n", specStats.EffectiveSpeedup))

	astEvals := globalASTDiffEngine.metrics.TotalDiffEvaluations.Load()
	astReused := globalASTDiffEngine.metrics.TotalReusedTokens.Load()
	astPrefill := globalASTDiffEngine.metrics.TotalPrefillTokens.Load()
	astRatio := 0.0
	if astReused+astPrefill > 0 {
		astRatio = float64(astReused) / float64(astReused+astPrefill)
	}

	sb.WriteString("# HELP swizzle_ast_tree_evaluations_total Total syntactic AST-diff evaluations executed\n")
	sb.WriteString("# TYPE swizzle_ast_tree_evaluations_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_ast_tree_evaluations_total %d\n\n", astEvals))

	sb.WriteString("# HELP swizzle_ast_tree_reused_tokens_total Prompt tokens preserved from re-prefill via non-linear AST virtual paging\n")
	sb.WriteString("# TYPE swizzle_ast_tree_reused_tokens_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_ast_tree_reused_tokens_total %d\n\n", astReused))

	sb.WriteString("# HELP swizzle_ast_tree_kv_savings_ratio Ratio of tokens preserved by AST-Tree virtual paging (0.0 to 1.0)\n")
	sb.WriteString("# TYPE swizzle_ast_tree_kv_savings_ratio gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_ast_tree_kv_savings_ratio %.4f\n\n", astRatio))

	pdStats := globalPDScheduler.GetStats()
	sb.WriteString("# HELP swizzle_pd_requests_scheduled_total Total inference requests processed by Swizzle-PD scheduler\n")
	sb.WriteString("# TYPE swizzle_pd_requests_scheduled_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_pd_requests_scheduled_total %d\n\n", pdStats.TotalScheduled))

	sb.WriteString("# HELP swizzle_pd_requests_disaggregated_total Requests split across dedicated Prefill and Decode workers\n")
	sb.WriteString("# TYPE swizzle_pd_requests_disaggregated_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_pd_requests_disaggregated_total %d\n\n", pdStats.TotalDisaggregated))

	sb.WriteString("# HELP swizzle_pd_disaggregated_ratio Ratio of requests disaggregated across distinct P/D nodes\n")
	sb.WriteString("# TYPE swizzle_pd_disaggregated_ratio gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_pd_disaggregated_ratio %.4f\n\n", pdStats.DisaggregatedRatio))

	sb.WriteString("# HELP swizzle_pd_kv_bytes_transferred_total Total bytes of KV activation tensors transferred via RDMA\n")
	sb.WriteString("# TYPE swizzle_pd_kv_bytes_transferred_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_pd_kv_bytes_transferred_total %d\n\n", pdStats.TotalKVBytesXfer))

	sb.WriteString("# HELP swizzle_pd_mean_transfer_latency_ms Mean transfer latency for KV activation blocks between P and D nodes\n")
	sb.WriteString("# TYPE swizzle_pd_mean_transfer_latency_ms gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_pd_mean_transfer_latency_ms %.2f\n\n", pdStats.MeanTransferTimeMs))

	sb.WriteString("# HELP swizzle_pd_active_prefill_nodes Number of healthy dedicated prefill workers online\n")
	sb.WriteString("# TYPE swizzle_pd_active_prefill_nodes gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_pd_active_prefill_nodes %d\n\n", pdStats.ActivePNodes))

	sb.WriteString("# HELP swizzle_pd_active_decode_nodes Number of healthy dedicated decode workers online\n")
	sb.WriteString("# TYPE swizzle_pd_active_decode_nodes gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_pd_active_decode_nodes %d\n\n", pdStats.ActiveDNodes))

	sb.WriteString("# HELP swizzle_prefix_cache_hit_ratio Ratio of input tokens served from prefix cache (0.0 to 1.0)\n")
	sb.WriteString("# TYPE swizzle_prefix_cache_hit_ratio gauge\n")
	sb.WriteString(fmt.Sprintf("swizzle_prefix_cache_hit_ratio %.4f\n\n", hitRatio))

	sb.WriteString("# HELP swizzle_platform_saved_usd_total Total USD saved by customers via 10x prefix caching and AST compression\n")
	sb.WriteString("# TYPE swizzle_platform_saved_usd_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_platform_saved_usd_total %.6f\n\n", savedUSD))

	sb.WriteString("# HELP swizzle_platform_spent_usd_total Total metered USD billed across all tenants\n")
	sb.WriteString("# TYPE swizzle_platform_spent_usd_total counter\n")
	sb.WriteString(fmt.Sprintf("swizzle_platform_spent_usd_total %.6f\n\n", spentUSD))

	sb.WriteString("# HELP swizzle_worker_healthy Health status of registered GPU/TPU silicon workers (1 = healthy, 0 = circuit open)\n")
	sb.WriteString("# TYPE swizzle_worker_healthy gauge\n")
	workers := prefixRouter.ListWorkers()
	for _, wk := range workers {
		val := 0
		if wk.Healthy {
			val = 1
		}
		sb.WriteString(fmt.Sprintf("swizzle_worker_healthy{worker_id=%q,hardware=%q,region=%q} %d\n", wk.ID, wk.Hardware, wk.Region, val))
	}
	sb.WriteString("\n# HELP swizzle_worker_kv_cache_usage_pct HBM KV-cache occupancy percentage per worker\n")
	sb.WriteString("# TYPE swizzle_worker_kv_cache_usage_pct gauge\n")
	for _, wk := range workers {
		sb.WriteString(fmt.Sprintf("swizzle_worker_kv_cache_usage_pct{worker_id=%q} %.2f\n", wk.ID, wk.KVCacheUsagePct))
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(sb.String()))
}

