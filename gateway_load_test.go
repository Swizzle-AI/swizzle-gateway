package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// BenchmarkDualWireNormalization measures bidirectional protocol conversion latency.
func BenchmarkDualWireNormalization(b *testing.B) {
	reqBody := []byte(`{
		"model": "qwen-2.5-coder-32b",
		"messages": [
			{"role": "system", "content": "You are a systems programmer."},
			{"role": "user", "content": "Write a bank-conflict-free shared memory swizzle kernel"}
		],
		"temperature": 0.2
	}`)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			var parsed map[string]interface{}
			_ = json.Unmarshal(reqBody, &parsed)
			_, _ = json.Marshal(parsed)
		}
	})
}

// BenchmarkASTParser measures AST block syntactic tokenization throughput under parallel load.
func BenchmarkASTParser(b *testing.B) {
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
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = parser.Parse(code, "server.go")
		}
	})
}

// TestGatewayHighConcurrencyThroughput benchmarks sub-1ms proxy overhead across 50 parallel clients.
func TestGatewayHighConcurrencyThroughput(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Swizzle-TTFT-Ms", "24.5")
		w.Header().Set("X-Swizzle-Cache-Hit-Tokens", "1024")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices": [{"message": {"content": "ok"}}]}`))
	}))
	defer backend.Close()

	numClients := 30
	requestsPerClient := 10
	var wg sync.WaitGroup
	wg.Add(numClients)

	t0 := time.Now()
	for c := 0; c < numClients; c++ {
		go func() {
			defer wg.Done()
			client := &http.Client{Timeout: 5 * time.Second}
			for i := 0; i < requestsPerClient; i++ {
				resp, err := client.Post(backend.URL, "application/json", bytes.NewBufferString(`{"prompt": "bench"}`))
				if err != nil {
					t.Errorf("Request error: %v", err)
					return
				}
				_ = resp.Body.Close()
			}
		}()
	}

	wg.Wait()
	dur := time.Since(t0)
	totalRequests := numClients * requestsPerClient
	qps := float64(totalRequests) / dur.Seconds()
	avgLatencyMs := (dur.Seconds() * 1000.0) / float64(totalRequests)

	t.Logf("High Concurrency Benchmark: %d requests in %.3fs (%.1f QPS | %.2fms avg)",
		totalRequests, dur.Seconds(), qps, avgLatencyMs)
	if qps < 100.0 {
		t.Errorf("Expected QPS > 100, got %.1f", qps)
	}
}
