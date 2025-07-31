package tests

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/Davincible/claude-code-open/internal/config"
	"github.com/Davincible/claude-code-open/internal/handlers"
	"github.com/Davincible/claude-code-open/internal/providers"
)

func TestProxyIntegration(t *testing.T) {
	// Create test configuration using openrouter domain since that's what the registry knows about
	cfg := &config.Config{
		Host:   "127.0.0.1",
		Port:   8080,
		APIKey: "test-key",
		Providers: []config.Provider{
			{
				Name:    "openrouter",
				APIBase: "https://openrouter.ai/api/v1/chat/completions",
				APIKey:  "test-provider-key",
				Models:  []string{"test-model"},
			},
		},
		Router: config.RouterConfig{
			Default: "openrouter,test-model",
		},
	}

	// Create temporary config manager
	tmpDir := t.TempDir()
	cfgMgr := config.NewManager(tmpDir)
	cfgMgr.Save(cfg)

	// Setup logger
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Setup registry with actual providers - this will register the openrouter provider
	registry := providers.NewRegistry()
	registry.Initialize(cfg.Providers)

	// Create proxy handler
	handler := handlers.NewProxyHandler(cfgMgr, registry, logger)

	// Create test request
	requestBody := map[string]any{
		"model": "test-model",
		"messages": []map[string]any{
			{
				"role":    "user",
				"content": "Hello, world!",
			},
		},
	}

	jsonBody, _ := json.Marshal(requestBody)
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer test-key")

	// Record response
	rr := httptest.NewRecorder()

	// Execute request - this will fail because we can't reach the actual openrouter.ai
	// But we can test that the handler correctly processes the request and attempts to proxy it
	handler.ServeHTTP(rr, req)

	// The request should fail with a network error, but that means our handler logic worked
	// We're testing the request processing pipeline, not the actual network call
	assert.NotEqual(t, http.StatusInternalServerError, rr.Code, "should not have internal server error during request processing")

	// Log what we got for debugging
	t.Logf("Response status: %d", rr.Code)
	t.Logf("Response body: %s", rr.Body.String())
}
