package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/Davincible/claude-code-open/internal/config"
	"github.com/Davincible/claude-code-open/internal/providers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRemoveFieldsRecursively(t *testing.T) {
	testData := map[string]any{
		"keep": "this",
		"cache_control": map[string]any{
			"type": "ephemeral",
		},
		"nested": map[string]any{
			"keep_nested": "value",
			"cache_control": map[string]any{
				"type": "ephemeral",
			},
			"deep": map[string]any{
				"cache_control": "remove_me",
				"keep_deep":     "deep_value",
			},
		},
		"array": []any{
			map[string]any{
				"cache_control": "remove",
				"keep_array":    "array_value",
			},
		},
	}

	result, ok := providers.RemoveFieldsRecursively(testData, []string{"cache_control"}).(map[string]any)
	require.True(t, ok, "result should be a map")

	// Check root level
	assert.NotContains(t, result, "cache_control", "cache_control should be removed from root")
	assert.Equal(t, "this", result["keep"], "other fields should be preserved")

	// Check nested level
	nested, ok := result["nested"].(map[string]any)
	require.True(t, ok, "nested should be a map")
	assert.NotContains(t, nested, "cache_control", "cache_control should be removed from nested object")
	assert.Equal(t, "value", nested["keep_nested"], "other nested fields should be preserved")

	// Check deep nested level
	deep, ok := nested["deep"].(map[string]any)
	require.True(t, ok, "deep should be a map")
	assert.NotContains(t, deep, "cache_control", "cache_control should be removed from deep nested object")
	assert.Equal(t, "deep_value", deep["keep_deep"], "other deep nested fields should be preserved")

	// Check array level
	array, ok := result["array"].([]any)
	require.True(t, ok, "array should be a slice")
	require.Len(t, array, 1, "array should have 1 item")

	arrayItem, ok := array[0].(map[string]any)
	require.True(t, ok, "array item should be a map")
	assert.NotContains(t, arrayItem, "cache_control", "cache_control should be removed from array items")
	assert.Equal(t, "array_value", arrayItem["keep_array"], "other array item fields should be preserved")
}

func TestSelectModel_DynamicProviderSelection(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	handler := &ProxyHandler{logger: logger}

	routerConfig := &config.RouterConfig{
		Default:     "default,claude-3-5-sonnet",
		LongContext: "longcontext,claude-3-opus",
		Think:       "think,claude-3-5-sonnet",
		WebSearch:   "websearch,claude-3-5-sonnet:online",
		Background:  "background,claude-3-5-haiku",
	}

	testCases := []struct {
		name          string
		inputModel    string
		tokens        int
		expectedModel string
		expectedBody  string
		description   string
	}{
		{
			name:          "explicit provider with comma",
			inputModel:    "openrouter,anthropic/claude-sonnet-4",
			tokens:        1000,
			expectedModel: "openrouter,anthropic/claude-sonnet-4",
			expectedBody:  "anthropic/claude-sonnet-4",
			description:   "should use explicit provider/model when comma format is used",
		},
		{
			name:          "explicit provider overrides long context",
			inputModel:    "openrouter,anthropic/claude-sonnet-4",
			tokens:        70000, // This would normally trigger LongContext
			expectedModel: "openrouter,anthropic/claude-sonnet-4",
			expectedBody:  "anthropic/claude-sonnet-4",
			description:   "should prioritize explicit provider over automatic routing",
		},
		{
			name:          "automatic routing for long context",
			inputModel:    "claude-3-5-sonnet",
			tokens:        70000,
			expectedModel: "longcontext,claude-3-opus",
			expectedBody:  "claude-3-opus",
			description:   "should use long context routing for high token count",
		},
		{
			name:          "automatic routing for haiku background",
			inputModel:    "claude-3-5-haiku",
			tokens:        1000,
			expectedModel: "background,claude-3-5-haiku",
			expectedBody:  "claude-3-5-haiku",
			description:   "should use background routing for haiku model",
		},
		{
			name:          "passthrough for simple model",
			inputModel:    "claude-3-5-sonnet",
			tokens:        1000,
			expectedModel: "think,claude-3-5-sonnet",
			expectedBody:  "claude-3-5-sonnet",
			description:   "should use think routing when no other rules apply",
		},
		{
			name:          "online suffix preservation",
			inputModel:    "openrouter,anthropic/claude-sonnet-4:online",
			tokens:        1000,
			expectedModel: "openrouter,anthropic/claude-sonnet-4:online",
			expectedBody:  "anthropic/claude-sonnet-4:online",
			description:   "should preserve :online suffix for web search",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test request body
			requestBody := map[string]any{
				"model":      tc.inputModel,
				"messages":   []any{},
				"max_tokens": 100,
			}

			inputBody, err := json.Marshal(requestBody)
			require.NoError(t, err)

			// Call selectModel
			resultBody, selectedModel := handler.selectModel(inputBody, tc.tokens, routerConfig)

			// Verify selected model
			assert.Equal(t, tc.expectedModel, selectedModel, tc.description)

			// Verify request body has correct model
			var parsedResult map[string]any
			err = json.Unmarshal(resultBody, &parsedResult)
			require.NoError(t, err)

			assert.Equal(t, tc.expectedBody, parsedResult["model"], "request body should contain the final model name")
		})
	}
}

func TestSelectModel_NoModelProvided(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))
	handler := &ProxyHandler{logger: logger}

	routerConfig := &config.RouterConfig{
		Default: "default,claude-3-5-sonnet",
	}

	// Create test request body without model
	requestBody := map[string]any{
		"messages":   []any{},
		"max_tokens": 100,
	}

	inputBody, err := json.Marshal(requestBody)
	require.NoError(t, err)

	// Call selectModel
	resultBody, selectedModel := handler.selectModel(inputBody, 1000, routerConfig)

	// Should use default
	assert.Equal(t, "default,claude-3-5-sonnet", selectedModel)

	// Verify request body has correct model
	var parsedResult map[string]any
	err = json.Unmarshal(resultBody, &parsedResult)
	require.NoError(t, err)

	assert.Equal(t, "claude-3-5-sonnet", parsedResult["model"])
}

func TestHandleResponse_ErrorForwarding(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Create a mock provider that would normally transform responses
	mockProvider := &MockProvider{
		shouldTransform: true,
	}

	handler := &ProxyHandler{logger: logger}

	testCases := []struct {
		name            string
		statusCode      int
		responseBody    string
		shouldTransform bool
		description     string
	}{
		{
			name:            "error response not transformed",
			statusCode:      400,
			responseBody:    `{"error":{"type":"invalid_request_error","message":"Invalid model specified"}}`,
			shouldTransform: false,
			description:     "error responses should be forwarded without transformation",
		},
		{
			name:            "success response transformed",
			statusCode:      200,
			responseBody:    `{"id":"test","choices":[{"message":{"role":"assistant","content":"Hello"}}]}`,
			shouldTransform: true,
			description:     "success responses should be transformed",
		},
		{
			name:            "server error not transformed",
			statusCode:      500,
			responseBody:    `{"error":{"type":"internal_server_error","message":"Internal server error"}}`,
			shouldTransform: false,
			description:     "server errors should be forwarded without transformation",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset mock provider
			mockProvider.transformCalled = false

			// Create mock HTTP response
			resp := &http.Response{
				StatusCode: tc.statusCode,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(tc.responseBody)),
			}
			resp.Header.Set("Content-Type", "application/json")

			// Create response writer
			w := &MockResponseWriter{
				headers: make(http.Header),
				body:    &bytes.Buffer{},
			}

			// Call handleResponse
			handler.handleResponse(w, resp, mockProvider, 100)

			// Verify transformation was called only for success responses
			if tc.shouldTransform {
				assert.True(t, mockProvider.transformCalled, tc.description)
			} else {
				assert.False(t, mockProvider.transformCalled, tc.description)
			}

			// Verify status code is preserved
			assert.Equal(t, tc.statusCode, w.statusCode, "status code should be preserved")

			// Verify response body
			responseBody := w.body.String()
			if tc.shouldTransform {
				// For successful responses, we expect transformation
				assert.Contains(t, responseBody, "TRANSFORMED", "successful response should be transformed")
			} else {
				// For error responses, we expect original body
				assert.Equal(t, tc.responseBody, responseBody, "error response should be forwarded as-is")
			}
		})
	}
}

// Mock provider for testing
type MockProvider struct {
	transformCalled bool
	shouldTransform bool
}

func (m *MockProvider) Name() string                                 { return "mock" }
func (m *MockProvider) SupportsStreaming() bool                      { return true }
func (m *MockProvider) GetEndpoint() string                          { return "mock" }
func (m *MockProvider) GetAPIKey() string                          { return "mock-key" }
func (m *MockProvider) SetAPIKey(key string)                         {}
func (m *MockProvider) IsStreaming(headers map[string][]string) bool { return false }
func (m *MockProvider) TransformStream(chunk []byte, state *providers.StreamState) ([]byte, error) {
	return chunk, nil
}

func (m *MockProvider) TransformRequest(request []byte) ([]byte, error) {
	return request, nil
}

func (m *MockProvider) TransformResponse(response []byte) ([]byte, error) {
	m.transformCalled = true
	if m.shouldTransform {
		return []byte(`{"transformed": true, "original": "TRANSFORMED"}`), nil
	}
	return response, nil
}

// Mock response writer for testing
type MockResponseWriter struct {
	headers    http.Header
	body       *bytes.Buffer
	statusCode int
}

func (m *MockResponseWriter) Header() http.Header {
	return m.headers
}

func (m *MockResponseWriter) Write(data []byte) (int, error) {
	return m.body.Write(data)
}

func (m *MockResponseWriter) WriteHeader(statusCode int) {
	m.statusCode = statusCode
}

func TestHandleStreamingResponse_ErrorForwarding(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelError}))

	// Create a mock provider
	mockProvider := &MockProvider{shouldTransform: true}

	handler := &ProxyHandler{logger: logger}

	// Test error response body (simulating SSE error stream)
	errorStreamBody := `data: {"error":{"type":"invalid_request_error","message":"Invalid model specified"}}

`

	// Create mock HTTP response with error status
	resp := &http.Response{
		StatusCode: 400,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(errorStreamBody)),
	}
	resp.Header.Set("Content-Type", "text/event-stream")

	// Create response writer
	w := &MockResponseWriter{
		headers: make(http.Header),
		body:    &bytes.Buffer{},
	}

	// Call handleStreamingResponse
	handler.handleStreamingResponse(w, resp, mockProvider, 100)

	// Verify transformation was NOT called for error response
	assert.False(t, mockProvider.transformCalled, "error streaming responses should not be transformed")

	// Verify status code is preserved
	assert.Equal(t, 400, w.statusCode, "error status code should be preserved")

	// Verify response body contains original error data
	responseBody := w.body.String()
	assert.Contains(t, responseBody, "invalid_request_error", "error response should be forwarded as-is")
	assert.Contains(t, responseBody, "Invalid model specified", "error message should be preserved")
}
