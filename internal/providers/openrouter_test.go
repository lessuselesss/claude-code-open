package providers

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Davincible/claude-code-open/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenRouterProvider_TransformRequest(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})

	// Test Anthropic to OpenAI/OpenRouter request transformation
	anthropicRequest := map[string]any{
		"model":      "claude-3-5-sonnet",
		"system":     "You are a helpful assistant",
		"max_tokens": 100,
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "Hello, world!",
			},
		},
		"tools": []any{
			map[string]any{
				"name":        "get_weather",
				"description": "Get current weather",
				"input_schema": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "City name",
						},
					},
					"required": []string{"location"},
				},
			},
		},
		"tool_choice": "auto",
	}

	anthropicJSON, err := json.Marshal(anthropicRequest)
	require.NoError(t, err)

	result, err := provider.TransformRequest(anthropicJSON)
	require.NoError(t, err)

	var openrouterReq map[string]any
	err = json.Unmarshal(result, &openrouterReq)
	require.NoError(t, err)

	// Verify system message was moved to messages array (OpenAI format)
	assert.NotContains(t, openrouterReq, "system", "system field should be removed from root")
	messages, ok := openrouterReq["messages"].([]any)
	require.True(t, ok, "messages should be an array")
	require.Len(t, messages, 2, "should have system + user message")

	systemMsg := messages[0].(map[string]any)
	assert.Equal(t, "system", systemMsg["role"])
	assert.Equal(t, "You are a helpful assistant", systemMsg["content"])

	// Verify max_tokens -> max_completion_tokens transformation
	assert.NotContains(t, openrouterReq, "max_tokens", "max_tokens should be converted")
	assert.Equal(t, float64(100), openrouterReq["max_completion_tokens"], "should have max_completion_tokens")

	// Verify tools transformation to OpenAI format
	tools, ok := openrouterReq["tools"].([]any)
	require.True(t, ok, "tools should be an array")
	require.Len(t, tools, 1, "should have one tool")

	tool := tools[0].(map[string]any)
	assert.Equal(t, "function", tool["type"])
	function := tool["function"].(map[string]any)
	assert.Equal(t, "get_weather", function["name"])
	assert.Contains(t, function, "parameters", "should have parameters not input_schema")

	// Verify tool_choice is preserved
	assert.Equal(t, "auto", openrouterReq["tool_choice"])
}

func TestOpenRouterProvider_Transform(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})

	// Test OpenRouter response format
	openRouterResponse := map[string]any{
		"id":     "chatcmpl-123",
		"object": "chat.completion",
		"model":  "anthropic/claude-3.5-sonnet",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Hello! How can I help you today?",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     25,
			"completion_tokens": 8,
			"total_tokens":      33,
			"prompt_tokens_details": map[string]any{
				"cached_tokens": 10,
			},
		},
	}

	inputData, _ := json.Marshal(openRouterResponse)
	result, err := provider.TransformResponse(inputData)

	require.NoError(t, err, "transform should not fail")

	var anthropicResponse map[string]any
	err = json.Unmarshal(result, &anthropicResponse)
	require.NoError(t, err, "should be able to parse result")

	// Verify Anthropic format
	assert.Equal(t, "message", anthropicResponse["type"], "should have message type")
	assert.Equal(t, "assistant", anthropicResponse["role"], "should have assistant role")

	// Check content format
	content, ok := anthropicResponse["content"].([]any)
	require.True(t, ok, "content should be an array")
	require.NotEmpty(t, content, "content should not be empty")

	firstContent, ok := content[0].(map[string]any)
	require.True(t, ok, "first content should be a map")
	assert.Equal(t, "text", firstContent["type"], "content type should be text")
	assert.Equal(t, "Hello! How can I help you today?", firstContent["text"], "text content should match")

	// Check usage transformation
	usage, ok := anthropicResponse["usage"].(map[string]any)
	require.True(t, ok, "usage should be an object")
	assert.Equal(t, float64(25), usage["input_tokens"], "input_tokens should match")
	assert.Equal(t, float64(8), usage["output_tokens"], "output_tokens should match")
	assert.Equal(t, float64(10), usage["cache_read_input_tokens"], "cache_read_input_tokens should match")
}

func TestOpenRouterProvider_SupportsStreaming(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})
	assert.True(t, provider.SupportsStreaming(), "OpenRouter should support streaming")
}

func TestOpenRouterProvider_Name(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})
	assert.Equal(t, "openrouter", provider.Name(), "provider name should be openrouter")
}

func TestOpenRouterProvider_IsStreaming(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})

	testCases := []struct {
		name     string
		headers  map[string][]string
		expected bool
	}{
		{
			name: "content-type event-stream",
			headers: map[string][]string{
				"Content-Type": {"text/event-stream"},
			},
			expected: true,
		},
		{
			name: "transfer-encoding chunked",
			headers: map[string][]string{
				"Transfer-Encoding": {"chunked"},
			},
			expected: true,
		},
		{
			name: "no streaming headers",
			headers: map[string][]string{
				"Content-Type": {"application/json"},
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := provider.IsStreaming(tc.headers)
			assert.Equal(t, tc.expected, result, "streaming detection should match expected")
		})
	}
}

func TestOpenRouterProvider_TransformStream(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})
	state := &StreamState{}

	// Test streaming chunk
	chunk := map[string]any{
		"id":    "chatcmpl-123",
		"model": "anthropic/claude-3.5-sonnet",
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{
					"content": "Hello",
				},
			},
		},
	}

	chunkData, _ := json.Marshal(chunk)
	result, err := provider.TransformStream(chunkData, state)

	require.NoError(t, err, "TransformStream should not fail")

	// Verify result contains SSE events
	resultStr := string(result)
	assert.Contains(t, resultStr, "event: message_start", "should contain message_start event")
	assert.Contains(t, resultStr, "event: content_block_start", "should contain content_block_start event")
	assert.Contains(t, resultStr, "event: content_block_delta", "should contain content_block_delta event")
	assert.Contains(t, resultStr, "text_delta", "should contain text_delta in response")
}

func TestOpenRouterProvider_ConvertStopReason(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})

	testCases := []struct {
		openaiReason    string
		anthropicReason string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"function_call", "tool_use"},
		{"content_filter", "stop_sequence"},
		{"null", "end_turn"},
		{"unknown", "end_turn"}, // default case
	}

	for _, tc := range testCases {
		t.Run(tc.openaiReason, func(t *testing.T) {
			result := provider.convertStopReason(tc.openaiReason)
			require.NotNil(t, result, "result should not be nil")
			assert.Equal(t, tc.anthropicReason, *result, "stop reason should be converted correctly")
		})
	}
}

func TestOpenRouterProvider_ToolCallsTransform(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})

	// Test OpenRouter response with tool calls
	openRouterResponse := map[string]any{
		"id":    "chatcmpl-123",
		"model": "anthropic/claude-3.5-sonnet",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role": "assistant",
					"tool_calls": []any{
						map[string]any{
							"id":   "call_abc123",
							"type": "function",
							"function": map[string]any{
								"name":      "get_weather",
								"arguments": "{\"location\":\"San Francisco\",\"unit\":\"celsius\"}",
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     50,
			"completion_tokens": 25,
		},
	}

	inputData, _ := json.Marshal(openRouterResponse)
	result, err := provider.TransformResponse(inputData)

	require.NoError(t, err, "transform should not fail")

	var anthropicResponse map[string]any
	err = json.Unmarshal(result, &anthropicResponse)
	require.NoError(t, err, "should be able to parse result")

	// Check content contains tool_use
	content, ok := anthropicResponse["content"].([]any)
	require.True(t, ok, "content should be an array")
	require.NotEmpty(t, content, "content should not be empty")

	toolUse, ok := content[0].(map[string]any)
	require.True(t, ok, "first content should be a map")
	assert.Equal(t, "tool_use", toolUse["type"], "content type should be tool_use")
	assert.Equal(t, "toolu_abc123", toolUse["id"], "tool ID should be converted")
	assert.Equal(t, "get_weather", toolUse["name"], "tool name should match")

	input, ok := toolUse["input"].(map[string]any)
	require.True(t, ok, "input should be a map")
	assert.Equal(t, "San Francisco", input["location"], "location parameter should match")
	assert.Equal(t, "celsius", input["unit"], "unit parameter should match")

	// Check stop reason
	if stopReasonPtr, ok := anthropicResponse["stop_reason"].(*string); ok {
		assert.Equal(t, "tool_use", *stopReasonPtr, "stop_reason should be tool_use")
	} else if stopReasonStr, ok := anthropicResponse["stop_reason"].(string); ok {
		assert.Equal(t, "tool_use", stopReasonStr, "stop_reason should be tool_use")
	} else {
		t.Fatalf("stop_reason has unexpected type: %T", anthropicResponse["stop_reason"])
	}
}

func TestOpenRouterProvider_WebSearchAnnotations(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})

	// Test OpenRouter response with web search annotations
	openRouterResponse := map[string]any{
		"id":    "chatcmpl-123",
		"model": "anthropic/claude-3.5-sonnet:online",
		"choices": []any{
			map[string]any{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": "Based on my search, here's what I found about the weather.",
					"annotations": []any{
						map[string]any{
							"type":  "web_search",
							"query": "current weather San Francisco",
							"results": []any{
								map[string]any{
									"title": "Weather in San Francisco",
									"url":   "https://weather.com/sf",
								},
							},
						},
					},
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     30,
			"completion_tokens": 15,
			"server_tool_use": map[string]any{
				"web_search_requests": 1,
			},
		},
	}

	inputData, _ := json.Marshal(openRouterResponse)
	result, err := provider.TransformResponse(inputData)

	require.NoError(t, err, "transform should not fail")

	var anthropicResponse map[string]any
	err = json.Unmarshal(result, &anthropicResponse)
	require.NoError(t, err, "should be able to parse result")

	// Check annotations are preserved
	annotations, ok := anthropicResponse["annotations"].([]any)
	require.True(t, ok, "annotations should be preserved")
	require.NotEmpty(t, annotations, "annotations should not be empty")

	annotation, ok := annotations[0].(map[string]any)
	require.True(t, ok, "annotation should be a map")
	assert.Equal(t, "web_search", annotation["type"], "annotation type should match")

	// Check usage includes server tool use
	usage, ok := anthropicResponse["usage"].(map[string]any)
	require.True(t, ok, "usage should exist")

	serverToolUse, ok := usage["server_tool_use"].(map[string]any)
	require.True(t, ok, "server_tool_use should be preserved")
	assert.Equal(t, float64(1), serverToolUse["web_search_requests"], "web_search_requests should match")
}

func TestOpenRouterProvider_StreamingToolCalls(t *testing.T) {
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})
	state := &StreamState{}

	// Test first chunk with tool call start
	chunk1 := map[string]any{
		"id":    "chatcmpl-123",
		"model": "anthropic/claude-3.5-sonnet",
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{
					"tool_calls": []any{
						map[string]any{
							"id":   "call_abc123",
							"type": "function",
							"function": map[string]any{
								"name":      "ls",
								"arguments": "",
							},
						},
					},
				},
			},
		},
	}

	chunkData1, _ := json.Marshal(chunk1)
	result1, err := provider.TransformStream(chunkData1, state)
	require.NoError(t, err, "first chunk should not fail")

	// Test second chunk with tool arguments
	chunk2 := map[string]any{
		"id":    "chatcmpl-123",
		"model": "anthropic/claude-3.5-sonnet",
		"choices": []any{
			map[string]any{
				"index": 0,
				"delta": map[string]any{
					"tool_calls": []any{
						map[string]any{
							"id":   "call_abc123",
							"type": "function",
							"function": map[string]any{
								"name":      "ls",
								"arguments": "{\"path\":\"/home\"}",
							},
						},
					},
				},
			},
		},
	}

	chunkData2, _ := json.Marshal(chunk2)
	result2, err := provider.TransformStream(chunkData2, state)
	require.NoError(t, err, "second chunk should not fail")

	// Verify result contains proper SSE events
	resultStr1 := string(result1)
	resultStr2 := string(result2)
	combinedResult := resultStr1 + resultStr2

	assert.Contains(t, combinedResult, "event: message_start", "should contain message_start event")
	assert.Contains(t, combinedResult, "event: content_block_start", "should contain content_block_start event")
	assert.Contains(t, combinedResult, "\"type\":\"tool_use\"", "should contain tool_use type")
	assert.Contains(t, combinedResult, "\"id\":\"toolu_abc123\"", "should convert tool call ID")
	assert.Contains(t, combinedResult, "\"name\":\"ls\"", "should contain tool name")
	assert.Contains(t, combinedResult, "input_json_delta", "should contain input_json_delta")

	// Ensure no duplicate content_block_start events
	startEventCount := strings.Count(combinedResult, "content_block_start")
	assert.Equal(t, 2, startEventCount, "should have exactly 2 content_block_start events (message_start + tool_use)")
}
