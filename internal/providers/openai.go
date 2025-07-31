package providers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Davincible/claude-code-open/internal/config"
)

type OpenAIProvider struct {
	Provider *config.Provider
}

func NewOpenAIProvider(provider *config.Provider) *OpenAIProvider {
	return &OpenAIProvider{
		Provider: provider,
	}
}

func (p *OpenAIProvider) Name() string {
	return p.Provider.Name
}

func (p *OpenAIProvider) SupportsStreaming() bool {
	return true
}

func (p *OpenAIProvider) GetEndpoint() string {
	return p.Provider.APIBase
}

func (p *OpenAIProvider) GetAPIKey() string {
	return p.Provider.GetAPIKey()
}

func (p *OpenAIProvider) IsStreaming(headers map[string][]string) bool {
	if contentType, ok := headers["Content-Type"]; ok {
		for _, ct := range contentType {
			if ct == ContentTypeEventStream || strings.Contains(ct, "stream") {
				return true
			}
		}
	}

	if transferEncoding, ok := headers["Transfer-Encoding"]; ok {
		for _, te := range transferEncoding {
			if te == TransferEncodingChunked {
				return true
			}
		}
	}

	return false
}

func (p *OpenAIProvider) TransformRequest(request []byte) ([]byte, error) {
	// OpenAI uses OpenAI format, so we need to transform Anthropic to OpenAI
	// We can reuse the logic from OpenRouter since both use OpenAI format
	return p.transformAnthropicToOpenAI(request)
}

func (p *OpenAIProvider) TransformResponse(response []byte) ([]byte, error) {
	// Transform OpenAI response to Anthropic format
	return p.convertOpenAIToAnthropic(response)
}

func (p *OpenAIProvider) TransformStream(chunk []byte, state *StreamState) ([]byte, error) {
	return p.convertOpenAIToAnthropicStream(chunk, state)
}


// Anthropic format structures
type anthropicResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Content      []anthropicContent `json:"content"`
	Model        string             `json:"model"`
	StopReason   *string            `json:"stop_reason,omitempty"`
	StopSequence *string            `json:"stop_sequence,omitempty"`
	Usage        *anthropicUsage    `json:"usage,omitempty"`
	Error        *anthropicError    `json:"error,omitempty"`
}

type anthropicContent struct {
	Type      string                 `json:"type"`
	Text      *string                `json:"text,omitempty"`
	ID        *string                `json:"id,omitempty"`
	Name      *string                `json:"name,omitempty"`
	Input     map[string]any `json:"input,omitempty"`
	ToolUseID *string                `json:"tool_use_id,omitempty"`
	Content   any            `json:"content,omitempty"`
	IsError   *bool                  `json:"is_error,omitempty"`
}

type anthropicUsage struct {
	InputTokens            int  `json:"input_tokens"`
	OutputTokens           int  `json:"output_tokens"`
	CacheReadInputTokens   *int `json:"cache_read_input_tokens,omitempty"`
	CacheCreateInputTokens *int `json:"cache_create_input_tokens,omitempty"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func (p *OpenAIProvider) convertOpenAIToAnthropic(openaiData []byte) ([]byte, error) {
	return ConvertToAnthropic(openaiData, p.mapOpenAIErrorType, p.convertToolCallID)
}

func (p *OpenAIProvider) convertStopReason(openaiReason string) *string {
	mapping := map[string]string{
		"stop":           "end_turn",
		"length":         "max_tokens",
		"tool_calls":     "tool_use",
		"function_call":  "tool_use",
		"content_filter": "stop_sequence",
		"null":           "end_turn",
	}

	if anthropicReason, exists := mapping[openaiReason]; exists {
		return &anthropicReason
	}

	defaultReason := "end_turn"

	return &defaultReason
}

func (p *OpenAIProvider) mapOpenAIErrorType(openaiType string) string {
	mapping := map[string]string{
		"invalid_request_error":    "invalid_request_error",
		"authentication_error":     "authentication_error",
		"permission_error":         "permission_error",
		"not_found_error":          "not_found_error",
		"rate_limit_error":         "rate_limit_error",
		"api_error":                "api_error",
		"overloaded_error":         "overloaded_error",
		"insufficient_quota_error": "billing_error",
	}

	if anthropicType, exists := mapping[openaiType]; exists {
		return anthropicType
	}

	return "api_error"
}

func (p *OpenAIProvider) convertOpenAIToAnthropicStream(openaiData []byte, state *StreamState) ([]byte, error) {
	return ConvertOpenAIStyleToAnthropicStream(openaiData, state, p, "OpenAI")
}

func (p *OpenAIProvider) createMessageStartEvent(messageID, model string, firstChunk map[string]any) map[string]any {
	usage := map[string]any{
		"input_tokens":  0,
		"output_tokens": 1,
	}

	if chunkUsage, ok := firstChunk["usage"].(map[string]any); ok {
		if promptTokens, ok := chunkUsage["prompt_tokens"]; ok {
			usage["input_tokens"] = promptTokens
		}

		if promptDetails, ok := chunkUsage["prompt_tokens_details"].(map[string]any); ok {
			if cachedTokens, ok := promptDetails["cached_tokens"]; ok {
				usage["cache_read_input_tokens"] = cachedTokens
			}
		}
	}

	return map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            messageID,
			"type":          "message",
			"role":          "assistant",
			"model":         model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         usage,
		},
	}
}

func (p *OpenAIProvider) formatSSEEvent(eventType string, data map[string]any) []byte {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return []byte("event: error\ndata: {\"error\":\"failed to marshal data\"}\n\n")
	}
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, string(jsonData)))
}

// handleTextContent processes text content streaming
func (p *OpenAIProvider) handleTextContent(content string, state *StreamState) []byte {
	var events []byte

	// Get or create text content block at index 0
	textIndex := p.getOrCreateTextBlock(state)
	contentBlock := state.ContentBlocks[textIndex]

	// Send content_block_start event if needed
	if !contentBlock.StartSent {
		events = append(events, p.createTextBlockStartEvent(textIndex)...)
		contentBlock.StartSent = true
	}

	// Send content_block_delta event
	events = append(events, p.createTextDeltaEvent(textIndex, content)...)

	return events
}

// handleToolCalls processes tool call streaming
func (p *OpenAIProvider) handleToolCalls(toolCalls []any, state *StreamState) []byte {
	var events []byte

	for _, toolCall := range toolCalls {
		if tcMap, ok := toolCall.(map[string]any); ok {
			toolCallEvents := p.handleSingleToolCall(tcMap, state)
			events = append(events, toolCallEvents...)
		}
	}

	return events
}

// handleSingleToolCall processes a single tool call
func (p *OpenAIProvider) handleSingleToolCall(toolCall map[string]any, state *StreamState) []byte {
	var events []byte

	// Parse tool call data
	toolCallData := p.parseToolCallData(toolCall)

	// Find or create content block
	contentBlockIndex := p.findOrCreateContentBlock(toolCallData, state)
	if contentBlockIndex == -1 {
		return events // Skip if couldn't find or create
	}

	contentBlock := state.ContentBlocks[contentBlockIndex]

	// Update content block with new data
	p.updateContentBlock(contentBlock, toolCallData)

	// Send content_block_start event if needed
	if !contentBlock.StartSent && p.shouldSendStartEvent(contentBlock) {
		events = append(events, p.createContentBlockStartEvent(contentBlockIndex, contentBlock)...)
		contentBlock.StartSent = true
	}

	// Handle argument streaming
	if toolCallData.Arguments != "" && toolCallData.Arguments != contentBlock.Arguments {
		newPart := p.calculateArgumentsDelta(toolCallData.Arguments, contentBlock.Arguments)
		contentBlock.Arguments = toolCallData.Arguments

		if newPart != "" {
			events = append(events, p.createInputDeltaEvent(contentBlockIndex, newPart)...)
		}
	}

	return events
}

// OpenAIToolCallData holds parsed tool call information for OpenAI provider
type OpenAIToolCallData struct {
	Index        int
	HasIndex     bool
	ID           string
	FunctionName string
	Arguments    string
}

// parseToolCallData extracts tool call information from OpenAI chunk
func (p *OpenAIProvider) parseToolCallData(toolCall map[string]any) OpenAIToolCallData {
	data := OpenAIToolCallData{}

	// Parse tool call index
	toolCallIndex, hasIndex := toolCall["index"].(float64)
	if !hasIndex {
		if idx, ok := toolCall["index"].(int); ok {
			toolCallIndex = float64(idx)
			hasIndex = true
		}
	}

	data.Index = int(toolCallIndex)
	data.HasIndex = hasIndex

	// Parse ID and function details
	data.ID, _ = toolCall["id"].(string)
	if function, ok := toolCall["function"].(map[string]any); ok {
		data.FunctionName, _ = function["name"].(string)
		data.Arguments, _ = function["arguments"].(string)
	}

	return data
}

// findOrCreateContentBlock locates existing content block or creates new one
func (p *OpenAIProvider) findOrCreateContentBlock(data OpenAIToolCallData, state *StreamState) int {
	// First try to find by tool call index
	if data.HasIndex {
		for blockIdx, block := range state.ContentBlocks {
			if block.Type == "tool_use" && block.ToolCallIndex == data.Index {
				return blockIdx
			}
		}
	}

	// Then try to find by ID
	if data.ID != "" {
		for blockIdx, block := range state.ContentBlocks {
			if block.Type == "tool_use" && block.ToolCallID == data.ID {
				return blockIdx
			}
		}
	}

	// Create new content block if we have an ID (first chunk)
	if data.ID != "" {
		contentBlockIndex := len(state.ContentBlocks)
		state.ContentBlocks[contentBlockIndex] = &ContentBlockState{
			Type:          "tool_use",
			ToolCallID:    data.ID,
			ToolCallIndex: data.Index,
			ToolName:      data.FunctionName,
			Arguments:     "",
		}

		return contentBlockIndex
	}

	return -1 // Couldn't find or create
}

// updateContentBlock updates content block with new tool call data
func (p *OpenAIProvider) updateContentBlock(block *ContentBlockState, data OpenAIToolCallData) {
	if data.FunctionName != "" {
		block.ToolName = data.FunctionName
	}
}

// shouldSendStartEvent determines if content_block_start event should be sent
func (p *OpenAIProvider) shouldSendStartEvent(block *ContentBlockState) bool {
	return block.ToolCallID != "" && block.ToolName != ""
}

// createContentBlockStartEvent creates content_block_start SSE event
func (p *OpenAIProvider) createContentBlockStartEvent(index int, block *ContentBlockState) []byte {
	claudeToolID := p.convertToolCallID(block.ToolCallID)

	contentBlockStartEvent := map[string]any{
		"type":  "content_block_start",
		"index": index,
		"content_block": map[string]any{
			"type":  "tool_use",
			"id":    claudeToolID,
			"name":  block.ToolName,
			"input": map[string]any{},
		},
	}

	return p.formatSSEEvent("content_block_start", contentBlockStartEvent)
}

// convertToolCallID converts OpenAI tool call ID to Claude format
func (p *OpenAIProvider) convertToolCallID(toolCallID string) string {
	if strings.HasPrefix(toolCallID, "toolu_") {
		return toolCallID
	}

	if strings.HasPrefix(toolCallID, "call_") {
		return "toolu_" + strings.TrimPrefix(toolCallID, "call_")
	}

	return "toolu_" + toolCallID
}

// calculateArgumentsDelta calculates the incremental part of arguments
func (p *OpenAIProvider) calculateArgumentsDelta(newArgs, oldArgs string) string {
	// Check if arguments are incremental (common case)
	if len(newArgs) > len(oldArgs) && strings.HasPrefix(newArgs, oldArgs) {
		return newArgs[len(oldArgs):] // Extract new part
	}
	// Non-incremental case - return entire new arguments
	return newArgs
}

// createInputDeltaEvent creates input_json_delta SSE event
func (p *OpenAIProvider) createInputDeltaEvent(index int, partialJSON string) []byte {
	inputDeltaEvent := map[string]any{
		"type":  "content_block_delta",
		"index": index,
		"delta": map[string]any{
			"type":         "input_json_delta",
			"partial_json": partialJSON,
		},
	}

	return p.formatSSEEvent("content_block_delta", inputDeltaEvent)
}

// getOrCreateTextBlock gets or creates text content block at index 0
func (p *OpenAIProvider) getOrCreateTextBlock(state *StreamState) int {
	textIndex := 0
	if _, exists := state.ContentBlocks[textIndex]; !exists {
		state.ContentBlocks[textIndex] = &ContentBlockState{
			Type: "text",
		}
	}

	return textIndex
}

// createTextBlockStartEvent creates content_block_start event for text
func (p *OpenAIProvider) createTextBlockStartEvent(index int) []byte {
	contentBlockStartEvent := map[string]any{
		"type":  "content_block_start",
		"index": index,
		"content_block": map[string]any{
			"type": "text",
			"text": "",
		},
	}

	return p.formatSSEEvent("content_block_start", contentBlockStartEvent)
}

// createTextDeltaEvent creates content_block_delta event for text
func (p *OpenAIProvider) createTextDeltaEvent(index int, text string) []byte {
	contentDeltaEvent := map[string]any{
		"type":  "content_block_delta",
		"index": index,
		"delta": map[string]any{
			"type": "text_delta",
			"text": text,
		},
	}

	return p.formatSSEEvent("content_block_delta", contentDeltaEvent)
}

// handleFinishReason processes finish reasons and sends appropriate events
func (p *OpenAIProvider) handleFinishReason(reason string, chunk map[string]any, state *StreamState) []byte {
	return HandleFinishReason(p, reason, chunk, state, func(chunk map[string]any) map[string]any {
		if usage, ok := chunk["usage"].(map[string]any); ok {
			return p.convertUsage(usage)
		}

		return nil
	})
}

// convertUsage handles usage information conversion
func (p *OpenAIProvider) convertUsage(usage map[string]any) map[string]any {
	anthropicUsage := make(map[string]any)

	// Map token fields
	if promptTokens, ok := usage["prompt_tokens"]; ok {
		anthropicUsage["input_tokens"] = promptTokens
	}

	if completionTokens, ok := usage["completion_tokens"]; ok {
		anthropicUsage["output_tokens"] = completionTokens
	}

	// Handle cached tokens
	if promptDetails, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		if cachedTokens, ok := promptDetails["cached_tokens"]; ok {
			anthropicUsage["cache_read_input_tokens"] = cachedTokens
		}
	}

	// Handle cache creation tokens (if available)
	if cacheCreationTokens, ok := usage["cache_creation_input_tokens"]; ok {
		anthropicUsage["cache_creation_input_tokens"] = cacheCreationTokens
	}

	return anthropicUsage
}

// transformAnthropicToOpenAI converts Anthropic/Claude format to OpenAI format for OpenAI
func (p *OpenAIProvider) transformAnthropicToOpenAI(anthropicRequest []byte) ([]byte, error) {
	return TransformAnthropicToOpenAI(anthropicRequest, p)
}

// Helper methods for transformAnthropicToOpenAI (similar to OpenRouter)
func (p *OpenAIProvider) removeAnthropicSpecificFields(request map[string]any) map[string]any {
	fieldsToRemove := []string{"cache_control"}

	if store, hasStore := request["store"]; !hasStore || store != true {
		fieldsToRemove = append(fieldsToRemove, "metadata")
	}

	cleaned := p.removeFieldsRecursively(request, fieldsToRemove).(map[string]any)

	if tools, hasTools := cleaned["tools"]; !hasTools || tools == nil {
		delete(cleaned, "tool_choice")
	} else if toolsArray, ok := tools.([]any); ok && len(toolsArray) == 0 {
		delete(cleaned, "tool_choice")
	}

	return cleaned
}

func (p *OpenAIProvider) removeFieldsRecursively(data any, fieldsToRemove []string) any {
	switch v := data.(type) {
	case map[string]any:
		result := make(map[string]any)

		for key, value := range v {
			shouldRemove := false

			for _, field := range fieldsToRemove {
				if key == field {
					shouldRemove = true
					break
				}
			}

			if !shouldRemove {
				result[key] = p.removeFieldsRecursively(value, fieldsToRemove)
			}
		}

		return result
	case []any:
		result := make([]any, len(v))
		for i, item := range v {
			result[i] = p.removeFieldsRecursively(item, fieldsToRemove)
		}

		return result
	default:
		return v
	}
}

func (p *OpenAIProvider) transformTools(tools []any) ([]any, error) {
	return TransformTools(tools)
}

func (p *OpenAIProvider) transformMessages(messages []any) []any {
	transformedMessages := make([]any, 0, len(messages))

	for _, message := range messages {
		if msgMap, ok := message.(map[string]any); ok {
			if role, ok := msgMap["role"].(string); ok {
				if role == "user" {
					if content, ok := msgMap["content"].([]any); ok {
						toolResultMessages := p.extractToolResults(content)
						if len(toolResultMessages) > 0 {
							transformedMessages = append(transformedMessages, toolResultMessages...)
							continue
						}
					}
				} else if role == RoleAssistant {
					if content, ok := msgMap["content"].([]any); ok {
						transformedMsg := p.transformAssistantMessage(msgMap, content)
						transformedMessages = append(transformedMessages, transformedMsg)

						continue
					}
				}
			}
		}

		transformedMessages = append(transformedMessages, message)
	}

	return transformedMessages
}

func (p *OpenAIProvider) extractToolResults(content []any) []any {
	var toolMessages []any

	for _, block := range content {
		if blockMap, ok := block.(map[string]any); ok {
			if blockType, ok := blockMap["type"].(string); ok && blockType == "tool_result" {
				if toolUseID, ok := blockMap["tool_use_id"].(string); ok {
					toolCallID := strings.Replace(toolUseID, "toolu_", "call_", 1)

					toolMessage := map[string]any{
						"role":         "tool",
						"tool_call_id": toolCallID,
						"content":      blockMap["content"],
					}
					toolMessages = append(toolMessages, toolMessage)
				}
			}
		}
	}

	if len(toolMessages) > 0 {
		return toolMessages
	}

	return nil
}

func (p *OpenAIProvider) transformAssistantMessage(msgMap map[string]any, content []any) map[string]any {
	return TransformAssistantMessage(msgMap, content)
}
