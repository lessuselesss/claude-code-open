package providers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/Davincible/claude-code-open/internal/config"
)

type OpenRouterProvider struct {
	Provider *config.Provider
}

func NewOpenRouterProvider(provider *config.Provider) *OpenRouterProvider {
	return &OpenRouterProvider{
		Provider: provider,
	}
}

func (p *OpenRouterProvider) Name() string {
	return p.Provider.Name
}

func (p *OpenRouterProvider) SupportsStreaming() bool {
	return true
}

func (p *OpenRouterProvider) GetEndpoint() string {
	return p.Provider.APIBase
}

func (p *OpenRouterProvider) GetAPIKey() string {
	return p.Provider.GetAPIKey()
}

func (p *OpenRouterProvider) IsStreaming(headers map[string][]string) bool {
	if contentType, ok := headers["Content-Type"]; ok {
		for _, ct := range contentType {
			if ct == "text/event-stream" || strings.Contains(ct, "stream") {
				return true
			}
		}
	}

	if transferEncoding, ok := headers["Transfer-Encoding"]; ok {
		for _, te := range transferEncoding {
			if te == "chunked" {
				return true
			}
		}
	}

	return false
}

func (p *OpenRouterProvider) TransformRequest(request []byte) ([]byte, error) {
	// OpenRouter uses OpenAI format, so we need to transform from Anthropic to OpenAI
	return p.transformAnthropicToOpenAI(request)
}

func (p *OpenRouterProvider) TransformResponse(response []byte) ([]byte, error) {
	// This method transforms OpenRouter RESPONSES to Anthropic format
	return p.convertToAnthropic(response)
}

func (p *OpenRouterProvider) TransformStream(chunk []byte, state *StreamState) ([]byte, error) {
	var orChunk map[string]any
	if err := json.Unmarshal(chunk, &orChunk); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenRouter chunk: %w", err)
	}

	// Initialize content blocks map if needed
	if state.ContentBlocks == nil {
		state.ContentBlocks = make(map[int]*ContentBlockState)
	}

	var events []byte

	// Store message ID and model from first chunk
	if id, ok := orChunk["id"].(string); ok && state.MessageID == "" {
		state.MessageID = id
	}

	if model, ok := orChunk["model"].(string); ok && state.Model == "" {
		state.Model = model
	}

	// Handle choices array
	if choices, ok := orChunk["choices"].([]any); ok && len(choices) > 0 {
		if firstChoice, ok := choices[0].(map[string]any); ok {
			// Send message_start event if not sent yet
			if !state.MessageStartSent {
				messageStartEvent := p.createMessageStartEvent(state.MessageID, state.Model, orChunk)
				events = append(events, p.formatSSEEvent("message_start", messageStartEvent)...)
				state.MessageStartSent = true
			}

			// Handle delta content
			if delta, ok := firstChoice["delta"].(map[string]any); ok {
				// Check if we have tool calls - if so, prioritize them over text content
				if toolCalls, ok := delta["tool_calls"].([]any); ok {
					toolEvents := p.handleToolCalls(toolCalls, state)
					events = append(events, toolEvents...)
				} else if content, ok := delta["content"].(string); ok && content != "" {
					// Only handle text content if no tool calls are present
					textEvents := p.handleTextContent(content, state)
					events = append(events, textEvents...)
				}
			}

			// Handle finish_reason
			if finishReason, ok := firstChoice["finish_reason"]; ok && finishReason != nil {
				if reason, ok := finishReason.(string); ok {
					finishEvents := p.handleFinishReason(reason, orChunk, state)
					events = append(events, finishEvents...)
				}
			}
		}
	}

	return events, nil
}

// convertContent handles both text content and tool calls conversion
func (p *OpenRouterProvider) convertContent(message map[string]any) []map[string]any {
	var content []map[string]any

	// Handle text content
	if textContent, ok := message["content"].(string); ok && textContent != "" {
		content = append(content, map[string]any{
			"type": "text",
			"text": textContent,
		})
	}

	// Handle tool calls
	if toolCalls, ok := message["tool_calls"].([]any); ok {
		for _, toolCall := range toolCalls {
			if tcMap, ok := toolCall.(map[string]any); ok {
				// Convert tool call to Claude format
				toolContent := p.convertToolCall(tcMap)
				if toolContent != nil {
					content = append(content, toolContent)
				}
			}
		}
	}

	// Return at least empty text content if nothing else
	if len(content) == 0 {
		content = append(content, map[string]any{
			"type": "text",
			"text": "",
		})
	}

	return content
}

// convertToolCall converts OpenRouter tool call to Anthropic tool_use format
func (p *OpenRouterProvider) convertToolCall(toolCall map[string]any) map[string]any {
	function, ok := toolCall["function"].(map[string]any)
	if !ok {
		return nil
	}

	toolCallID, _ := toolCall["id"].(string)
	functionName, _ := function["name"].(string)
	arguments, _ := function["arguments"].(string)

	// Parse arguments JSON
	input := p.parseToolArguments(arguments)

	// Convert ID format: call_ -> toolu_
	claudeID := p.convertToolCallID(toolCallID)

	return map[string]any{
		"type":  "tool_use",
		"id":    claudeID,
		"name":  functionName,
		"input": input,
	}
}

// parseToolArguments parses JSON arguments or returns empty map
func (p *OpenRouterProvider) parseToolArguments(arguments string) map[string]any {
	if arguments == "" {
		return map[string]any{}
	}

	var input map[string]any
	if err := json.Unmarshal([]byte(arguments), &input); err != nil {
		// If parsing fails, use empty input
		return map[string]any{}
	}

	return input
}

// convertToolCallID converts OpenRouter tool call ID to Claude format
func (p *OpenRouterProvider) convertToolCallID(toolCallID string) string {
	if strings.HasPrefix(toolCallID, "toolu_") {
		return toolCallID
	}

	if strings.HasPrefix(toolCallID, "call_") {
		return "toolu_" + strings.TrimPrefix(toolCallID, "call_")
	}

	return "toolu_" + toolCallID
}

// convertAnnotations handles OpenRouter web search annotations
func (p *OpenRouterProvider) convertAnnotations(annotations any) any {
	// OpenRouter and Claude use the same annotation format according to docs
	// Just pass through, but we could add validation or transformation here if needed
	return annotations
}

// convertUsage handles enhanced usage information conversion
func (p *OpenRouterProvider) convertUsage(usage map[string]any) map[string]any {
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

	// Handle server tool use (web search) usage
	if serverToolUse, ok := usage["server_tool_use"].(map[string]any); ok {
		if webSearchRequests, ok := serverToolUse["web_search_requests"]; ok {
			// Add as additional usage info
			anthropicUsage["server_tool_use"] = map[string]any{
				"web_search_requests": webSearchRequests,
			}
		}
	}

	return anthropicUsage
}

func (p *OpenRouterProvider) convertToAnthropic(openRouterData []byte) ([]byte, error) {
	var orResponse map[string]any
	if err := json.Unmarshal(openRouterData, &orResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal OpenRouter response: %w", err)
	}

	// Create Anthropic response structure
	anthropicResponse := make(map[string]any)

	// Copy ID if present
	if id, ok := orResponse["id"]; ok {
		anthropicResponse["id"] = id
	}

	// Set type
	anthropicResponse["type"] = "message"

	// Extract role and content from choices[0].message
	if choices, ok := orResponse["choices"].([]any); ok && len(choices) > 0 {
		if firstChoice, ok := choices[0].(map[string]any); ok {
			if message, ok := firstChoice["message"].(map[string]any); ok {
				// Extract role
				if role, ok := message["role"]; ok {
					anthropicResponse["role"] = role
				}

				// Handle content and tool_calls
				content := p.convertContent(message)
				anthropicResponse["content"] = content

				// Handle annotations (web search results)
				if annotations, ok := message["annotations"]; ok {
					anthropicResponse["annotations"] = p.convertAnnotations(annotations)
				}
			}

			// Map finish_reason to stop_reason
			if finishReason, ok := firstChoice["finish_reason"]; ok {
				anthropicResponse["stop_reason"] = p.convertStopReason(fmt.Sprintf("%v", finishReason))
			}
		}
	}

	// Copy model if present
	if model, ok := orResponse["model"]; ok {
		anthropicResponse["model"] = model
	}

	// Transform usage object with enhanced handling
	if usage, ok := orResponse["usage"].(map[string]any); ok {
		anthropicUsage := p.convertUsage(usage)
		anthropicResponse["usage"] = anthropicUsage
	}

	// Default values
	if _, ok := anthropicResponse["stop_reason"]; !ok {
		anthropicResponse["stop_reason"] = nil
	}

	if _, ok := anthropicResponse["stop_sequence"]; !ok {
		anthropicResponse["stop_sequence"] = nil
	}

	// Remove any tool_choice field that might be present in OpenRouter responses
	// Claude response format doesn't include tool_choice (only in requests)
	delete(anthropicResponse, "tool_choice")

	return json.Marshal(anthropicResponse)
}

func (p *OpenRouterProvider) convertStopReason(openaiReason string) *string {
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

func (p *OpenRouterProvider) createMessageStartEvent(messageID, model string, firstChunk map[string]any) map[string]any {
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

func (p *OpenRouterProvider) formatSSEEvent(eventType string, data map[string]any) []byte {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return []byte("event: error\ndata: {\"error\":\"failed to marshal data\"}\n\n")
	}
	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, string(jsonData)))
}

// handleTextContent processes text content streaming
func (p *OpenRouterProvider) handleTextContent(content string, state *StreamState) []byte {
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
func (p *OpenRouterProvider) handleToolCalls(toolCalls []any, state *StreamState) []byte {
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
func (p *OpenRouterProvider) handleSingleToolCall(toolCall map[string]any, state *StreamState) []byte {
	var events []byte

	// Parse tool call data using helper
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

// ToolCallData holds parsed tool call information
type ToolCallData struct {
	Index        int
	HasIndex     bool
	ID           string
	FunctionName string
	Arguments    string
}

// parseToolCallData extracts tool call information from OpenRouter chunk
func (p *OpenRouterProvider) parseToolCallData(toolCall map[string]any) ToolCallData {
	data := ToolCallData{}

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

	// Tool call data parsed successfully

	return data
}

// findOrCreateContentBlock locates existing content block or creates new one
func (p *OpenRouterProvider) findOrCreateContentBlock(data ToolCallData, state *StreamState) int {
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
func (p *OpenRouterProvider) updateContentBlock(block *ContentBlockState, data ToolCallData) {
	if data.FunctionName != "" {
		block.ToolName = data.FunctionName
	}
}

// shouldSendStartEvent determines if content_block_start event should be sent
func (p *OpenRouterProvider) shouldSendStartEvent(block *ContentBlockState) bool {
	return block.ToolCallID != "" && block.ToolName != ""
}

// createContentBlockStartEvent creates content_block_start SSE event
func (p *OpenRouterProvider) createContentBlockStartEvent(index int, block *ContentBlockState) []byte {
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

// calculateArgumentsDelta calculates the incremental part of arguments
func (p *OpenRouterProvider) calculateArgumentsDelta(newArgs, oldArgs string) string {
	// Check if arguments are incremental (common case)
	if len(newArgs) > len(oldArgs) && strings.HasPrefix(newArgs, oldArgs) {
		return newArgs[len(oldArgs):] // Extract new part
	}
	// Non-incremental case - return entire new arguments
	return newArgs
}

// createInputDeltaEvent creates input_json_delta SSE event
func (p *OpenRouterProvider) createInputDeltaEvent(index int, partialJSON string) []byte {
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
func (p *OpenRouterProvider) getOrCreateTextBlock(state *StreamState) int {
	textIndex := 0
	if _, exists := state.ContentBlocks[textIndex]; !exists {
		state.ContentBlocks[textIndex] = &ContentBlockState{
			Type: "text",
		}
	}

	return textIndex
}

// createTextBlockStartEvent creates content_block_start event for text
func (p *OpenRouterProvider) createTextBlockStartEvent(index int) []byte {
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
func (p *OpenRouterProvider) createTextDeltaEvent(index int, text string) []byte {
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
func (p *OpenRouterProvider) handleFinishReason(reason string, orChunk map[string]any, state *StreamState) []byte {
	return HandleFinishReason(p, reason, orChunk, state, func(chunk map[string]any) map[string]any {
		if usage, ok := chunk["usage"].(map[string]any); ok {
			return p.convertUsage(usage)
		}

		return nil
	})
}

// transformAnthropicToOpenAI converts Anthropic/Claude format to OpenAI format for OpenRouter
func (p *OpenRouterProvider) transformAnthropicToOpenAI(anthropicRequest []byte) ([]byte, error) {
	return TransformAnthropicToOpenAI(anthropicRequest, p)
}

// removeAnthropicSpecificFields removes fields that OpenAI doesn't support
func (p *OpenRouterProvider) removeAnthropicSpecificFields(request map[string]any) map[string]any {
	// Remove Claude/Anthropic-specific fields that OpenAI/OpenRouter don't support
	fieldsToRemove := []string{"cache_control"}

	// Remove metadata if store is not enabled (OpenAI requirement)
	if store, hasStore := request["store"]; !hasStore || store != true {
		fieldsToRemove = append(fieldsToRemove, "metadata")
	}

	cleaned := p.removeFieldsRecursively(request, fieldsToRemove).(map[string]any)

	// Handle tool_choice logic: only remove if no tools are present, tools is null, or tools is empty array
	if tools, hasTools := cleaned["tools"]; !hasTools || tools == nil {
		delete(cleaned, "tool_choice")
	} else if toolsArray, ok := tools.([]any); ok && len(toolsArray) == 0 {
		delete(cleaned, "tool_choice")
	}

	return cleaned
}

// removeFieldsRecursively removes specified fields from a nested structure
func (p *OpenRouterProvider) removeFieldsRecursively(data any, fieldsToRemove []string) any {
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

// transformTools converts Claude tools to OpenAI format
func (p *OpenRouterProvider) transformTools(tools []any) ([]any, error) {
	return TransformTools(tools)
}

// transformMessages converts Anthropic messages to OpenAI format
func (p *OpenRouterProvider) transformMessages(messages []any) []any {
	transformedMessages := make([]any, 0, len(messages))

	for _, message := range messages {
		if msgMap, ok := message.(map[string]any); ok {
			// Check role-specific transformations
			if role, ok := msgMap["role"].(string); ok {
				if role == "user" {
					// Transform user messages with tool_result blocks to OpenAI tool message format
					if content, ok := msgMap["content"].([]any); ok {
						toolResultMessages := p.extractToolResults(content)
						if len(toolResultMessages) > 0 {
							transformedMessages = append(transformedMessages, toolResultMessages...)
							continue // Skip the original message as we've replaced it
						}
					}
				} else if role == "assistant" {
					// Transform assistant messages with tool_use blocks to OpenAI tool_calls format
					if content, ok := msgMap["content"].([]any); ok {
						transformedMsg := p.transformAssistantMessage(msgMap, content)
						transformedMessages = append(transformedMessages, transformedMsg)

						continue
					}
				}
			}
		}

		// Default: keep message as-is
		transformedMessages = append(transformedMessages, message)
	}

	return transformedMessages
}

// extractToolResults extracts tool_result blocks and converts them to OpenAI tool messages
func (p *OpenRouterProvider) extractToolResults(content []any) []any {
	var toolMessages []any

	for _, block := range content {
		if blockMap, ok := block.(map[string]any); ok {
			if blockType, ok := blockMap["type"].(string); ok && blockType == MessageTypeToolResult {
				// Convert tool_result to OpenAI tool message format
				if toolUseID, ok := blockMap["tool_use_id"].(string); ok {
					// Convert Claude tool ID format to OpenAI format
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

	return nil // No tool results found
}

// transformAssistantMessage converts assistant messages with tool_use to tool_calls format
func (p *OpenRouterProvider) transformAssistantMessage(msgMap map[string]any, content []any) map[string]any {
	return TransformAssistantMessage(msgMap, content)
}
