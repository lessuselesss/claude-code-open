package handlers

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	
	"strings"

	"github.com/andybalholm/brotli"
	"github.com/pkoukk/tiktoken-go"

	"github.com/Davincible/claude-code-open/internal/config"
	"github.com/Davincible/claude-code-open/internal/providers"
)

type ProxyHandler struct {
	config   *config.Manager
	registry *providers.Registry
	logger   *slog.Logger
}

func NewProxyHandler(config *config.Manager, registry *providers.Registry, logger *slog.Logger) *ProxyHandler {
	return &ProxyHandler{
		config:   config,
		registry: registry,
		logger:   logger,
	}
}

func (h *ProxyHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	cfg := h.config.Get()

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		h.httpError(w, http.StatusBadRequest, "failed to read request body: %v", err)
		return
	}

	// Count input tokens
	inputTokens := h.countInputTokens(string(body))

	// Select model and transform request body
	transformedBody, modelName := h.selectModel(body, inputTokens, &cfg.Router)

	// Find provider for the model
	provider, providerConfig, err := h.findProvider(modelName, cfg)
	if err != nil {
		h.httpError(w, http.StatusBadRequest, "provider not found: %v", err)
		return
	}

	// Transform from Anthropic format to provider format
	finalBody, err := provider.TransformRequest(transformedBody)
	if err != nil {
		h.logger.Warn("Request transformation failed, using original", "error", err)

		finalBody = transformedBody
	}

	// Debug: Log request being sent to provider (truncated for readability)
	if len(finalBody) > 500 {
		h.logger.Debug("Sending request to provider", "provider", provider.Name(), "body_preview", string(finalBody[:500])+"...")
	} else {
		h.logger.Debug("Sending request to provider", "provider", provider.Name(), "body", string(finalBody))
	}

	// Build final endpoint URL (handle special cases like Gemini)
	finalURL := h.buildEndpointURL(provider, providerConfig.APIBase, modelName)

	// Create upstream request
	req, err := http.NewRequest(r.Method, finalURL, strings.NewReader(string(finalBody)))
	if err != nil {
		h.httpError(w, http.StatusInternalServerError, "failed to create upstream request: %v", err)
		return
	}

	// Copy headers and set auth
	req.Header = r.Header.Clone()
	apiKey := providerConfig.GetAPIKey()
	if apiKey != "" {
		h.setAuthHeader(req, provider, apiKey)
	}

	h.logger.Info("Proxying request",
		"provider", provider.Name(),
		"model", modelName,
		"url", finalURL,
		"input_tokens", inputTokens,
	)

	// Make upstream request
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		h.httpError(w, http.StatusBadGateway, "upstream request failed: %v", err)
		return
	}

	defer func() {
		if err := resp.Body.Close(); err != nil {
			h.logger.Warn("Failed to close response body", "error", err)
		}
	}()

	// Handle response based on streaming
	if provider.IsStreaming(resp.Header) {
		h.handleStreamingResponse(w, resp, provider, inputTokens)
	} else {
		h.handleResponse(w, resp, provider, inputTokens)
	}
}

func (h *ProxyHandler) handleStreamingResponse(w http.ResponseWriter, resp *http.Response, provider providers.Provider, inputTokens int) {
	// Handle decompression
	bodyReader, err := h.decompressReader(resp)
	if err != nil {
		h.httpError(w, http.StatusBadGateway, "decompression error: %v", err)
		return
	}

	if closer, ok := bodyReader.(io.Closer); ok {
		defer func() {
			if err := closer.Close(); err != nil {
				h.logger.Warn("Failed to close body reader", "error", err)
			}
		}()
	}

	// Set streaming headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Copy relevant headers
	h.copyHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)

	// For error responses, capture and print the body
	var errorBodyLines []string

	captureError := resp.StatusCode != http.StatusOK

	// Create scanner and state
	scanner := bufio.NewScanner(bodyReader)
	state := &providers.StreamState{}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Capture error response body
		if captureError && line != "" {
			errorBodyLines = append(errorBodyLines, line)
		}

		// Skip empty lines and comments
		if line == "" {
			if _, err := fmt.Fprint(w, "\n"); err != nil {
				h.logger.Error("Failed to write newline", "error", err)
				return
			}

			h.flushResponse(w)

			continue
		}

		if strings.HasPrefix(line, ": ") {
			continue // Skip SSE comments
		}

		// Handle [DONE] message
		if line == "data: [DONE]" {
			if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
				h.logger.Error("Failed to write DONE message", "error", err)
				return
			}

			h.flushResponse(w)

			break
		}

		// Process data lines
		if strings.HasPrefix(line, "data: ") {
			// For error responses, forward data as-is without transformation
			if captureError {
				if _, err := fmt.Fprintf(w, "%s\n\n", line); err != nil {
					h.logger.Error("Failed to write error response", "error", err)
					return
				}
			} else {
				jsonData := strings.TrimPrefix(line, "data: ")

				// Transform chunk through provider for successful responses
				events, err := provider.TransformStream([]byte(jsonData), state)
				if err != nil {
					h.logger.Error("Stream transformation error", "error", err)
					// Send original chunk on error
					if _, err := fmt.Fprintf(w, "%s\n\n", line); err != nil {
						h.logger.Error("Failed to write original chunk on transformation error", "error", err)
						return
					}
				} else {
					if len(events) > 0 {
						if _, err := w.Write(events); err != nil {
							h.logger.Error("Failed to write events", "error", err)
							return
						}
					}
				}
			}

			h.flushResponse(w)
		} else {
			// Pass through other SSE lines
			if _, err := fmt.Fprintf(w, "%s\n", line); err != nil {
				h.logger.Error("Failed to write SSE line", "error", err)
				return
			}

			h.flushResponse(w)
		}
	}

	if err := scanner.Err(); err != nil {
		h.logger.Error("Stream scanning error", "error", err)
	}

	// Print captured error response body
	if captureError && len(errorBodyLines) > 0 {
		fmt.Printf("\nUpstream streaming error response body:\n%s\n", strings.Join(errorBodyLines, "\n"))
	}

	h.logger.Info("Completed streaming response",
		"status", resp.StatusCode,
		"input_tokens", inputTokens,
	)
}

func (h *ProxyHandler) handleResponse(w http.ResponseWriter, resp *http.Response, provider providers.Provider, inputTokens int) {
	// Handle decompression
	bodyReader, err := h.decompressReader(resp)
	if err != nil {
		h.httpError(w, http.StatusBadGateway, "decompression error: %v", err)
		return
	}

	if closer, ok := bodyReader.(io.Closer); ok {
		defer func() {
			if closeErr := closer.Close(); closeErr != nil {
				h.logger.Warn("Failed to close body reader", "error", closeErr)
			}
		}()
	}

	// Read full response
	respBody, err := io.ReadAll(bodyReader)
	if err != nil {
		h.httpError(w, http.StatusBadGateway, "failed to read upstream response: %v", err)
		return
	}

	var finalBody []byte

	// For error responses, forward original response without transformation
	if resp.StatusCode != http.StatusOK {
		fmt.Printf("\nUpstream error response body:\n%s\n", string(respBody))
		finalBody = respBody
	} else {
		// Transform successful responses
		transformedBody, err := provider.TransformResponse(respBody)
		if err != nil {
			h.logger.Warn("Response transformation failed, using original", "error", err)

			finalBody = respBody
		} else {
			finalBody = transformedBody
		}
	}

	// Copy headers and send response
	h.copyHeaders(w, resp)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)

	if _, err := w.Write(finalBody); err != nil {
		h.logger.Error("Failed to write response body", "error", err)
	}

	h.logResponseTokens(finalBody, resp.StatusCode, inputTokens)
}

func (h *ProxyHandler) findProvider(modelName string, cfg *config.Config) (providers.Provider, *config.Provider, error) {
	parts := strings.SplitN(modelName, ",", 2)
	var providerName, actualModelName string

	if len(parts) > 1 {
		providerName = parts[0]
		actualModelName = parts[1]
	} else {
		actualModelName = modelName
	}

	// If provider name is not explicit, search for the model in all providers
	if providerName == "" {
		for _, p := range cfg.Providers {
			// Check both DefaultModels and Models lists
			allModels := append(p.DefaultModels, p.Models...)
			for _, m := range allModels {
				if m == actualModelName {
					providerName = p.Name
					h.logger.Debug("Inferred provider for model", "model", actualModelName, "provider", providerName)
					break
				}
			}
			if providerName != "" {
				break
			}
		}
	}

	// If still no provider, we have a problem
	if providerName == "" {
		// Fallback to default provider if model is not found in any provider list
		if cfg.Router.Default != "" {
			h.logger.Debug("Could not determine provider for model, falling back to default", "model", modelName, "default", cfg.Router.Default)
			defaultParts := strings.SplitN(cfg.Router.Default, ",", 2)
			if len(defaultParts) > 1 {
				providerName = defaultParts[0]
			}
		}

		if providerName == "" {
			return nil, nil, fmt.Errorf("could not determine provider for model '%s' and no default is set", modelName)
		}
	}

	// Now that we have a providerName, find its config
	var providerConfig *config.Provider
	for i, p := range cfg.Providers {
		if p.Name == providerName {
			providerConfig = &cfg.Providers[i]
			break
		}
	}

	// This should not happen if the logic above is correct, but as a safeguard:
	if providerConfig == nil {
		return nil, nil, fmt.Errorf("configuration for provider '%s' not found", providerName)
	}

	// Get the provider implementation from the registry
	provider, ok := h.registry.Get(providerName)
	if !ok {
		return nil, nil, fmt.Errorf("provider '%s' not found in registry", providerName)
	}

	// Use provider-specific API key if available, otherwise fallback to CCO_API_KEY
	
	

	

	return provider, providerConfig, nil
}

func (h *ProxyHandler) selectModel(inputBody []byte, tokens int, routerConfig *config.RouterConfig) ([]byte, string) {
	var modelBody map[string]any
	if err := json.Unmarshal(inputBody, &modelBody); err != nil {
		h.logger.Error("Failed to unmarshal request body for model selection", "error", err)
		return inputBody, routerConfig.Default
	}

	// Model selection logic
	var selectedModel string

	// Check if user provided explicit model in request
	if model, ok := modelBody["model"].(string); ok && len(model) > 0 {
		// If model contains comma (provider,model format), use it directly
		if strings.Contains(model, ",") {
			selectedModel = model
		} else {
			// Apply automatic routing logic for non-explicit provider requests
			if tokens > 60000 && routerConfig.LongContext != "" {
				selectedModel = routerConfig.LongContext
			} else if strings.HasPrefix(model, "claude-3-5-haiku") && routerConfig.Background != "" {
				selectedModel = routerConfig.Background
			} else if routerConfig.Think != "" {
				selectedModel = routerConfig.Think
			} else if routerConfig.WebSearch != "" {
				selectedModel = routerConfig.WebSearch
			} else {
				selectedModel = model
			}
		}
	} else {
		// No model specified, use default
		selectedModel = routerConfig.Default
	}

	// Update model in request body
	var finalModel string
	if parts := strings.SplitN(selectedModel, ",", 2); len(parts) > 1 {
		finalModel = parts[1]
	} else {
		finalModel = selectedModel
	}

	// Handle :online suffix for web search (preserve it for OpenRouter)
	// OpenRouter expects model:online format, so we keep it as-is
	modelBody["model"] = finalModel

	updatedBody, err := json.Marshal(modelBody)
	if err != nil {
		h.logger.Error("Failed to marshal updated request body", "error", err)
		return inputBody, selectedModel
	}

	return updatedBody, selectedModel
}

func (h *ProxyHandler) countInputTokens(text string) int {
	tke, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		h.logger.Error("Failed to get tiktoken encoding", "error", err)
		return 0
	}

	return len(tke.Encode(text, nil, nil))
}

func (h *ProxyHandler) decompressReader(resp *http.Response) (io.Reader, error) {
	var bodyReader io.Reader = resp.Body

	encoding := resp.Header.Get("Content-Encoding")

	switch encoding {
	case "gzip":
		gzipReader, err := gzip.NewReader(resp.Body)
		if err != nil {
			return nil, err
		}

		bodyReader = gzipReader
	case "br":
		bodyReader = brotli.NewReader(resp.Body)
	}

	return bodyReader, nil
}

func (h *ProxyHandler) copyHeaders(w http.ResponseWriter, resp *http.Response) {
	for key, values := range resp.Header {
		// Skip compression headers since we handle decompression
		if key == "Content-Encoding" || key == "Content-Length" {
			continue
		}

		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
}

func (h *ProxyHandler) flushResponse(w http.ResponseWriter) {
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (h *ProxyHandler) httpError(w http.ResponseWriter, code int, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	h.logger.Error("HTTP Error", "code", code, "message", msg)
	http.Error(w, msg, code)
}

// buildEndpointURL constructs the final endpoint URL for the provider
func (h *ProxyHandler) buildEndpointURL(provider providers.Provider, baseURL, modelName string) string {
	// Handle Gemini's special URL requirement
	if provider.Name() == "gemini" {
		// Extract actual model name from modelName (remove provider prefix if present)
		actualModel := modelName
		if parts := strings.SplitN(modelName, ",", 2); len(parts) > 1 {
			actualModel = parts[1]
		}

		// Gemini requires the model in the URL path
		// Format: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
		if strings.HasSuffix(baseURL, "/models") {
			return fmt.Sprintf("%s/%s:generateContent", baseURL, actualModel)
		} else if strings.Contains(baseURL, "/models/") {
			// Base URL already has a model specified, replace it
			baseIndex := strings.LastIndex(baseURL, "/models/")
			basePart := baseURL[:baseIndex+8] // Keep "/models/"

			return fmt.Sprintf("%s%s:generateContent", basePart, actualModel)
		}
		// Fallback to appending the model
		return fmt.Sprintf("%s/%s:generateContent", strings.TrimSuffix(baseURL, "/"), actualModel)
	}

	// For all other providers, use the base URL as-is
	return baseURL
}

// setAuthHeader sets the appropriate authentication header for the provider
func (h *ProxyHandler) setAuthHeader(req *http.Request, provider providers.Provider, apiKey string) {
	switch provider.Name() {
	case "gemini":
		// Gemini uses x-goog-api-key header
		req.Header.Set("x-goog-api-key", apiKey)
	default:
		// All other providers use Bearer token
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
}

func (h *ProxyHandler) logResponseTokens(respBody []byte, statusCode int, inputTokens int) {
	logFields := []any{
		"status", statusCode,
		"input_tokens", inputTokens,
	}

	// Try to extract output tokens from response
	var response map[string]any
	if err := json.Unmarshal(respBody, &response); err == nil {
		if usage, ok := response["usage"].(map[string]any); ok {
			if outputTokens, ok := usage["output_tokens"]; ok {
				logFields = append(logFields, "output_tokens", outputTokens)
			}
		}
	}

	if statusCode != http.StatusOK {
		h.logger.Error("Upstream error response", logFields...)
	} else {
		h.logger.Info("Successful response", logFields...)
	}
}
