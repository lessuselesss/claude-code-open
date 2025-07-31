package providers

import (
	"strings"

	"github.com/Davincible/claude-code-open/internal/config"
)

type AnthropicProvider struct {
	Provider *config.Provider
}

func NewAnthropicProvider(provider *config.Provider) *AnthropicProvider {
	return &AnthropicProvider{
		Provider: provider,
	}
}

func (p *AnthropicProvider) Name() string {
	return p.Provider.Name
}

func (p *AnthropicProvider) SupportsStreaming() bool {
	return true
}

func (p *AnthropicProvider) GetEndpoint() string {
	return p.Provider.APIBase
}

func (p *AnthropicProvider) GetAPIKey() string {
	return p.Provider.GetAPIKey()
}

func (p *AnthropicProvider) IsStreaming(headers map[string][]string) bool {
	if contentType, ok := headers["Content-Type"]; ok {
		for _, ct := range contentType {
			if ct == "text/event-stream" || strings.Contains(ct, "stream") {
				return true
			}
		}
	}

	return false
}

func (p *AnthropicProvider) TransformRequest(request []byte) ([]byte, error) {
	// Anthropic format doesn't need request transformation
	return request, nil
}

func (p *AnthropicProvider) TransformResponse(response []byte) ([]byte, error) {
	// Anthropic format doesn't need response transformation
	return response, nil
}

func (p *AnthropicProvider) TransformStream(chunk []byte, state *StreamState) ([]byte, error) {
	// Anthropic format doesn't need transformation for streaming
	return chunk, nil
}
