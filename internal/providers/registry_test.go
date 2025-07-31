package providers

import (
	"testing"

	"github.com/Davincible/claude-code-open/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegistry_RegisterAndGet(t *testing.T) {
	registry := NewRegistry()
	provider := NewOpenRouterProvider(&config.Provider{Name: "openrouter"})

	// Register provider
	registry.Register(provider)

	// Get provider
	retrievedProvider, exists := registry.Get("openrouter")
	assert.True(t, exists, "provider should exist after registration")
	assert.Equal(t, "openrouter", retrievedProvider.Name(), "provider name should match")
}

func TestRegistry_GetByDomain(t *testing.T) {
	registry := NewRegistry()
	registry.Initialize([]config.Provider{{Name: "openrouter"}, {Name: "openai"}, {Name: "anthropic"}, {Name: "nvidia"}, {Name: "gemini"}})

	testCases := []struct {
		domain   string
		expected string
	}{
		{"https://openrouter.ai/api/v1/chat/completions", "openrouter"},
		{"https://api.openrouter.ai/api/v1/chat/completions", "openrouter"},
		{"https://api.openai.com/v1/chat/completions", "openai"},
		{"https://api.anthropic.com/v1/messages", "anthropic"},
		{"https://integrate.api.nvidia.com/v1/chat/completions", "nvidia"},
		{"https://api.nvidia.com/v1/chat/completions", "nvidia"},
		{"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent", "gemini"},
		{"https://googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent", "gemini"},
	}

	for _, tc := range testCases {
		provider, err := registry.GetByDomain(tc.domain)
		require.NoError(t, err, "should get provider for domain %s", tc.domain)
		assert.Equal(t, tc.expected, provider.Name(), "provider name should match for domain %s", tc.domain)
	}
}

func TestRegistry_GetByDomain_InvalidURL(t *testing.T) {
	registry := NewRegistry()
	registry.Initialize([]config.Provider{{Name: "openrouter"}, {Name: "openai"}, {Name: "anthropic"}, {Name: "nvidia"}, {Name: "gemini"}})

	_, err := registry.GetByDomain("invalid-url")
	assert.Error(t, err, "should get error for invalid URL")
}

func TestRegistry_GetByDomain_UnknownDomain(t *testing.T) {
	registry := NewRegistry()
	registry.Initialize([]config.Provider{{Name: "openrouter"}, {Name: "openai"}, {Name: "anthropic"}, {Name: "nvidia"}, {Name: "gemini"}})

	_, err := registry.GetByDomain("https://unknown-provider.com/api")
	assert.Error(t, err, "should get error for unknown domain")
}

func TestRegistry_List(t *testing.T) {
	registry := NewRegistry()
	registry.Initialize([]config.Provider{{Name: "openrouter"}, {Name: "openai"}, {Name: "anthropic"}, {Name: "nvidia"}, {Name: "gemini"}})

	providers := registry.List()

	expectedProviders := []string{"openrouter", "openai", "anthropic", "nvidia", "gemini"}
	assert.Len(t, providers, len(expectedProviders), "should have expected number of providers")

	// Check that all expected providers are present
	for _, expected := range expectedProviders {
		assert.Contains(t, providers, expected, "should contain expected provider %s", expected)
	}
}

func TestRegistry_GetNonExistent(t *testing.T) {
	registry := NewRegistry()

	_, exists := registry.Get("nonexistent")
	assert.False(t, exists, "non-existent provider should not exist")
}
