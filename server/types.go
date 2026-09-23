// Package server implements an OpenAI-compatible HTTP API for the gonano model,
// including chat completions (streaming and non-streaming) and tool-call
// support. Tool calls are executed server-side through an inference.Registry of Go
// tools, so a single request can carry a full tool-using turn end to end.
package server

// ChatMessage is one message in a chat completion request.
type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall is a structured tool invocation.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

// FunctionCall names a function and its (JSON) arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// FunctionDef declares a function available to the model.
type FunctionDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ToolDef declares a tool available to the model.
type ToolDef struct {
	Type     string      `json:"type"` // "function"
	Function FunctionDef `json:"function"`
}

// ChatCompletionRequest is the body of POST /v1/chat/completions.
type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature *float32      `json:"temperature,omitempty"`
	TopP        *float32      `json:"top_p,omitempty"`
	TopK        int           `json:"top_k,omitempty"`
	N           int           `json:"n,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Stop        []string      `json:"stop,omitempty"`
	Tools       []ToolDef     `json:"tools,omitempty"`
	Seed        *uint64       `json:"seed,omitempty"`
	// ReasoningEffort optionally selects a reasoning-effort level b in [1,100]
	// by prepending the effort instruction to the system prompt
	// (DeepSeek-V4.1 §5.1.4).
	ReasoningEffort *int `json:"reasoning_effort,omitempty"`
}

// Usage reports token counts.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ResponseMessage is the assistant message in a chat completion response.
type ResponseMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Choice is one completion choice.
type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// ChatCompletionResponse is the body of a chat completion response.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// StreamDelta is one delta in a streamed chunk.
type StreamDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// StreamChoice is one choice in a streamed chunk.
type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

// ChatCompletionChunk is one SSE data frame of a streamed response.
type ChatCompletionChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// ModelEntry describes an available model.
type ModelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse lists available models.
type ModelsResponse struct {
	Object string       `json:"object"`
	Data   []ModelEntry `json:"data"`
}
