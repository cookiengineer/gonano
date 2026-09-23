package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func testServer(test *testing.T) *Server {
	test.Helper()
	config := model.Config{
		SequenceLen: 32, VocabSize: 265, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(0))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return NewServer(transformer, tokenizerImpl, nil, "gonano")
}

func TestRenderMessagesSystemMerge(test *testing.T) {
	srv := testServer(test)
	ids := srv.renderMessages([]ChatMessage{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "hi"},
	}, nil, nil)
	got := srv.Tokenizer.Decode(ids)
	want := "<|bos|><|user_start|>You are helpful\n\nhi<|user_end|><|assistant_start|>"
	if got != want {
		test.Fatalf("decoded = %q, want %q", got, want)
	}
}

// TestRenderMessagesReasoningEffort checks the optional reasoning_effort field
// is prepended to the system prompt.
func TestRenderMessagesReasoningEffort(test *testing.T) {
	srv := testServer(test)
	effort := 75
	ids := srv.renderMessages([]ChatMessage{{Role: "user", Content: "hi"}}, nil, &effort)
	got := srv.Tokenizer.Decode(ids)
	want := "<|bos|><|user_start|>Reasoning Effort: 75 (range 1--100; higher values request more thorough reasoning)\n\nhi<|user_end|><|assistant_start|>"
	if got != want {
		test.Fatalf("decoded = %q, want %q", got, want)
	}
}

func TestRenderMessagesToolRoundTrip(test *testing.T) {
	srv := testServer(test)
	ids := srv.renderMessages([]ChatMessage{
		{Role: "user", Content: "what is 2+2"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_0", Type: "function", Function: FunctionCall{Name: "calculator", Arguments: "2+2"}}}},
		{Role: "tool", Content: "4"},
	}, nil, nil)
	got := srv.Tokenizer.Decode(ids)
	want := "<|bos|><|user_start|>what is 2+2<|user_end|>" +
		"<|assistant_start|><|tool_start|>2+2<|tool_end|><|assistant_end|>" +
		"<|tool_output_start|>4<|tool_output_end|><|assistant_start|>"
	if got != want {
		test.Fatalf("decoded = %q, want %q", got, want)
	}
}

func TestOptionsDefaults(test *testing.T) {
	genOptions := resolveOptions(ChatCompletionRequest{}, 128)
	if genOptions.temperature != 0.7 {
		test.Fatalf("temperature = %v, want 0.7", genOptions.temperature)
	}
	if genOptions.numSamples != 1 {
		test.Fatalf("numSamples = %d, want 1", genOptions.numSamples)
	}
	if genOptions.maxTokens != 128 {
		test.Fatalf("maxTokens = %d, want 128", genOptions.maxTokens)
	}
	if genOptions.seed != 42 {
		test.Fatalf("seed = %d, want 42", genOptions.seed)
	}
}

func TestOptionsExplicit(test *testing.T) {
	temperature := float32(0.2)
	seed := uint64(7)
	genOptions := resolveOptions(ChatCompletionRequest{Temperature: &temperature, N: 2, MaxTokens: 10, TopK: 5, Seed: &seed}, 128)
	if genOptions.temperature != 0.2 || genOptions.numSamples != 2 || genOptions.maxTokens != 10 || genOptions.topK != 5 || genOptions.seed != 7 {
		test.Fatalf("options = %+v", genOptions)
	}
}

func TestModelsEndpoint(test *testing.T) {
	srv := testServer(test)
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if recorder.Code != http.StatusOK {
		test.Fatalf("status = %d", recorder.Code)
	}
	var resp ModelsResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		test.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 1 || resp.Data[0].ID != "gonano" {
		test.Fatalf("models = %+v", resp.Data)
	}
}

func TestChatCompletionsNonStreaming(test *testing.T) {
	srv := testServer(test)
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"temperature":0.0}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		test.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var resp ChatCompletionResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		test.Fatalf("unmarshal: %v", err)
	}
	if resp.Object != "chat.completion" {
		test.Fatalf("object = %q", resp.Object)
	}
	if len(resp.Choices) != 1 {
		test.Fatalf("choices = %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message.Role != "assistant" {
		test.Fatalf("role = %q", resp.Choices[0].Message.Role)
	}
	if resp.Usage.PromptTokens <= 0 || resp.Usage.CompletionTokens <= 0 {
		test.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestChatCompletionsStreaming(test *testing.T) {
	srv := testServer(test)
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":6,"temperature":0.0,"stream":true}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		test.Fatalf("status = %d", recorder.Code)
	}
	contentType := recorder.Header().Get("Content-Type")
	if !strings.Contains(contentType, "text/event-stream") {
		test.Fatalf("content-type = %q", contentType)
	}
	output := recorder.Body.String()
	if !strings.Contains(output, "data: ") {
		test.Fatalf("no SSE frames in body")
	}
	if !strings.Contains(output, "[DONE]") {
		test.Fatalf("no [DONE] sentinel")
	}
}

func TestChatCompletionsRejectsEmptyMessages(test *testing.T) {
	srv := testServer(test)
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gonano","messages":[]}`)))
	if recorder.Code != http.StatusBadRequest {
		test.Fatalf("status = %d, want 400", recorder.Code)
	}
}
