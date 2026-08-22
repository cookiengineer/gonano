package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
)

func testServer(t *testing.T) *Server {
	t.Helper()
	cfg := model.Config{
		SequenceLen: 32, VocabSize: 265, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(0))
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return NewServer(m, tok, nil, "gonano")
}

func TestRenderMessagesSystemMerge(t *testing.T) {
	srv := testServer(t)
	ids := srv.renderMessages([]ChatMessage{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "hi"},
	}, nil)
	got := srv.Tokenizer.Decode(ids)
	want := "<|bos|><|user_start|>You are helpful\n\nhi<|user_end|><|assistant_start|>"
	if got != want {
		t.Fatalf("decoded = %q, want %q", got, want)
	}
}

func TestRenderMessagesToolRoundTrip(t *testing.T) {
	srv := testServer(t)
	ids := srv.renderMessages([]ChatMessage{
		{Role: "user", Content: "what is 2+2"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_0", Type: "function", Function: FunctionCall{Name: "calculator", Arguments: "2+2"}}}},
		{Role: "tool", Content: "4"},
	}, nil)
	got := srv.Tokenizer.Decode(ids)
	want := "<|bos|><|user_start|>what is 2+2<|user_end|>" +
		"<|assistant_start|><|tool_start|>2+2<|tool_end|><|assistant_end|>" +
		"<|tool_output_start|>4<|tool_output_end|><|assistant_start|>"
	if got != want {
		t.Fatalf("decoded = %q, want %q", got, want)
	}
}

func TestOptionsDefaults(t *testing.T) {
	opts := options(ChatCompletionRequest{}, 128)
	if opts.temperature != 0.7 {
		t.Fatalf("temperature = %v, want 0.7", opts.temperature)
	}
	if opts.n != 1 {
		t.Fatalf("n = %d, want 1", opts.n)
	}
	if opts.maxTokens != 128 {
		t.Fatalf("maxTokens = %d, want 128", opts.maxTokens)
	}
	if opts.seed != 42 {
		t.Fatalf("seed = %d, want 42", opts.seed)
	}
}

func TestOptionsExplicit(t *testing.T) {
	temp := float32(0.2)
	seed := uint64(7)
	opts := options(ChatCompletionRequest{Temperature: &temp, N: 2, MaxTokens: 10, TopK: 5, Seed: &seed}, 128)
	if opts.temperature != 0.2 || opts.n != 2 || opts.maxTokens != 10 || opts.topK != 5 || opts.seed != 7 {
		t.Fatalf("options = %+v", opts)
	}
}

func TestModelsEndpoint(t *testing.T) {
	srv := testServer(t)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var resp ModelsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Data) != 1 || resp.Data[0].ID != "gonano" {
		t.Fatalf("models = %+v", resp.Data)
	}
}

func TestChatCompletionsNonStreaming(t *testing.T) {
	srv := testServer(t)
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":8,"temperature":0.0}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Object != "chat.completion" {
		t.Fatalf("object = %q", resp.Object)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Fatalf("role = %q", resp.Choices[0].Message.Role)
	}
	if resp.Usage.PromptTokens <= 0 || resp.Usage.CompletionTokens <= 0 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestChatCompletionsStreaming(t *testing.T) {
	srv := testServer(t)
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":6,"temperature":0.0,"stream":true}`
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	ct := rec.Header().Get("Content-Type")
	if !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}
	out := rec.Body.String()
	if !strings.Contains(out, "data: ") {
		t.Fatalf("no SSE frames in body")
	}
	if !strings.Contains(out, "[DONE]") {
		t.Fatalf("no [DONE] sentinel")
	}
}

func TestChatCompletionsRejectsEmptyMessages(t *testing.T) {
	srv := testServer(t)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gonano","messages":[]}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}
