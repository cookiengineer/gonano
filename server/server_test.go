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
		SequenceLen: 32, VocabSize: 256 + len(tokenizer.SpecialTokens), NumLayer: 1, NumHead: 2, NumKVHead: 2,
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
	}, nil, nil, nil)
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
	ids := srv.renderMessages([]ChatMessage{{Role: "user", Content: "hi"}}, nil, &effort, nil)
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
	}, nil, nil, nil)
	got := srv.Tokenizer.Decode(ids)
	want := "<|bos|><|user_start|>what is 2+2<|user_end|>" +
		"<|assistant_start|><|tool_start|>2+2<|tool_end|><|assistant_end|>" +
		"<|tool_output_start|>4<|tool_output_end|><|assistant_start|>"
	if got != want {
		test.Fatalf("decoded = %q, want %q", got, want)
	}
}

// testThinkingServer builds a zero-weight model whose argmax decoding always
// emits token 0, so reasoning/content splitting is deterministic.
func testThinkingServer(test *testing.T) *Server {
	test.Helper()
	config := model.Config{
		SequenceLen: 256, VocabSize: 256 + len(tokenizer.SpecialTokens), NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return NewServer(transformer, tokenizerImpl, nil, "gonano")
}

func TestRenderMessagesThinking(test *testing.T) {
	srv := testThinkingServer(test)
	enabled := true
	got := srv.Tokenizer.Decode(srv.renderMessages([]ChatMessage{{Role: "user", Content: "hi"}}, nil, nil, &enabled))
	if !strings.HasSuffix(got, "<|assistant_start|><|think_start|>") {
		test.Fatalf("enabled decoded = %q, want think_start suffix", got)
	}
	if !strings.Contains(got, "Thinking mode: enabled") {
		test.Fatalf("missing enabled instruction: %q", got)
	}
	disabled := false
	gotDisabled := srv.Tokenizer.Decode(srv.renderMessages([]ChatMessage{{Role: "user", Content: "hi"}}, nil, nil, &disabled))
	if strings.Contains(gotDisabled, "<|think_start|>") {
		test.Fatalf("disabled render primed think_start: %q", gotDisabled)
	}
	if !strings.Contains(gotDisabled, "Thinking mode: disabled") {
		test.Fatalf("missing disabled instruction: %q", gotDisabled)
	}
}

func TestRenderMessagesReasoningContentReplay(test *testing.T) {
	srv := testThinkingServer(test)
	got := srv.Tokenizer.Decode(srv.renderMessages([]ChatMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", ReasoningContent: "thinking", Content: "answer"},
	}, nil, nil, nil))
	want := "<|assistant_start|><|think_start|>thinking<|think_end|>answer<|assistant_end|>"
	if !strings.Contains(got, want) {
		test.Fatalf("decoded = %q, want it to contain %q", got, want)
	}
}

func TestGenerateSplitsReasoningAndContent(test *testing.T) {
	srv := testThinkingServer(test)
	prompt := []int{
		srv.Tokenizer.BOSTokenID(),
		srv.Tokenizer.EncodeSpecial("<|assistant_start|>"),
		srv.Tokenizer.EncodeSpecial("<|think_start|>"),
	}
	rows := srv.generate(srv.Engine, prompt, 0, 0, 6, 1, 1, thinkingOptions{enabled: true, budget: 2})
	if len(rows) != 1 {
		test.Fatalf("rows = %d, want 1", len(rows))
	}
	if len(rows[0].reasoning) != 2 {
		test.Fatalf("reasoning length = %d, want 2", len(rows[0].reasoning))
	}
	if len(rows[0].content) != 3 {
		test.Fatalf("content length = %d, want 3", len(rows[0].content))
	}
	if rows[0].completion != len(rows[0].reasoning)+len(rows[0].content) {
		test.Fatalf("completion = %d, want %d", rows[0].completion, len(rows[0].reasoning)+len(rows[0].content))
	}
}

func TestChatCompletionsReasoningContent(test *testing.T) {
	srv := testThinkingServer(test)
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":6,"temperature":0.0,"thinking":true,"thinking_budget":2}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		test.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var resp ChatCompletionResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		test.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Choices) != 1 || len(resp.Choices[0].Message.ReasoningContent) != 2 {
		test.Fatalf("reasoning = %q", resp.Choices[0].Message.ReasoningContent)
	}
}

func TestChatCompletionsStreamingReasoningContent(test *testing.T) {
	srv := testThinkingServer(test)
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":6,"temperature":0.0,"stream":true,"thinking":true,"thinking_budget":2}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		test.Fatalf("status = %d", recorder.Code)
	}
	if !strings.Contains(recorder.Body.String(), `"reasoning_content"`) {
		test.Fatalf("no reasoning_content in stream: %s", recorder.Body.String())
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
