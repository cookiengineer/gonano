package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/cookiengineer/gonano/bank"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/router"
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
	}, nil, nil, nil, "")
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
	ids := srv.renderMessages([]ChatMessage{{Role: "user", Content: "hi"}}, nil, &effort, nil, "")
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
	}, nil, nil, nil, "")
	got := srv.Tokenizer.Decode(ids)
	want := "<|bos|><|user_start|>what is 2+2<|user_end|>" +
		"<|assistant_start|><|tool_start|>{\"arguments\":\"2+2\",\"name\":\"calculator\"}<|tool_end|><|assistant_end|>" +
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
	got := srv.Tokenizer.Decode(srv.renderMessages([]ChatMessage{{Role: "user", Content: "hi"}}, nil, nil, &enabled, ""))
	if !strings.HasSuffix(got, "<|assistant_start|><|think_start|>") {
		test.Fatalf("enabled decoded = %q, want think_start suffix", got)
	}
	if !strings.Contains(got, "Thinking mode: enabled") {
		test.Fatalf("missing enabled instruction: %q", got)
	}
	disabled := false
	gotDisabled := srv.Tokenizer.Decode(srv.renderMessages([]ChatMessage{{Role: "user", Content: "hi"}}, nil, nil, &disabled, ""))
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
	}, nil, nil, nil, ""))
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
	rows := srv.generate(srv.Engine, prompt, generationOptions{temperature: 0, topK: 0, maxTokens: 6, numSamples: 1, seed: 1}, thinkingOptions{enabled: true, budget: 2})
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

func TestOptionsJSONResponseFormat(t *testing.T) {
	genOptions := resolveOptions(ChatCompletionRequest{ResponseFormat: &ResponseFormat{Type: "json_object"}}, 128)
	if genOptions.constraint == nil {
		t.Fatal("json_object response format should install a constraint")
	}
}

type scriptedGenerator struct {
	tokens []int
}

func (generator scriptedGenerator) GenerateWith(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64, options inference.GenerateOptions) func(yield func([]int, []int) bool) {
	return func(yield func([]int, []int) bool) {
		for _, token := range generator.tokens {
			if !yield([]int{token}, []int{1}) {
				return
			}
		}
	}
}

func TestStreamingStopWithholdsSuffix(t *testing.T) {
	srv := testServer(t)
	tokens := srv.Tokenizer.Encode("helloSTOPworld")
	recorder := httptest.NewRecorder()
	srv.streamCompletion(context.Background(), recorder, scriptedGenerator{tokens: tokens}, []int{srv.Tokenizer.BOSTokenID()}, ChatCompletionRequest{Stop: []string{"STOP"}}, generationOptions{
		temperature: 0, maxTokens: len(tokens), numSamples: 1, seed: 1, stop: []string{"STOP"},
	})
	body := recorder.Body.String()
	if strings.Contains(body, "STOP") || strings.Contains(body, "world") {
		t.Fatalf("stream leaked stop suffix: %s", body)
	}
	if got := streamedContent(t, body); got != "hello" {
		t.Fatalf("streamed content = %q, want hello", got)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("stream missing stop finish reason: %s", body)
	}
}

func streamedContent(t *testing.T, body string) string {
	t.Helper()
	var content strings.Builder
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatalf("decode stream chunk: %v", err)
		}
		for _, choice := range chunk.Choices {
			content.WriteString(choice.Delta.Content)
		}
	}
	return content.String()
}

func TestGenerateStopTrimsContent(t *testing.T) {
	srv := testServer(t)
	tokens := srv.Tokenizer.Encode("helloSTOPworld")
	rows := srv.generate(scriptedGenerator{tokens: tokens}, []int{srv.Tokenizer.BOSTokenID()}, generationOptions{
		temperature: 0, maxTokens: len(tokens), numSamples: 1, seed: 1, stop: []string{"STOP"},
	}, thinkingOptions{})
	if len(rows) != 1 || rows[0].content != "hello" {
		t.Fatalf("content = %q, want hello", rows[0].content)
	}
}

func TestGenerateLogprobs(t *testing.T) {
	srv := testThinkingServer(t)
	prompt := []int{srv.Tokenizer.BOSTokenID()}
	rows := srv.generate(srv.Engine, prompt, generationOptions{
		temperature: 1, maxTokens: 4, numSamples: 1, seed: 1, logprobs: true, topLogprobs: 2,
	}, thinkingOptions{})
	if len(rows) != 1 {
		t.Fatalf("rows = %d", len(rows))
	}
	if len(rows[0].logprobs) != rows[0].completion {
		t.Fatalf("logprobs = %d, completion = %d", len(rows[0].logprobs), rows[0].completion)
	}
	for _, entry := range rows[0].logprobs {
		if len(entry.TopLogprobs) > 2 {
			t.Fatalf("top logprobs = %d, want <= 2", len(entry.TopLogprobs))
		}
	}
}

func TestChatCompletionsLogprobs(t *testing.T) {
	srv := testThinkingServer(t)
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":4,"temperature":1.0,"logprobs":true,"top_logprobs":2}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var resp ChatCompletionResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Logprobs == nil || len(resp.Choices[0].Logprobs.Content) == 0 {
		t.Fatalf("missing logprobs: %+v", resp.Choices)
	}
}

func TestSchedulerGeneratorMatchesEngine(t *testing.T) {
	srv := testServer(t)
	srv.StartScheduler(2, 0)
	defer srv.Close()
	prompt := srv.renderMessages([]ChatMessage{{Role: "user", Content: "hello"}}, nil, nil, nil, "")
	options := generationOptions{temperature: 0.7, topK: 5, maxTokens: 4, numSamples: 1, seed: 3}
	scheduled := srv.generate(srv.schedulerGenerator, prompt, options, thinkingOptions{})
	reference := srv.generate(srv.Engine, prompt, options, thinkingOptions{})
	if len(scheduled) != 1 || len(reference) != 1 {
		t.Fatalf("rows = %d/%d", len(scheduled), len(reference))
	}
	if scheduled[0].content != reference[0].content {
		t.Fatalf("scheduler content %q != engine content %q", scheduled[0].content, reference[0].content)
	}
	if scheduled[0].finish != reference[0].finish {
		t.Fatalf("scheduler finish %q != engine finish %q", scheduled[0].finish, reference[0].finish)
	}
}

func TestChatCompletionsScheduler(t *testing.T) {
	srv := testServer(t)
	srv.StartScheduler(2, 0)
	defer srv.Close()
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":4,"temperature":0.5,"seed":7}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	var response ChatCompletionResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(response.Choices) != 1 {
		t.Fatalf("choices = %d", len(response.Choices))
	}
	if response.Choices[0].Message.Role != "assistant" {
		t.Fatalf("role = %q", response.Choices[0].Message.Role)
	}
}

func TestChatCompletionsSchedulerStreaming(t *testing.T) {
	srv := testServer(t)
	srv.StartScheduler(2, 0)
	defer srv.Close()
	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":4,"temperature":0.5,"seed":7,"stream":true}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "[DONE]") {
		t.Fatalf("stream missing [DONE]: %s", recorder.Body.String())
	}
}

func TestChatCompletionsSchedulerConcurrent(t *testing.T) {
	srv := testServer(t)
	srv.StartScheduler(2, 0)
	defer srv.Close()

	var wait sync.WaitGroup
	errors := make(chan error, 2)
	for index := 0; index < 2; index++ {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			body := fmt.Sprintf(`{"model":"gonano","messages":[{"role":"user","content":"hello %d"}],"max_tokens":4,"temperature":0.5,"seed":%d}`, index, index+1)
			recorder := httptest.NewRecorder()
			srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
			if recorder.Code != http.StatusOK {
				errors <- fmt.Errorf("request %d status = %d, body = %s", index, recorder.Code, recorder.Body.String())
			}
		}(index)
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

func TestDomainSchedulerSelectsRoutedDomain(t *testing.T) {
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
	path := t.TempDir() + "/math.gn"
	if err := checkpoint.Save(path, checkpoint.Meta{ModelConfig: config}, transformer.NamedParameters()); err != nil {
		t.Fatalf("save checkpoint: %v", err)
	}
	manifest := &bank.Manifest{Domains: []bank.Domain{
		{ID: "math", Checkpoint: path},
		{ID: "physics", Checkpoint: path},
	}}
	domainBank := bank.NewBank(manifest, tokenizerImpl, bank.BankOptions{})
	if _, err := domainBank.Acquire("math"); err != nil {
		t.Fatalf("acquire domain: %v", err)
	}

	srv := NewServer(transformer, tokenizerImpl, nil, "gonano")
	srv.Bank = domainBank
	srv.Router = &router.Router{Domains: []string{"math", "physics"}}
	srv.EnableDomainSchedulers(2, 0, 4)
	defer srv.Close()

	prompt := srv.renderMessages([]ChatMessage{{Role: "user", Content: "what is 2+2"}}, nil, nil, nil, "")
	generator, domains := srv.selectGenerator(ChatCompletionRequest{Model: "math"}, prompt)
	if len(domains) != 1 || domains[0] != "math" {
		t.Fatalf("domains = %v, want [math]", domains)
	}
	if _, ok := generator.(*schedulerGenerator); !ok {
		t.Fatalf("generator type = %T, want *schedulerGenerator", generator)
	}
	options := generationOptions{temperature: 0, maxTokens: 3, numSamples: 1, seed: 1}
	scheduled := srv.generate(generator, prompt, options, thinkingOptions{})
	reference := srv.generate(srv.Engine, prompt, options, thinkingOptions{})
	if len(scheduled) != 1 || len(reference) != 1 {
		t.Fatalf("rows = %d/%d", len(scheduled), len(reference))
	}
	if scheduled[0].content != reference[0].content {
		t.Fatalf("domain scheduler content %q != engine %q", scheduled[0].content, reference[0].content)
	}
	if again := srv.schedulerForDomain("math"); again != generator {
		t.Fatal("domain scheduler was not cached")
	}
}

func TestMultiDomainEnsembleSchedulerBatches(t *testing.T) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 256 + len(tokenizer.SpecialTokens), NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	dir := t.TempDir()
	save := func(name string, seed uint64) string {
		transformer := model.NewTransformer(config)
		transformer.InitWeights(tensors.NewRNG(seed))
		path := dir + "/" + name + ".gn"
		if err := checkpoint.Save(path, checkpoint.Meta{ModelConfig: config}, transformer.NamedParameters()); err != nil {
			t.Fatalf("save %s: %v", name, err)
		}
		return path
	}
	mathPath := save("math", 1)
	physicsPath := save("physics", 2)

	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerImpl := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	manifest := &bank.Manifest{Domains: []bank.Domain{
		{ID: "math", Checkpoint: mathPath},
		{ID: "physics", Checkpoint: physicsPath},
	}}
	domainBank := bank.NewBank(manifest, tokenizerImpl, bank.BankOptions{})
	classifier := router.NewNgramClassifier(2, router.NgramConfig{Buckets: 16})

	fallbackModel := model.NewTransformer(config)
	fallbackModel.InitWeights(tensors.NewRNG(0))
	srv := NewServer(fallbackModel, tokenizerImpl, nil, "gonano")
	srv.Bank = domainBank
	srv.Router = &router.Router{Domains: []string{"math", "physics"}, Ngram: classifier}
	srv.MaxDomains = 2
	srv.MinDomainScore = 0
	srv.EnableDomainSchedulers(2, 0, 0)
	defer srv.Close()

	request := ChatCompletionRequest{Messages: []ChatMessage{{Role: "user", Content: "hello"}}}
	promptA := srv.renderMessages(request.Messages, nil, nil, nil, "")
	generatorA, domains := srv.selectGenerator(request, promptA)
	if len(domains) != 2 {
		t.Fatalf("domains = %v, want 2", domains)
	}
	first, ok := generatorA.(*ensembleSchedulerGenerator)
	if !ok {
		t.Fatalf("generator type = %T, want *ensembleSchedulerGenerator", generatorA)
	}
	requestB := ChatCompletionRequest{Messages: []ChatMessage{{Role: "user", Content: "world"}}}
	promptB := srv.renderMessages(requestB.Messages, nil, nil, nil, "")
	generatorB, _ := srv.selectGenerator(requestB, promptB)
	second, ok := generatorB.(*ensembleSchedulerGenerator)
	if !ok || second.runner != first.runner {
		t.Fatal("multi-domain requests did not share an ensemble scheduler")
	}

	options := generationOptions{temperature: 0, maxTokens: 4, numSamples: 1, seed: 1}
	var scheduled [2][]rowResult
	var wait sync.WaitGroup
	wait.Add(2)
	go func() {
		defer wait.Done()
		scheduled[0] = srv.generate(generatorA, promptA, options, thinkingOptions{})
	}()
	go func() {
		defer wait.Done()
		scheduled[1] = srv.generate(generatorB, promptB, options, thinkingOptions{})
	}()
	wait.Wait()

	referenceA := srv.generate(first.fallback, promptA, options, thinkingOptions{})
	referenceB := srv.generate(first.fallback, promptB, options, thinkingOptions{})
	if len(scheduled[0]) != 1 || len(scheduled[1]) != 1 {
		t.Fatalf("scheduled rows = %d/%d", len(scheduled[0]), len(scheduled[1]))
	}
	if scheduled[0][0].content != referenceA[0].content || scheduled[1][0].content != referenceB[0].content {
		t.Fatalf("ensemble scheduler content mismatch: %q/%q vs %q/%q",
			scheduled[0][0].content, scheduled[1][0].content, referenceA[0].content, referenceB[0].content)
	}

	body := `{"model":"gonano","messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0,"seed":1}`
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if recorder.Code != http.StatusOK {
		t.Fatalf("HTTP status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
}

func TestResolveOptionsStructuredFormats(t *testing.T) {
	jsonSchema := resolveOptions(ChatCompletionRequest{ResponseFormat: &ResponseFormat{
		Type: "json_schema",
		JSONSchema: map[string]any{
			"name": "answer",
			"schema": map[string]any{
				"type":                 "object",
				"properties":           map[string]any{"a": map[string]any{"type": "string", "enum": []any{"a"}}},
				"required":             []any{"a"},
				"additionalProperties": false,
			},
		},
	}}, 32)
	if jsonSchema.constraint == nil {
		t.Fatal("json_schema did not install a constraint")
	}
	jsonProbe := jsonSchema.constraint.Clone()
	if !jsonProbe.Allowed([]byte("{\"a\":\"a\"}")) {
		t.Fatal("json_schema constraint rejected a valid value")
	}
	jsonProbe.Advance([]byte("{\"a\":\"a\"}"))
	if !jsonProbe.Finished() {
		t.Fatal("json_schema constraint did not finish")
	}

	regex := resolveOptions(ChatCompletionRequest{ResponseFormat: &ResponseFormat{
		Type: "regex", JSONSchema: map[string]any{"pattern": "a(b|c)*d"},
	}}, 32)
	if regex.constraint == nil {
		t.Fatal("regex constraint not installed")
	}
	regexProbe := regex.constraint.Clone()
	if !regexProbe.Allowed([]byte("abd")) {
		t.Fatal("regex constraint rejected a valid value")
	}
	regexProbe.Advance([]byte("abd"))
	if !regexProbe.Finished() {
		t.Fatal("regex constraint did not finish")
	}

	ebnf := resolveOptions(ChatCompletionRequest{ResponseFormat: &ResponseFormat{
		Type: "ebnf", JSONSchema: map[string]any{"grammar": `start = "a" { "b" } "c" ;`},
	}}, 32)
	if ebnf.constraint == nil {
		t.Fatal("ebnf constraint not installed")
	}
	ebnfProbe := ebnf.constraint.Clone()
	if !ebnfProbe.Allowed([]byte("abbc")) {
		t.Fatal("ebnf constraint rejected a valid value")
	}
	ebnfProbe.Advance([]byte("abbc"))
	if !ebnfProbe.Finished() {
		t.Fatal("ebnf constraint did not finish")
	}
}

func TestRenderMessagesToolCallJSON(t *testing.T) {
	srv := testServer(t)
	ids := srv.renderMessages([]ChatMessage{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_0", Type: "function", Function: FunctionCall{Name: "weather", Arguments: `{"city":"Paris"}`}}}},
		{Role: "tool", ToolCallID: "call_0", Content: `{"temp":20}`},
	}, []ToolDef{{Type: "function", Function: FunctionDef{Name: "weather", Parameters: map[string]any{"type": "object"}}}}, nil, nil, "")
	got := srv.Tokenizer.Decode(ids)
	if !strings.Contains(got, `<|tool_start|>{"arguments":{"city":"Paris"},"name":"weather"}<|tool_end|>`) {
		t.Fatalf("tool call JSON missing: %q", got)
	}
	if !strings.Contains(got, `Arguments must be a JSON object matching the tool schema`) {
		t.Fatalf("tool schema instruction missing: %q", got)
	}
}

func TestGenerateParsesJSONToolCall(t *testing.T) {
	srv := testServer(t)
	call := `{"name":"weather","arguments":{"city":"Paris"}}`
	tokens := []int{srv.Tokenizer.EncodeSpecial("<|tool_start|>")}
	tokens = append(tokens, srv.Tokenizer.Encode(call)...)
	tokens = append(tokens, srv.Tokenizer.EncodeSpecial("<|tool_end|>"))
	prompt := []int{srv.Tokenizer.BOSTokenID()}
	rows := srv.generate(scriptedGenerator{tokens: tokens}, prompt, generationOptions{
		temperature: 0, maxTokens: len(tokens) + 4, numSamples: 1, seed: 1,
	}, thinkingOptions{})
	if len(rows) != 1 || len(rows[0].toolCalls) != 1 {
		t.Fatalf("tool calls = %+v", rows)
	}
	callResult := rows[0].toolCalls[0]
	if callResult.Function.Name != "weather" || callResult.Function.Arguments != `{"city":"Paris"}` {
		t.Fatalf("tool call = %+v", callResult)
	}
	if rows[0].finish != "tool_calls" {
		t.Fatalf("finish = %q, want tool_calls", rows[0].finish)
	}
	if rows[0].content != "" {
		t.Fatalf("tool expression leaked into content: %q", rows[0].content)
	}
}

func TestStreamingJSONToolCall(t *testing.T) {
	srv := testServer(t)
	call := `{"name":"weather","arguments":{"city":"Paris"}}`
	tokens := []int{srv.Tokenizer.EncodeSpecial("<|tool_start|>")}
	tokens = append(tokens, srv.Tokenizer.Encode(call)...)
	tokens = append(tokens, srv.Tokenizer.EncodeSpecial("<|tool_end|>"))
	recorder := httptest.NewRecorder()
	srv.streamCompletion(context.Background(), recorder, scriptedGenerator{tokens: tokens},
		[]int{srv.Tokenizer.BOSTokenID()}, ChatCompletionRequest{},
		generationOptions{temperature: 0, maxTokens: len(tokens) + 4, numSamples: 1, seed: 1})
	body := recorder.Body.String()
	if !strings.Contains(body, `"name":"weather"`) || !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Fatalf("tool call stream missing: %s", body)
	}
	if strings.Contains(body, `"content":"{`) {
		t.Fatalf("tool expression leaked into stream content: %s", body)
	}
}
