package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/cookiengineer/gonano/inference"
)

// Handler returns the HTTP handler for the OpenAI-compatible API.
func (server *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", server.handleChatCompletions)
	mux.HandleFunc("/v1/models", server.handleModels)
	mux.HandleFunc("/v1/models/", server.handleModel)
	return mux
}

// handleModels lists the available model.
func (server *Server) handleModels(writer http.ResponseWriter, request *http.Request) {
	writeJSON(writer, http.StatusOK, ModelsResponse{
		Object: "list",
		Data: []ModelEntry{
			{ID: server.ModelName, Object: "model", Created: 0, OwnedBy: "gonano"},
		},
	})
}

// handleModel serves GET /v1/models/{name}.
func (server *Server) handleModel(writer http.ResponseWriter, request *http.Request) {
	writeJSON(writer, http.StatusOK, ModelEntry{ID: server.ModelName, Object: "model", OwnedBy: "gonano"})
}

// handleChatCompletions serves POST /v1/chat/completions, streaming or not.
func (server *Server) handleChatCompletions(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var chatRequest ChatCompletionRequest
	if err := json.NewDecoder(request.Body).Decode(&chatRequest); err != nil {
		http.Error(writer, "invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(chatRequest.Messages) == 0 {
		http.Error(writer, "messages must not be empty", http.StatusBadRequest)
		return
	}

	prompt := server.renderMessages(chatRequest.Messages, chatRequest.Tools, chatRequest.ReasoningEffort, chatRequest.Thinking)
	genOptions := resolveOptions(chatRequest, server.Model.Config.SequenceLen)

	if chatRequest.Stream {
		server.streamCompletion(writer, prompt, chatRequest, genOptions)
		return
	}
	server.complete(writer, prompt, chatRequest, genOptions)
}

// complete writes a single non-streaming chat completion response.
func (server *Server) complete(writer http.ResponseWriter, prompt []int, chatRequest ChatCompletionRequest, genOptions generationOptions) {
	thinking := resolveThinking(chatRequest)
	rows := server.generate(prompt, genOptions.temperature, genOptions.topK, genOptions.maxTokens, genOptions.numSamples, genOptions.seed, thinking)

	choices := make([]Choice, len(rows))
	var totalPrompt, totalCompletion int
	for index, row := range rows {
		choices[index] = Choice{
			Index:        index,
			Message:      ResponseMessage{Role: "assistant", Content: row.content, ReasoningContent: row.reasoning, ToolCalls: row.toolCalls},
			FinishReason: row.finish,
		}
		totalPrompt += row.promptLen
		totalCompletion += row.completion
	}
	writeJSON(writer, http.StatusOK, ChatCompletionResponse{
		ID:      chatID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   server.ModelName,
		Choices: choices,
		Usage:   Usage{PromptTokens: totalPrompt, CompletionTokens: totalCompletion, TotalTokens: totalPrompt + totalCompletion},
	})
}

// streamCompletion writes a server-sent-events stream of completion chunks.
func (server *Server) streamCompletion(writer http.ResponseWriter, prompt []int, chatRequest ChatCompletionRequest, genOptions generationOptions) {
	flusher, ok := writer.(http.Flusher)
	if !ok {
		http.Error(writer, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")
	writer.Header().Set("Connection", "keep-alive")

	tokenizerImpl := server.Tokenizer
	toolStart := tokenizerImpl.EncodeSpecial("<|tool_start|>")
	toolEnd := tokenizerImpl.EncodeSpecial("<|tool_end|>")
	assistantEnd := tokenizerImpl.EncodeSpecial("<|assistant_end|>")
	thinkStart := tokenizerImpl.EncodeSpecial("<|think_start|>")
	thinkEnd := tokenizerImpl.EncodeSpecial("<|think_end|>")
	bosToken := tokenizerImpl.BOSTokenID()

	id := chatID()
	created := time.Now().Unix()

	// Emit the initial role chunk.
	writeSSE(writer, flusher, ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   server.ModelName,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Role: "assistant"}}},
	})

	thinking := resolveThinking(chatRequest)
	inToolCall := make([]bool, genOptions.numSamples)
	inThinking := make([]bool, genOptions.numSamples)
	primedThinking := len(prompt) > 0 && prompt[len(prompt)-1] == thinkStart
	for index := range inThinking {
		inThinking[index] = primedThinking
	}
	finished := make([]bool, genOptions.numSamples)
	completion := 0

	generate := server.Engine.GenerateWith(prompt, genOptions.numSamples, genOptions.maxTokens, genOptions.temperature, genOptions.topK, genOptions.seed,
		inference.GenerateOptions{Thinking: thinking.enabled || thinking.budget > 0, ThinkingBudget: thinking.budget})
	generate(func(column, mask []int) bool {
		for index := 0; index < genOptions.numSamples; index++ {
			if finished[index] {
				continue
			}
			token := column[index]
			// Thinking delimiters are meaningful even when forced by the
			// budget, so handle them before the forced-token filter.
			switch token {
			case thinkStart:
				inThinking[index] = true
				continue
			case thinkEnd:
				inThinking[index] = false
				continue
			}
			if mask[index] == 0 {
				continue // forced tool output; hidden from the stream
			}
			switch token {
			case toolStart:
				inToolCall[index] = true
			case toolEnd:
				inToolCall[index] = false
			case assistantEnd, bosToken:
				finished[index] = true
			default:
				if inToolCall[index] {
					continue // tool-call expression; hidden from the stream
				}
				text := tokenizerImpl.Decode([]int{token})
				completion++
				delta := StreamDelta{Content: text}
				if inThinking[index] {
					delta = StreamDelta{ReasoningContent: text}
				}
				writeSSE(writer, flusher, ChatCompletionChunk{
					ID:      id,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   server.ModelName,
					Choices: []StreamChoice{{Index: index, Delta: delta}},
				})
			}
		}
		for _, done := range finished {
			if !done {
				return true
			}
		}
		return false
	})

	// Final chunk with the finish reason.
	finish := "stop"
	if completion >= genOptions.maxTokens {
		finish = "length"
	}
	writeSSE(writer, flusher, ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   server.ModelName,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{}, FinishReason: &finish}},
	})
	fmt.Fprintf(writer, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeJSON(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}

func writeSSE(writer http.ResponseWriter, flusher http.Flusher, chunk ChatCompletionChunk) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(writer, "data: %s\n\n", data)
	flusher.Flush()
}

// chatID returns a short unique completion id.
func chatID() string {
	return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}
