package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// Handler returns the HTTP handler for the OpenAI-compatible API.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/models/", s.handleModel)
	return mux
}

// handleModels lists the available model.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, ModelsResponse{
		Object: "list",
		Data: []ModelEntry{
			{ID: s.ModelName, Object: "model", Created: 0, OwnedBy: "gonano"},
		},
	})
}

// handleModel serves GET /v1/models/{name}.
func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, ModelEntry{ID: s.ModelName, Object: "model", OwnedBy: "gonano"})
}

// handleChatCompletions serves POST /v1/chat/completions, streaming or not.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if len(req.Messages) == 0 {
		http.Error(w, "messages must not be empty", http.StatusBadRequest)
		return
	}

	prompt := s.renderMessages(req.Messages, req.Tools)
	opts := options(req, s.Model.Config.SequenceLen)

	if req.Stream {
		s.streamCompletion(w, prompt, req, opts)
		return
	}
	s.complete(w, prompt, req, opts)
}

// complete writes a single non-streaming chat completion response.
func (s *Server) complete(w http.ResponseWriter, prompt []int, req ChatCompletionRequest, opts generationOptions) {
	rows := s.generate(prompt, opts.temperature, opts.topK, opts.maxTokens, opts.n, opts.seed)

	choices := make([]Choice, len(rows))
	var totalPrompt, totalCompletion int
	for i, r := range rows {
		choices[i] = Choice{
			Index:        i,
			Message:      ResponseMessage{Role: "assistant", Content: r.content, ToolCalls: r.toolCalls},
			FinishReason: r.finish,
		}
		totalPrompt += r.promptLen
		totalCompletion += r.completion
	}
	writeJSON(w, http.StatusOK, ChatCompletionResponse{
		ID:      chatID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.ModelName,
		Choices: choices,
		Usage:   Usage{PromptTokens: totalPrompt, CompletionTokens: totalCompletion, TotalTokens: totalPrompt + totalCompletion},
	})
}

// streamCompletion writes a server-sent-events stream of completion chunks.
func (s *Server) streamCompletion(w http.ResponseWriter, prompt []int, req ChatCompletionRequest, opts generationOptions) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	tok := s.Tokenizer
	toolStart := tok.EncodeSpecial("<|tool_start|>")
	toolEnd := tok.EncodeSpecial("<|tool_end|>")
	assistantEnd := tok.EncodeSpecial("<|assistant_end|>")
	bos := tok.BOSTokenID()

	id := chatID()
	created := time.Now().Unix()

	// Emit the initial role chunk.
	writeSSE(w, flusher, ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   s.ModelName,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{Role: "assistant"}}},
	})

	inCall := make([]bool, opts.n)
	finished := make([]bool, opts.n)
	completion := 0

	gen := s.Engine.Generate(prompt, opts.n, opts.maxTokens, opts.temperature, opts.topK, opts.seed)
	gen(func(column, mask []int) bool {
		for i := 0; i < opts.n; i++ {
			if finished[i] {
				continue
			}
			if mask[i] == 0 {
				continue // forced tool output; hidden from the stream
			}
			tk := column[i]
			switch tk {
			case toolStart:
				inCall[i] = true
			case toolEnd:
				inCall[i] = false
			case assistantEnd, bos:
				finished[i] = true
			default:
				if inCall[i] {
					continue // tool-call expression; hidden from the stream
				}
				text := tok.Decode([]int{tk})
				completion++
				writeSSE(w, flusher, ChatCompletionChunk{
					ID:      id,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   s.ModelName,
					Choices: []StreamChoice{{Index: i, Delta: StreamDelta{Content: text}}},
				})
			}
		}
		for _, f := range finished {
			if !f {
				return true
			}
		}
		return false
	})

	// Final chunk with the finish reason.
	finish := "stop"
	if completion >= opts.maxTokens {
		finish = "length"
	}
	writeSSE(w, flusher, ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   s.ModelName,
		Choices: []StreamChoice{{Index: 0, Delta: StreamDelta{}, FinishReason: &finish}},
	})
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeSSE(w http.ResponseWriter, f http.Flusher, chunk ChatCompletionChunk) {
	b, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", b)
	f.Flush()
}

// chatID returns a short unique completion id.
func chatID() string {
	return fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
}
