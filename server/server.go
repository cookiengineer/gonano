package server

import (
	"fmt"

	"github.com/cookiengineer/gonano/infer"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tokenizer"
)

// Server serves an OpenAI-compatible chat completions API backed by a gonano
// model. Tool calls are executed server-side through the Tools registry, so a
// single request can run a complete tool-using turn.
type Server struct {
	Model     *model.Transformer
	Tokenizer *tokenizer.Tokenizer
	Engine    *infer.Engine
	Tools     *infer.Registry
	ModelName string
}

// NewServer builds a Server. tools may be nil, in which case the engine's
// built-in calculator registry is used.
func NewServer(m *model.Transformer, tok *tokenizer.Tokenizer, tools *infer.Registry, modelName string) *Server {
	engine := infer.NewEngine(m, tok)
	if tools != nil {
		engine.Tools = tools
	}
	return &Server{
		Model:     m,
		Tokenizer: tok,
		Engine:    engine,
		Tools:     engine.Tools,
		ModelName: modelName,
	}
}

// rowResult is the parsed output of one generated sample.
type rowResult struct {
	content    string
	toolCalls  []ToolCall
	finish     string
	promptLen  int
	completion int
}

// generate runs n samples of autoregressive generation and parses each row
// into its natural-language content, tool calls, and finish reason.
func (s *Server) generate(prompt []int, temperature float32, topK, maxTokens, n int, seed uint64) []rowResult {
	tok := s.Tokenizer
	results := make([]rowResult, n)

	toolStart := tok.EncodeSpecial("<|tool_start|>")
	toolEnd := tok.EncodeSpecial("<|tool_end|>")
	assistantEnd := tok.EncodeSpecial("<|assistant_end|>")
	bos := tok.BOSTokenID()

	var sampled [][]int
	var callArgs [][]int
	inCall := make([]bool, n)
	finished := make([]bool, n)
	sampled = make([][]int, n)
	callArgs = make([][]int, n)
	callCounter := 0

	gen := s.Engine.Generate(prompt, n, maxTokens, temperature, topK, seed)
	gen(func(column, mask []int) bool {
		for i := 0; i < n; i++ {
			if finished[i] {
				continue
			}
			if mask[i] == 0 {
				// Forced tokens are tool outputs; the model already saw them.
				continue
			}
			tk := column[i]
			switch tk {
			case toolStart:
				inCall[i] = true
				callArgs[i] = nil
			case toolEnd:
				if inCall[i] {
					args := tok.Decode(callArgs[i])
					name, _, ok := s.Tools.ExecuteTool(args)
					if !ok {
						name = "unknown"
					}
					results[i].toolCalls = append(results[i].toolCalls, ToolCall{
						ID:       fmt.Sprintf("call_%d", callCounter),
						Type:     "function",
						Function: FunctionCall{Name: name, Arguments: args},
					})
					callCounter++
					inCall[i] = false
					callArgs[i] = nil
				}
			case assistantEnd, bos:
				finished[i] = true
			default:
				if inCall[i] {
					callArgs[i] = append(callArgs[i], tk)
				} else {
					sampled[i] = append(sampled[i], tk)
				}
			}
		}
		for _, f := range finished {
			if !f {
				return true
			}
		}
		return false
	})

	for i := 0; i < n; i++ {
		results[i].content = tok.Decode(sampled[i])
		results[i].finish = "stop"
		if !finished[i] {
			results[i].finish = "length"
		}
		results[i].promptLen = len(prompt)
		results[i].completion = len(sampled[i])
		for _, c := range results[i].toolCalls {
			results[i].completion += len(tok.Encode(c.Function.Arguments))
		}
	}
	return results
}

// renderMessages converts OpenAI messages into a prompt token sequence primed
// for the assistant to complete.
func (s *Server) renderMessages(messages []ChatMessage, tools []ToolDef) []int {
	tok := s.Tokenizer
	ids := []int{tok.BOSTokenID()}

	messages = mergeSystemMessage(messages)
	if len(tools) > 0 {
		// Advise the model about the tools it may invoke.
		names := make([]string, 0, len(tools))
		for _, t := range tools {
			names = append(names, t.Function.Name)
		}
		messages = append([]ChatMessage{{Role: "system", Content: toolsSystemPrompt(names)}}, messages...)
		messages = mergeSystemMessage(messages)
	}

	userStart := tok.EncodeSpecial("<|user_start|>")
	userEnd := tok.EncodeSpecial("<|user_end|>")
	assistantStart := tok.EncodeSpecial("<|assistant_start|>")
	assistantEnd := tok.EncodeSpecial("<|assistant_end|>")
	toolStart := tok.EncodeSpecial("<|tool_start|>")
	toolEnd := tok.EncodeSpecial("<|tool_end|>")
	toolOutputStart := tok.EncodeSpecial("<|tool_output_start|>")
	toolOutputEnd := tok.EncodeSpecial("<|tool_output_end|>")

	for _, m := range messages {
		switch m.Role {
		case "user":
			ids = append(ids, userStart)
			ids = append(ids, tok.Encode(m.Content)...)
			ids = append(ids, userEnd)
		case "assistant":
			ids = append(ids, assistantStart)
			if m.Content != "" {
				ids = append(ids, tok.Encode(m.Content)...)
			}
			for _, tc := range m.ToolCalls {
				ids = append(ids, toolStart)
				ids = append(ids, tok.Encode(tc.Function.Arguments)...)
				ids = append(ids, toolEnd)
			}
			ids = append(ids, assistantEnd)
		case "tool":
			ids = append(ids, toolOutputStart)
			ids = append(ids, tok.Encode(m.Content)...)
			ids = append(ids, toolOutputEnd)
		}
	}
	ids = append(ids, assistantStart)
	return ids
}

// mergeSystemMessage folds a leading system message into the following user
// message, matching the tokenizer's conversation rendering.
func mergeSystemMessage(messages []ChatMessage) []ChatMessage {
	if len(messages) >= 2 && messages[0].Role == "system" && messages[1].Role == "user" {
		messages[1].Content = messages[0].Content + "\n\n" + messages[1].Content
		messages = messages[1:]
	}
	return messages
}

func toolsSystemPrompt(names []string) string {
	s := "You may call the following tools by writing " +
		"<|tool_start|>EXPRESSION<|tool_end|>. The result will be provided to you. Available tools: "
	for i, n := range names {
		if i > 0 {
			s += ", "
		}
		s += n
	}
	return s + "."
}

// generationOptions carries the decoded sampling parameters for a request.
type generationOptions struct {
	temperature float32
	topK        int
	maxTokens   int
	n           int
	seed        uint64
}

// options resolves the sampling parameters from a request.
func options(req ChatCompletionRequest, defaultMaxTokens int) generationOptions {
	temperature := float32(0.7)
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	n := req.N
	if n < 1 {
		n = 1
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	seed := uint64(42)
	if req.Seed != nil {
		seed = *req.Seed
	}
	return generationOptions{
		temperature: temperature,
		topK:        req.TopK,
		maxTokens:   maxTokens,
		n:           n,
		seed:        seed,
	}
}
