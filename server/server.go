package server

import (
	"fmt"

	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tokenizer"
)

// Server serves an OpenAI-compatible chat completions API backed by a gonano
// model. Tool calls are executed server-side through the Tools registry, so a
// single request can run a complete tool-using turn.
type Server struct {
	Model     *model.Transformer
	Tokenizer *tokenizer.Tokenizer
	Engine    *inference.Engine
	Tools     *inference.Registry
	ModelName string
}

// NewServer builds a Server. tools may be nil, in which case the engine's
// built-in calculator registry is used.
func NewServer(transformer *model.Transformer, tokenizerImpl *tokenizer.Tokenizer, tools *inference.Registry, modelName string) *Server {
	engine := inference.NewEngine(transformer, tokenizerImpl)
	if tools != nil {
		engine.Tools = tools
	}
	return &Server{
		Model:     transformer,
		Tokenizer: tokenizerImpl,
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
func (server *Server) generate(prompt []int, temperature float32, topK, maxTokens, numSamples int, seed uint64) []rowResult {
	tokenizerImpl := server.Tokenizer
	results := make([]rowResult, numSamples)

	toolStart := tokenizerImpl.EncodeSpecial("<|tool_start|>")
	toolEnd := tokenizerImpl.EncodeSpecial("<|tool_end|>")
	assistantEnd := tokenizerImpl.EncodeSpecial("<|assistant_end|>")
	bosToken := tokenizerImpl.BOSTokenID()

	var sampled [][]int
	var callArgs [][]int
	inToolCall := make([]bool, numSamples)
	finished := make([]bool, numSamples)
	sampled = make([][]int, numSamples)
	callArgs = make([][]int, numSamples)
	callCounter := 0

	generate := server.Engine.Generate(prompt, numSamples, maxTokens, temperature, topK, seed)
	generate(func(column, mask []int) bool {
		for index := 0; index < numSamples; index++ {
			if finished[index] {
				continue
			}
			if mask[index] == 0 {
				// Forced tokens are tool outputs; the model already saw them.
				continue
			}
			token := column[index]
			switch token {
			case toolStart:
				inToolCall[index] = true
				callArgs[index] = nil
			case toolEnd:
				if inToolCall[index] {
					arguments := tokenizerImpl.Decode(callArgs[index])
					name, _, ok := server.Tools.ExecuteTool(arguments)
					if !ok {
						name = "unknown"
					}
					results[index].toolCalls = append(results[index].toolCalls, ToolCall{
						ID:       fmt.Sprintf("call_%d", callCounter),
						Type:     "function",
						Function: FunctionCall{Name: name, Arguments: arguments},
					})
					callCounter++
					inToolCall[index] = false
					callArgs[index] = nil
				}
			case assistantEnd, bosToken:
				finished[index] = true
			default:
				if inToolCall[index] {
					callArgs[index] = append(callArgs[index], token)
				} else {
					sampled[index] = append(sampled[index], token)
				}
			}
		}
		for _, done := range finished {
			if !done {
				return true
			}
		}
		return false
	})

	for index := 0; index < numSamples; index++ {
		results[index].content = tokenizerImpl.Decode(sampled[index])
		results[index].finish = "stop"
		if !finished[index] {
			results[index].finish = "length"
		}
		results[index].promptLen = len(prompt)
		results[index].completion = len(sampled[index])
		for _, toolCall := range results[index].toolCalls {
			results[index].completion += len(tokenizerImpl.Encode(toolCall.Function.Arguments))
		}
	}
	return results
}

// renderMessages converts OpenAI messages into a prompt token sequence primed
// for the assistant to complete.
func (server *Server) renderMessages(messages []ChatMessage, tools []ToolDef, reasoningEffort *int) []int {
	tokenizerImpl := server.Tokenizer
	ids := []int{tokenizerImpl.BOSTokenID()}

	if reasoningEffort != nil {
		messages = append([]ChatMessage{{Role: "system", Content: tokenizer.ReasoningEffortInstruction(*reasoningEffort)}}, messages...)
	}
	messages = mergeSystemMessage(messages)
	if len(tools) > 0 {
		// Advise the model about the tools it may invoke.
		names := make([]string, 0, len(tools))
		for _, tool := range tools {
			names = append(names, tool.Function.Name)
		}
		messages = append([]ChatMessage{{Role: "system", Content: toolsSystemPrompt(names)}}, messages...)
		messages = mergeSystemMessage(messages)
	}

	userStart := tokenizerImpl.EncodeSpecial("<|user_start|>")
	userEnd := tokenizerImpl.EncodeSpecial("<|user_end|>")
	assistantStart := tokenizerImpl.EncodeSpecial("<|assistant_start|>")
	assistantEnd := tokenizerImpl.EncodeSpecial("<|assistant_end|>")
	toolStart := tokenizerImpl.EncodeSpecial("<|tool_start|>")
	toolEnd := tokenizerImpl.EncodeSpecial("<|tool_end|>")
	toolOutputStart := tokenizerImpl.EncodeSpecial("<|tool_output_start|>")
	toolOutputEnd := tokenizerImpl.EncodeSpecial("<|tool_output_end|>")

	for _, message := range messages {
		switch message.Role {
		case "user":
			ids = append(ids, userStart)
			ids = append(ids, tokenizerImpl.Encode(message.Content)...)
			ids = append(ids, userEnd)
		case "assistant":
			ids = append(ids, assistantStart)
			if message.Content != "" {
				ids = append(ids, tokenizerImpl.Encode(message.Content)...)
			}
			for _, toolCall := range message.ToolCalls {
				ids = append(ids, toolStart)
				ids = append(ids, tokenizerImpl.Encode(toolCall.Function.Arguments)...)
				ids = append(ids, toolEnd)
			}
			ids = append(ids, assistantEnd)
		case "tool":
			ids = append(ids, toolOutputStart)
			ids = append(ids, tokenizerImpl.Encode(message.Content)...)
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
	prompt := "You may call the following tools by writing " +
		"<|tool_start|>EXPRESSION<|tool_end|>. The result will be provided to you. Available tools: "
	for index, name := range names {
		if index > 0 {
			prompt += ", "
		}
		prompt += name
	}
	return prompt + "."
}

// generationOptions carries the decoded sampling parameters for a request.
type generationOptions struct {
	temperature float32
	topK        int
	maxTokens   int
	numSamples  int
	seed        uint64
}

// resolveOptions resolves the sampling parameters from a request.
func resolveOptions(chatRequest ChatCompletionRequest, defaultMaxTokens int) generationOptions {
	temperature := float32(0.7)
	if chatRequest.Temperature != nil {
		temperature = *chatRequest.Temperature
	}
	numSamples := chatRequest.N
	if numSamples < 1 {
		numSamples = 1
	}
	maxTokens := chatRequest.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	seed := uint64(42)
	if chatRequest.Seed != nil {
		seed = *chatRequest.Seed
	}
	return generationOptions{
		temperature: temperature,
		topK:        chatRequest.TopK,
		maxTokens:   maxTokens,
		numSamples:  numSamples,
		seed:        seed,
	}
}
