package server

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/cookiengineer/gonano/bank"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/router"
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

	// Bank and Router, when both non-nil, enable domain routing: each request
	// is classified by Router and only the selected domain models are loaded
	// (via Bank) and blended (via inference.Ensemble).
	Bank   *bank.Bank
	Router *router.Router
	// MaxDomains caps how many routed domains are blended per request
	// (default 1).
	MaxDomains int
	// MinDomainScore is the minimum router probability to include an extra
	// domain in the blend.
	MinDomainScore float32
	// RouteScope selects what the router classifies: "last-turn" (default)
	// uses the latest user message, "full-prompt" uses the whole rendered
	// prompt.
	RouteScope string
}

// routeScopeLastTurn routes on the latest user message.
const routeScopeLastTurn = "last-turn"

// routingScope returns the effective route scope.
func (server *Server) routingScope() string {
	if server.RouteScope == "" {
		return routeScopeLastTurn
	}
	return server.RouteScope
}

// lastUserContent returns the content of the most recent user message.
func lastUserContent(messages []ChatMessage) string {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == "user" {
			return messages[index].Content
		}
	}
	return ""
}

// tokenGenerator is the shared generation contract of the single-model Engine
// and the blended Ensemble.
type tokenGenerator interface {
	GenerateWith(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64, options inference.GenerateOptions) func(yield func([]int, []int) bool)
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
	reasoning  string
	toolCalls  []ToolCall
	finish     string
	promptLen  int
	completion int
}

// thinkingOptions is the resolved thinking configuration of a request.
type thinkingOptions struct {
	enabled bool
	budget  int
}

// resolveThinking resolves the thinking enable flag and budget from a request.
func resolveThinking(chatRequest ChatCompletionRequest) thinkingOptions {
	options := thinkingOptions{}
	if chatRequest.Thinking != nil {
		options.enabled = *chatRequest.Thinking
	}
	if chatRequest.ThinkingBudget != nil && *chatRequest.ThinkingBudget > 0 {
		options.budget = *chatRequest.ThinkingBudget
		options.enabled = true
	}
	return options
}

// selectGenerator chooses the generator and routed domain names for a request.
// It routes on the last user turn by default (RouteScope "last-turn") or the
// full prompt ("full-prompt"), and it treats a requested model name that names
// a bank domain as a hard pin.
func (server *Server) selectGenerator(chatRequest ChatCompletionRequest, prompt []int) (tokenGenerator, []string) {
	if server.Bank == nil || server.Router == nil {
		return server.Engine, nil
	}
	maxDomains := server.MaxDomains
	if maxDomains < 1 {
		maxDomains = 1
	}

	// A requested domain pins routing (e.g. the OpenAI "model" field).
	if requested := chatRequest.Model; requested != "" {
		for _, domain := range server.Router.Domains {
			if domain == requested {
				return server.ensembleForScores([]router.Score{{Domain: requested, Value: 1}}), []string{requested}
			}
		}
	}

	routePrompt := prompt
	if server.routingScope() == routeScopeLastTurn {
		if last := lastUserContent(chatRequest.Messages); last != "" {
			routePrompt = server.Tokenizer.Encode(last)
		}
	}
	scores := server.Router.TopDomains(server.toTokenIDs(routePrompt), maxDomains, server.MinDomainScore)
	if len(scores) == 0 {
		return server.Engine, nil
	}
	ids := make([]string, 0, len(scores))
	routed := make([]string, 0, len(scores))
	for _, score := range scores {
		ids = append(ids, score.Domain)
		routed = append(routed, fmt.Sprintf("%s=%.3f", score.Domain, score.Value))
	}
	generator := server.ensembleForScores(scores)
	if generator == nil {
		return server.Engine, nil
	}
	slog.Info("domain route", "domains", routed)
	return generator, ids
}

// ensembleForScores acquires the selected domains and builds a blended
// generator. It returns nil when no model could be loaded.
func (server *Server) ensembleForScores(scores []router.Score) tokenGenerator {
	weights := make([]inference.ModelWeight, 0, len(scores))
	for _, score := range scores {
		transformer, err := server.Bank.Acquire(score.Domain)
		if err != nil {
			slog.Warn("domain model unavailable", "domain", score.Domain, "err", err)
			continue
		}
		weights = append(weights, inference.ModelWeight{Model: transformer, Weight: score.Value, Domain: score.Domain})
	}
	if len(weights) == 0 {
		return nil
	}
	return inference.NewEnsemble(weights, server.Tokenizer)
}

// toTokenIDs converts prompt token ids to the int32 encoding the router uses.
func (server *Server) toTokenIDs(prompt []int) []int32 {
	ids := make([]int32, len(prompt))
	for index, token := range prompt {
		ids[index] = int32(token)
	}
	return ids
}

// generate runs n samples of autoregressive generation and parses each row
// into its natural-language content, reasoning trace, tool calls, and finish
// reason.
func (server *Server) generate(generator tokenGenerator, prompt []int, temperature float32, topK, maxTokens, numSamples int, seed uint64, thinking thinkingOptions) []rowResult {
	tokenizerImpl := server.Tokenizer
	results := make([]rowResult, numSamples)

	toolStart := tokenizerImpl.EncodeSpecial("<|tool_start|>")
	toolEnd := tokenizerImpl.EncodeSpecial("<|tool_end|>")
	assistantEnd := tokenizerImpl.EncodeSpecial("<|assistant_end|>")
	thinkStart := tokenizerImpl.EncodeSpecial("<|think_start|>")
	thinkEnd := tokenizerImpl.EncodeSpecial("<|think_end|>")
	bosToken := tokenizerImpl.BOSTokenID()

	var sampled [][]int
	var callArgs [][]int
	var reasoningTokens [][]int
	inToolCall := make([]bool, numSamples)
	inThinking := make([]bool, numSamples)
	finished := make([]bool, numSamples)
	sampled = make([][]int, numSamples)
	callArgs = make([][]int, numSamples)
	reasoningTokens = make([][]int, numSamples)
	callCounter := 0

	// A prompt primed with <|think_start|> (thinking enabled) starts inside the
	// reasoning trace.
	primedThinking := len(prompt) > 0 && prompt[len(prompt)-1] == thinkStart
	for index := 0; index < numSamples; index++ {
		inThinking[index] = primedThinking
	}

	generate := generator.GenerateWith(prompt, numSamples, maxTokens, temperature, topK, seed,
		inference.GenerateOptions{Thinking: thinking.enabled || thinking.budget > 0, ThinkingBudget: thinking.budget})
	generate(func(column, mask []int) bool {
		for index := 0; index < numSamples; index++ {
			if finished[index] {
				continue
			}
			token := column[index]
			// Thinking delimiters matter even when forced by the budget, so
			// process them before the forced-token filter below.
			switch token {
			case thinkStart:
				inThinking[index] = true
				reasoningTokens[index] = nil
				continue
			case thinkEnd:
				inThinking[index] = false
				continue
			}
			if mask[index] == 0 {
				// Remaining forced tokens are tool outputs; the model already
				// saw them, so they must not appear in the response.
				continue
			}
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
				switch {
				case inToolCall[index]:
					callArgs[index] = append(callArgs[index], token)
				case inThinking[index]:
					reasoningTokens[index] = append(reasoningTokens[index], token)
				default:
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
		results[index].reasoning = tokenizerImpl.Decode(reasoningTokens[index])
		results[index].finish = "stop"
		if !finished[index] {
			results[index].finish = "length"
		}
		results[index].promptLen = len(prompt)
		results[index].completion = len(sampled[index]) + len(reasoningTokens[index])
		for _, toolCall := range results[index].toolCalls {
			results[index].completion += len(tokenizerImpl.Encode(toolCall.Function.Arguments))
		}
	}
	return results
}

// renderMessages converts OpenAI messages into a prompt token sequence primed
// for the assistant to complete. thinking is the optional request field: when
// non-nil it selects the thinking-mode instruction, and when enabled the
// assistant turn is primed with <|think_start|>.
func (server *Server) renderMessages(messages []ChatMessage, tools []ToolDef, reasoningEffort *int, thinking *bool) []int {
	tokenizerImpl := server.Tokenizer
	ids := []int{tokenizerImpl.BOSTokenID()}

	var instructions []string
	if reasoningEffort != nil {
		instructions = append(instructions, tokenizer.ReasoningEffortInstruction(*reasoningEffort))
	}
	if thinking != nil {
		instructions = append(instructions, tokenizer.ThinkingInstruction(*thinking))
	}
	if len(instructions) > 0 {
		messages = append([]ChatMessage{{Role: "system", Content: strings.Join(instructions, "\n")}}, messages...)
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
	thinkStart := tokenizerImpl.EncodeSpecial("<|think_start|>")
	thinkEnd := tokenizerImpl.EncodeSpecial("<|think_end|>")

	for _, message := range messages {
		switch message.Role {
		case "user":
			ids = append(ids, userStart)
			ids = append(ids, tokenizerImpl.Encode(message.Content)...)
			ids = append(ids, userEnd)
		case "assistant":
			ids = append(ids, assistantStart)
			if message.ReasoningContent != "" {
				ids = append(ids, thinkStart)
				ids = append(ids, tokenizerImpl.Encode(message.ReasoningContent)...)
				ids = append(ids, thinkEnd)
			}
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
	if thinking != nil && *thinking {
		ids = append(ids, thinkStart)
	}
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
