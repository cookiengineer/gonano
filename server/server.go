package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"

	"github.com/cookiengineer/gonano/bank"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/router"
	"github.com/cookiengineer/gonano/tokenizer"
)

// Server serves an OpenAI-compatible chat completions API backed by a gonano
// model. Model tool calls are returned to the client using OpenAI's JSON
// function-call protocol; legacy expression calls can still be dispatched to the
// server-side Tools registry as a fallback.
type Server struct {
	Model     *model.Transformer
	Tokenizer *tokenizer.Tokenizer
	Engine    *inference.Engine
	Tools     *inference.Registry
	ModelName string

	// Scheduler, when non-nil, serves single-model requests through the
	// batched scheduler. It is opt-in so prefix-cache and bank modes keep using
	// the engine.
	Scheduler              *inference.SchedulerRunner
	SchedulerPrefixEntries int
	SchedulerPreemption    inference.PreemptionPolicy
	schedulerGenerator     tokenGenerator

	domainMutex                  sync.Mutex
	domainEngines                map[string]*inference.Engine
	domainRunners                map[string]*inference.SchedulerRunner
	domainGenerators             map[string]tokenGenerator
	useDomainSchedulers          bool
	domainSchedulerMaxActive     int
	domainSchedulerPrefillChunk  int
	domainSchedulerPrefixEntries int
	useEnsembleSchedulers        bool
	ensembleRunners              map[string]*inference.SchedulerRunner
	ensembleFallbacks            map[string]*inference.Ensemble

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

// StartScheduler enables the batched scheduler for single-model requests.
// maxActive <= 0 leaves the engine in charge. Prefix-cache and speculative
// requests transparently fall back to the engine.
func (server *Server) StartScheduler(maxActive, prefillChunk int) {
	if maxActive <= 0 || server.Scheduler != nil {
		return
	}
	scheduler := inference.NewScheduler(server.Engine, maxActive)
	scheduler.PrefillChunk = prefillChunk
	scheduler.Preemption = server.SchedulerPreemption
	if server.SchedulerPrefixEntries > 0 {
		scheduler.Prefix = inference.NewBlockPrefixCache(server.SchedulerPrefixEntries, 0)
	}
	runner := inference.NewSchedulerRunner(scheduler)
	runner.Start()
	server.Scheduler = runner
	server.schedulerGenerator = &schedulerGenerator{
		runner:   runner,
		fallback: server.Engine,
		engine:   server.Engine,
	}
}

// EnableDomainSchedulers makes routed single-domain bank requests use a
// per-domain scheduler. Multi-domain blends continue to use the Ensemble.
func (server *Server) EnableDomainSchedulers(maxActive, prefillChunk, prefixEntries int) {
	if maxActive <= 0 {
		return
	}
	server.domainMutex.Lock()
	defer server.domainMutex.Unlock()
	server.useDomainSchedulers = true
	server.domainSchedulerMaxActive = maxActive
	server.domainSchedulerPrefillChunk = prefillChunk
	server.domainSchedulerPrefixEntries = prefixEntries
	if server.domainEngines == nil {
		server.domainEngines = make(map[string]*inference.Engine)
	}
	if server.domainRunners == nil {
		server.domainRunners = make(map[string]*inference.SchedulerRunner)
	}
	if server.domainGenerators == nil {
		server.domainGenerators = make(map[string]tokenGenerator)
	}
	if server.ensembleRunners == nil {
		server.ensembleRunners = make(map[string]*inference.SchedulerRunner)
	}
	if server.ensembleFallbacks == nil {
		server.ensembleFallbacks = make(map[string]*inference.Ensemble)
	}
	server.useEnsembleSchedulers = true
}

// ensembleSchedulerFor returns a shared ensemble scheduler for a domain set,
// its Ensemble fallback, and the weights in sorted-domain order.
func (server *Server) ensembleSchedulerFor(scores []router.Score) (*inference.SchedulerRunner, *inference.Ensemble, []float32, []string) {
	sorted := append([]router.Score(nil), scores...)
	sort.SliceStable(sorted, func(left, right int) bool { return sorted[left].Domain < sorted[right].Domain })
	domains := make([]string, len(sorted))
	weights := make([]float32, len(sorted))
	for index, score := range sorted {
		domains[index] = score.Domain
		weights[index] = score.Value
	}
	key := strings.Join(domains, ",")

	server.domainMutex.Lock()
	defer server.domainMutex.Unlock()
	if runner := server.ensembleRunners[key]; runner != nil {
		return runner, server.ensembleFallbacks[key], weights, domains
	}
	members := make([]inference.ModelWeight, 0, len(domains))
	for index, domain := range domains {
		transformer, err := server.Bank.Acquire(domain)
		if err != nil {
			slog.Warn("domain model unavailable", "domain", domain, "err", err)
			return nil, nil, nil, nil
		}
		if transformer.Config.CEDEnabled() {
			return nil, nil, nil, nil
		}
		members = append(members, inference.ModelWeight{Model: transformer, Weight: weights[index], Domain: domain})
	}
	ensemble := inference.NewEnsemble(members, server.Tokenizer)
	ensemble.Tools = server.Tools
	scheduler := inference.NewEnsembleScheduler(ensemble, server.domainSchedulerMaxActive)
	scheduler.PrefillChunk = server.domainSchedulerPrefillChunk
	runner := inference.NewEnsembleSchedulerRunner(scheduler)
	runner.Start()
	server.ensembleRunners[key] = runner
	server.ensembleFallbacks[key] = ensemble
	return runner, ensemble, weights, domains
}

// schedulerForDomain returns (or lazily builds) the scheduler for a bank
// domain. It returns nil when the domain model cannot be acquired.
func (server *Server) schedulerForDomain(domain string) tokenGenerator {
	server.domainMutex.Lock()
	defer server.domainMutex.Unlock()
	if generator := server.domainGenerators[domain]; generator != nil {
		return generator
	}
	transformer, err := server.Bank.Acquire(domain)
	if err != nil {
		slog.Warn("domain model unavailable", "domain", domain, "err", err)
		return nil
	}
	engine := inference.NewEngine(transformer, server.Tokenizer)
	engine.Tools = server.Tools
	engine.PrefillChunk = server.domainSchedulerPrefillChunk
	scheduler := inference.NewScheduler(engine, server.domainSchedulerMaxActive)
	scheduler.PrefillChunk = server.domainSchedulerPrefillChunk
	scheduler.Preemption = server.SchedulerPreemption
	if server.domainSchedulerPrefixEntries > 0 {
		scheduler.Prefix = inference.NewBlockPrefixCache(server.domainSchedulerPrefixEntries, 0)
	}
	runner := inference.NewSchedulerRunner(scheduler)
	runner.Start()
	generator := &schedulerGenerator{runner: runner, fallback: engine, engine: engine}
	server.domainEngines[domain] = engine
	server.domainRunners[domain] = runner
	server.domainGenerators[domain] = generator
	return generator
}

// Close stops a running scheduler.
func (server *Server) Close() {
	if server.Scheduler != nil {
		server.Scheduler.Stop()
		server.Scheduler = nil
		server.schedulerGenerator = nil
	}
	server.domainMutex.Lock()
	for _, runner := range server.domainRunners {
		runner.Stop()
	}
	for _, runner := range server.ensembleRunners {
		runner.Stop()
	}
	server.domainRunners = nil
	server.ensembleRunners = nil
	server.ensembleFallbacks = nil
	server.useEnsembleSchedulers = false
	server.domainGenerators = nil
	server.domainEngines = nil
	server.useDomainSchedulers = false
	server.domainMutex.Unlock()
}

// schedulerGenerator adapts SchedulerRunner to the tokenGenerator interface.
// Requests that need engine-only features (multi-sample, speculative decode or
// an engine prefix cache) fall back to the engine.
type schedulerGenerator struct {
	runner   *inference.SchedulerRunner
	fallback tokenGenerator
	engine   *inference.Engine
}

func (generator *schedulerGenerator) GenerateWith(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64, options inference.GenerateOptions) func(yield func([]int, []int) bool) {
	engine := generator.engine
	if numSamples != 1 || engine.Cache != nil || engine.SWACache != nil || engine.BlockPrefix != nil {
		return generator.fallback.GenerateWith(tokens, numSamples, maxTokens, temperature, topK, seed, options)
	}
	return func(yield func([]int, []int) bool) {
		schedulerOptions := options
		schedulerOptions.OnLogprobs = nil
		submission := generator.runner.Submit(inference.Request{
			Prompt:      tokens,
			MaxTokens:   maxTokens,
			Temperature: temperature,
			TopK:        topK,
			TopP:        options.TopP,
			MinP:        options.MinP,
			Seed:        seed,
			Options:     schedulerOptions,
			Constraint:  options.Constraint,
		})
		defer submission.Cancel()
		for {
			event, ok := submission.Next()
			if !ok || event.Finished {
				return
			}
			if event.Logprob != nil && options.OnLogprobs != nil {
				options.OnLogprobs([]inference.TokenLogprob{*event.Logprob})
			}
			if !yield([]int{event.Token}, []int{event.TokenMask}) {
				return
			}
		}
	}
}

// ensembleSchedulerGenerator adapts a shared EnsembleScheduler runner to the
// tokenGenerator contract. Requests with different weights can share the same
// runner because weights are per submission.
type ensembleSchedulerGenerator struct {
	runner   *inference.SchedulerRunner
	fallback *inference.Ensemble
	weights  []float32
}

func (generator *ensembleSchedulerGenerator) GenerateWith(tokens []int, numSamples, maxTokens int, temperature float32, topK int, seed uint64, options inference.GenerateOptions) func(yield func([]int, []int) bool) {
	if numSamples != 1 {
		return generator.fallback.GenerateWith(tokens, numSamples, maxTokens, temperature, topK, seed, options)
	}
	return func(yield func([]int, []int) bool) {
		schedulerOptions := options
		schedulerOptions.OnLogprobs = nil
		submission := generator.runner.Submit(inference.Request{
			Prompt:      tokens,
			MaxTokens:   maxTokens,
			Temperature: temperature,
			TopK:        topK,
			TopP:        options.TopP,
			MinP:        options.MinP,
			Seed:        seed,
			Options:     schedulerOptions,
			Constraint:  options.Constraint,
			Weights:     generator.weights,
		})
		defer submission.Cancel()
		for {
			event, ok := submission.Next()
			if !ok || event.Finished {
				return
			}
			if event.Logprob != nil && options.OnLogprobs != nil {
				options.OnLogprobs([]inference.TokenLogprob{*event.Logprob})
			}
			if !yield([]int{event.Token}, []int{event.TokenMask}) {
				return
			}
		}
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
	logprobs   []ChatLogprob
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
		if server.schedulerGenerator != nil {
			return server.schedulerGenerator, nil
		}
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
				if server.useDomainSchedulers {
					if generator := server.schedulerForDomain(requested); generator != nil {
						return generator, []string{requested}
					}
				}
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
	if len(scores) == 1 && server.useDomainSchedulers {
		if generator := server.schedulerForDomain(scores[0].Domain); generator != nil {
			return generator, []string{scores[0].Domain}
		}
	}
	if len(scores) > 1 && server.useEnsembleSchedulers {
		if runner, fallback, weights, domains := server.ensembleSchedulerFor(scores); runner != nil {
			return &ensembleSchedulerGenerator{runner: runner, fallback: fallback, weights: weights}, domains
		}
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
func (server *Server) generate(generator tokenGenerator, prompt []int, genOptions generationOptions, thinking thinkingOptions) []rowResult {
	tokenizerImpl := server.Tokenizer
	numSamples := genOptions.numSamples
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
	logprobRows := make([][]ChatLogprob, numSamples)
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

	generate := generator.GenerateWith(prompt, numSamples, genOptions.maxTokens, genOptions.temperature, genOptions.topK, genOptions.seed,
		inference.GenerateOptions{
			Thinking: thinking.enabled || thinking.budget > 0, ThinkingBudget: thinking.budget,
			TopP: genOptions.topP, MinP: genOptions.minP, Constraint: genOptions.constraint,
			Logprobs: genOptions.logprobs, TopLogprobs: genOptions.topLogprobs,
			ExternalTools: true,
			OnLogprobs: func(rows []inference.TokenLogprob) {
				for index, row := range rows {
					if index < len(logprobRows) {
						logprobRows[index] = append(logprobRows[index], convertLogprob(tokenizerImpl, row))
					}
				}
			},
		})
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
					if call, ok := server.parseToolCall(callArgs[index]); ok {
						call.ID = fmt.Sprintf("call_%d", callCounter)
						results[index].toolCalls = append(results[index].toolCalls, call)
						callCounter++
						finished[index] = true
						results[index].finish = "tool_calls"
					}
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
		if results[index].finish != "tool_calls" {
			results[index].finish = "stop"
			if !finished[index] {
				results[index].finish = "length"
			}
		}
		if trimmed, stopped := applyStop(results[index].content, genOptions.stop); stopped {
			results[index].content = trimmed
			results[index].finish = "stop"
		}
		results[index].promptLen = len(prompt)
		results[index].completion = len(sampled[index]) + len(reasoningTokens[index])
		results[index].logprobs = logprobRows[index]
		for _, toolCall := range results[index].toolCalls {
			results[index].completion += len(tokenizerImpl.Encode(toolCall.Function.Arguments))
		}
	}
	return results
}

func convertLogprob(tokenizerImpl *tokenizer.Tokenizer, row inference.TokenLogprob) ChatLogprob {
	converted := ChatLogprob{
		Token:   tokenizerImpl.Decode([]int{row.Token}),
		Logprob: row.Logprob,
	}
	for _, alternative := range row.Top {
		converted.TopLogprobs = append(converted.TopLogprobs, ChatTopLogprob{
			Token:   tokenizerImpl.Decode([]int{alternative.Token}),
			Logprob: alternative.Logprob,
		})
	}
	return converted
}

// stopIndex returns the earliest stop-string index, or -1.
func stopIndex(content string, stops []string) int {
	cutoff := -1
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		if index := strings.Index(content, stop); index >= 0 && (cutoff < 0 || index < cutoff) {
			cutoff = index
		}
	}
	return cutoff
}

// applyStop trims content at the earliest stop string.
func applyStop(content string, stops []string) (string, bool) {
	cutoff := stopIndex(content, stops)
	if cutoff < 0 {
		return content, false
	}
	return content[:cutoff], true
}

// renderMessages converts OpenAI messages into a prompt token sequence primed
// for the assistant to complete. thinking is the optional request field: when
// non-nil it selects the thinking-mode instruction, and when enabled the
// assistant turn is primed with <|think_start|>.
func (server *Server) renderMessages(messages []ChatMessage, tools []ToolDef, reasoningEffort *int, thinking *bool, thinkingStyle string) []int {
	tokenizerImpl := server.Tokenizer
	ids := []int{tokenizerImpl.BOSTokenID()}

	var instructions []string
	if reasoningEffort != nil {
		instructions = append(instructions, tokenizer.ReasoningEffortInstruction(*reasoningEffort))
	}
	if thinking != nil {
		instructions = append(instructions, tokenizer.ThinkingInstruction(*thinking))
		if *thinking {
			if instruction := tokenizer.ThinkingStyleInstruction(thinkingStyle); instruction != "" {
				instructions = append(instructions, instruction)
			}
		}
	}
	if len(instructions) > 0 {
		messages = append([]ChatMessage{{Role: "system", Content: strings.Join(instructions, "\n")}}, messages...)
	}
	messages = mergeSystemMessage(messages)
	if len(tools) > 0 {
		// Advise the model about the tools it may invoke.
		messages = append([]ChatMessage{{Role: "system", Content: toolsSystemPrompt(tools)}}, messages...)
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
				var arguments any
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
					arguments = toolCall.Function.Arguments
				}
				payload, _ := json.Marshal(map[string]any{
					"name":      toolCall.Function.Name,
					"arguments": arguments,
				})
				ids = append(ids, toolStart)
				ids = append(ids, tokenizerImpl.Encode(string(payload))...)
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

func toolsSystemPrompt(tools []ToolDef) string {
	builder := strings.Builder{}
	builder.WriteString("You may call tools by writing <|tool_start|>{\"name\":\"tool\",\"arguments\":{...}}<|tool_end|>. ")
	builder.WriteString("Arguments must be a JSON object matching the tool schema. Available tools:\n")
	for _, tool := range tools {
		parameters, _ := json.Marshal(tool.Function.Parameters)
		builder.WriteString("- ")
		builder.WriteString(tool.Function.Name)
		if tool.Function.Description != "" {
			builder.WriteString(": ")
			builder.WriteString(tool.Function.Description)
		}
		builder.WriteString(" schema=")
		builder.Write(parameters)
		builder.WriteByte('\n')
	}
	return builder.String()
}

// parseToolCall parses the model's JSON tool-call envelope and falls back to
// the legacy expression protocol for older checkpoints.
func (server *Server) parseToolCall(tokens []int) (ToolCall, bool) {
	text := strings.TrimSpace(server.Tokenizer.Decode(tokens))
	if text == "" {
		return ToolCall{}, false
	}
	var envelope struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(text), &envelope); err == nil && envelope.Name != "" {
		arguments := "{}"
		if len(envelope.Arguments) > 0 {
			var asString string
			if json.Unmarshal(envelope.Arguments, &asString) == nil {
				arguments = asString
			} else {
				arguments = string(envelope.Arguments)
			}
		}
		return ToolCall{Type: "function", Function: FunctionCall{Name: envelope.Name, Arguments: arguments}}, true
	}
	if server.Tools != nil {
		if name, _, ok := server.Tools.ExecuteTool(text); ok {
			return ToolCall{Type: "function", Function: FunctionCall{Name: name, Arguments: text}}, true
		}
	}
	return ToolCall{}, false
}

// generationOptions carries the decoded sampling parameters for a request.
type generationOptions struct {
	temperature float32
	topK        int
	topP        float32
	minP        float32
	maxTokens   int
	numSamples  int
	seed        uint64
	stop        []string
	constraint  inference.Constraint
	logprobs    bool
	topLogprobs int
}

// resolveOptions resolves the sampling parameters from a request.
func constraintForResponseFormat(format *ResponseFormat) inference.Constraint {
	if format == nil {
		return nil
	}
	switch format.Type {
	case "json_object":
		return inference.NewJSONConstraint()
	case "json_schema":
		schema := format.JSONSchema
		if object, ok := schema.(map[string]any); ok {
			if nested, exists := object["schema"]; exists {
				schema = nested
			}
		}
		if constraint, err := inference.NewJSONSchemaConstraint(schema); err == nil {
			return constraint
		}
		return inference.NewJSONConstraint()
	case "regex":
		pattern := responseFormatText(format.JSONSchema, "pattern")
		if pattern == "" {
			return nil
		}
		if constraint, err := inference.NewRegexConstraint(pattern); err == nil {
			return constraint
		}
	case "ebnf":
		grammar := responseFormatText(format.JSONSchema, "grammar")
		if grammar == "" {
			return nil
		}
		if constraint, err := inference.NewEBNFConstraint(grammar); err == nil {
			return constraint
		}
	}
	return nil
}

func responseFormatText(value any, key string) string {
	switch typed := value.(type) {
	case string:
		return typed
	case map[string]any:
		if text, ok := typed[key].(string); ok {
			return text
		}
	}
	return ""
}

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
	topP := float32(1)
	if chatRequest.TopP != nil {
		topP = *chatRequest.TopP
	}
	minP := float32(0)
	if chatRequest.MinP != nil {
		minP = *chatRequest.MinP
	}
	constraint := constraintForResponseFormat(chatRequest.ResponseFormat)
	return generationOptions{
		temperature: temperature,
		topK:        chatRequest.TopK,
		topP:        topP,
		minP:        minP,
		maxTokens:   maxTokens,
		numSamples:  numSamples,
		seed:        seed,
		stop:        append([]string(nil), chatRequest.Stop...),
		constraint:  constraint,
		logprobs:    chatRequest.Logprobs,
		topLogprobs: chatRequest.TopLogprobs,
	}
}
