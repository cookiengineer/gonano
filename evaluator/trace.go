package evaluator

import (
	"github.com/cookiengineer/gonano/evaluator/tasks"
	"github.com/cookiengineer/gonano/inference"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tokenizer"
)

// TraceFormat summarizes the structure of generated reasoning traces.
type TraceFormat struct {
	Total              int     // completions scored
	WithTrace          int     // completions with a non-empty <|think_start|> block
	Closed             int     // completions whose trace is closed with <|think_end|>
	AvgReasoningTokens float64 // average reasoning tokens per completion
}

// WithTraceRatio returns the fraction of completions that produced a non-empty
// reasoning trace.
func (format TraceFormat) WithTraceRatio() float32 {
	if format.Total == 0 {
		return 0
	}
	return float32(format.WithTrace) / float32(format.Total)
}

// ClosedRatio returns the fraction of completions that closed their reasoning
// trace before the final answer.
func (format TraceFormat) ClosedRatio() float32 {
	if format.Total == 0 {
		return 0
	}
	return float32(format.Closed) / float32(format.Total)
}

// EvaluateTraceFormat scores token sequences (prompt plus completion) for
// reasoning-trace structure. It is model-free so it can also be used to score
// stored rollouts.
func EvaluateTraceFormat(tok *tokenizer.Tokenizer, sequences [][]int) TraceFormat {
	start := tok.EncodeSpecial("<|think_start|>")
	end := tok.EncodeSpecial("<|think_end|>")
	format := TraceFormat{Total: len(sequences)}
	var reasoningTotal int
	for _, ids := range sequences {
		inThinking := false
		reasoning := 0
		closed := false
		for _, id := range ids {
			switch {
			case id == start:
				inThinking = true
				reasoning = 0
			case id == end:
				if inThinking && reasoning > 0 {
					closed = true
				}
				inThinking = false
			case inThinking:
				reasoning++
			}
		}
		if reasoning > 0 {
			format.WithTrace++
		}
		if closed {
			format.Closed++
		}
		reasoningTotal += reasoning
	}
	if format.Total > 0 {
		format.AvgReasoningTokens = float64(reasoningTotal) / float64(format.Total)
	}
	return format
}

// ReasoningAccuracy evaluates a generative task like GenerativeAccuracy and, in
// addition, reports the reasoning-trace format over every sampled completion.
func ReasoningAccuracy(task tasks.Task, transformer *model.Transformer, tok *tokenizer.Tokenizer, engine *inference.Engine, numSamples, maxTokens int, temperature float32, topK int) (float32, TraceFormat) {
	total := task.NumExamples()
	if total == 0 {
		return 0, TraceFormat{}
	}
	passed := 0
	var sequences [][]int
	for exampleIndex := 0; exampleIndex < total; exampleIndex++ {
		conversation := task.GetExample(exampleIndex)
		prompt := tok.RenderForCompletion(conversation)
		results, _ := engine.GenerateBatch(prompt, numSamples, maxTokens, temperature, topK, uint64(42+exampleIndex))
		prefix := len(prompt)
		ok := false
		for _, result := range results {
			sequences = append(sequences, result)
			if task.Evaluate(conversation, tok.Decode(result[prefix:])) {
				ok = true
			}
		}
		if ok {
			passed++
		}
	}
	return float32(passed) / float32(total), EvaluateTraceFormat(tok, sequences)
}
