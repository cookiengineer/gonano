package evaluator

import (
	"testing"

	"github.com/cookiengineer/gonano/tokenizer"
)

func traceTestTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func TestEvaluateTraceFormat(t *testing.T) {
	tok := traceTestTokenizer()
	start := tok.EncodeSpecial("<|think_start|>")
	end := tok.EncodeSpecial("<|think_end|>")
	open := func(ids ...int) []int { return ids }

	sequences := [][]int{
		open(start, 65, 66, end, 67), // trace with content, closed
		open(start, 65, end),         // trace with content, closed
		open(start, 65, 66),          // trace with content, not closed
		open(start, end, 65),         // empty trace, closed
		open(65, 66, 67),             // no trace at all
	}
	format := EvaluateTraceFormat(tok, sequences)
	if format.Total != 5 {
		t.Fatalf("total = %d, want 5", format.Total)
	}
	if format.WithTrace != 3 {
		t.Fatalf("with trace = %d, want 3", format.WithTrace)
	}
	if format.Closed != 2 {
		t.Fatalf("closed = %d, want 2", format.Closed)
	}
	if format.WithTraceRatio() != 0.6 {
		t.Fatalf("with trace ratio = %v, want 0.6", format.WithTraceRatio())
	}
	if format.ClosedRatio() != 0.4 {
		t.Fatalf("closed ratio = %v, want 0.4", format.ClosedRatio())
	}
	if format.AvgReasoningTokens <= 0 {
		t.Fatalf("average reasoning tokens = %v, want > 0", format.AvgReasoningTokens)
	}
}
