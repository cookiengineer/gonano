package eval

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
)

func evalModel() *model.Transformer {
	cfg := model.Config{
		SequenceLen: 128, VocabSize: 265, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(42))
	return m
}

func evalTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func TestBitsPerByteFinite(t *testing.T) {
	m := evalModel()
	tok := evalTokenizer()

	tokenBytes := make([]int32, tok.VocabSize())
	for i := 0; i < tok.VocabSize(); i++ {
		if tok.IsSpecial(i) {
			tokenBytes[i] = 0
		} else {
			tokenBytes[i] = 1
		}
	}

	batches := func() (*tensor.Int32s, *tensor.Int32s) {
		x := tensor.NewInt32sWithData([]int{1, 6}, []int32{1, 2, 3, 4, 5, 6})
		y := tensor.NewInt32sWithData([]int{1, 6}, []int32{2, 3, 4, 5, 6, 7})
		return x, y
	}

	bpb := BitsPerByte(m, batches, 1, tokenBytes)
	if math.IsNaN(float64(bpb)) || math.IsInf(float64(bpb), 0) {
		t.Fatalf("bpb = %v, want finite", bpb)
	}
	if bpb <= 0 {
		t.Fatalf("bpb = %v, want > 0", bpb)
	}
}

func TestChatCORE(t *testing.T) {
	acc := map[string]float32{"A": 0.25, "B": 0.5}
	base := map[string]float32{"A": 0.25, "B": 0.0}
	got := ChatCORE(acc, base)
	// (0.25-0.25)/(0.75) + (0.5-0)/(1.0) = 0 + 0.5 = 0.5, /2 = 0.25.
	if math.Abs(float64(got-0.25)) > 1e-5 {
		t.Fatalf("ChatCORE = %v, want 0.25", got)
	}
}

func TestEvaluateTaskMultipleChoice(t *testing.T) {
	m := evalModel()
	tok := evalTokenizer()
	data := []CoreExample{
		{"query": "What is 2+2?", "choices": []string{"3", "4", "5", "6"}, "gold": 1},
		{"query": "What is the capital of France?", "choices": []string{"London", "Paris", "Berlin", "Rome"}, "gold": 1},
	}
	meta := TaskMeta{TaskType: "multiple_choice", NumFewshot: 0, ContinuationDelimiter: " "}
	acc := EvaluateTask(m, tok, data, meta)
	if acc < 0 || acc > 1 {
		t.Fatalf("accuracy = %v, want in [0,1]", acc)
	}
}

func TestEvaluateExampleSingle(t *testing.T) {
	m := evalModel()
	tok := evalTokenizer()
	data := []CoreExample{
		{"query": "Pick the largest", "choices": []string{"one", "ten", "three", "five"}, "gold": 1},
	}
	meta := TaskMeta{TaskType: "multiple_choice", NumFewshot: 0, ContinuationDelimiter: " "}
	// Just verify it runs and returns a boolean.
	got := EvaluateExample(m, tok, data, 0, meta)
	if got != true && got != false {
		t.Fatalf("EvaluateExample returned non-bool")
	}
}
