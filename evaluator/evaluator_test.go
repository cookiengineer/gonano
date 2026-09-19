package evaluator

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func evalModel() *model.Transformer {
	config := model.Config{
		SequenceLen: 128, VocabSize: 265, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	return transformer
}

func evalTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func TestBitsPerByteFinite(tester *testing.T) {
	transformer := evalModel()
	tokenizer := evalTokenizer()

	tokenBytes := make([]int32, tokenizer.VocabSize())
	for index := 0; index < tokenizer.VocabSize(); index++ {
		if tokenizer.IsSpecial(index) {
			tokenBytes[index] = 0
		} else {
			tokenBytes[index] = 1
		}
	}

	batches := func() (*tensors.Int32s, *tensors.Int32s) {
		inputs := tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 2, 3, 4, 5, 6})
		targets := tensors.NewInt32sWithData([]int{1, 6}, []int32{2, 3, 4, 5, 6, 7})
		return inputs, targets
	}

	bpb := BitsPerByte(transformer, batches, 1, tokenBytes)
	if math.IsNaN(float64(bpb)) || math.IsInf(float64(bpb), 0) {
		tester.Fatalf("bpb = %v, want finite", bpb)
	}
	if bpb <= 0 {
		tester.Fatalf("bpb = %v, want > 0", bpb)
	}
}

func TestChatCORE(tester *testing.T) {
	accuracy := map[string]float32{"A": 0.25, "B": 0.5}
	baselines := map[string]float32{"A": 0.25, "B": 0.0}
	result := ChatCORE(accuracy, baselines)
	// (0.25-0.25)/(0.75) + (0.5-0)/(1.0) = 0 + 0.5 = 0.5, /2 = 0.25.
	if math.Abs(float64(result-0.25)) > 1e-5 {
		tester.Fatalf("ChatCORE = %v, want 0.25", result)
	}
}

func TestEvaluateTaskMultipleChoice(tester *testing.T) {
	transformer := evalModel()
	tokenizer := evalTokenizer()
	data := []CoreExample{
		{"query": "What is 2+2?", "choices": []string{"3", "4", "5", "6"}, "gold": 1},
		{"query": "What is the capital of France?", "choices": []string{"London", "Paris", "Berlin", "Rome"}, "gold": 1},
	}
	taskMeta := TaskMeta{TaskType: "multiple_choice", NumFewshot: 0, ContinuationDelimiter: " "}
	accuracy := EvaluateTask(transformer, tokenizer, data, taskMeta)
	if accuracy < 0 || accuracy > 1 {
		tester.Fatalf("accuracy = %v, want in [0,1]", accuracy)
	}
}

func TestEvaluateExampleSingle(tester *testing.T) {
	transformer := evalModel()
	tokenizer := evalTokenizer()
	data := []CoreExample{
		{"query": "Pick the largest", "choices": []string{"one", "ten", "three", "five"}, "gold": 1},
	}
	taskMeta := TaskMeta{TaskType: "multiple_choice", NumFewshot: 0, ContinuationDelimiter: " "}
	// Just verify it runs and returns a boolean.
	result := EvaluateExample(transformer, tokenizer, data, 0, taskMeta)
	if result != true && result != false {
		tester.Fatalf("EvaluateExample returned non-bool")
	}
}
