package router

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// syntheticExamples builds three linearly separable-ish token classes.
func syntheticExamples() []Example {
	random := tensors.NewRNG(7)
	ranges := [][2]int32{{5, 9}, {50, 54}, {120, 124}}
	examples := make([]Example, 0, 60)
	for class, bounds := range ranges {
		for index := 0; index < 20; index++ {
			length := 4 + random.IntN(5)
			tokens := make([]int32, length)
			for position := range tokens {
				tokens[position] = bounds[0] + int32(random.IntN(int(bounds[1]-bounds[0])))
			}
			examples = append(examples, Example{Tokens: tokens, Label: class})
		}
	}
	return examples
}

// tinyEncoder builds a small frozen encoder for router tests.
func tinyEncoder(t *testing.T) *model.Transformer {
	t.Helper()
	configuration := model.Config{
		SequenceLen: 16, VocabSize: 256, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(configuration)
	transformer.InitWeights(tensors.NewRNG(1))
	return transformer
}

func accuracy(classifier interface{ Predict([]int32) []float32 }, examples []Example) float32 {
	correct := 0
	for _, example := range examples {
		if argMax(classifier.Predict(example.Tokens)) == example.Label {
			correct++
		}
	}
	return float32(correct) / float32(len(examples))
}

func TestNgramClassifierLearns(t *testing.T) {
	examples := syntheticExamples()
	classifier := TrainNgram(examples, 3, NgramConfig{Epochs: 15, Seed: 11})
	if got := accuracy(classifier, examples); got < 0.99 {
		t.Fatalf("n-gram training accuracy = %.3f, want >= 0.99", got)
	}
}

func TestTransformerHeadLearns(t *testing.T) {
	encoder := tinyEncoder(t)
	classifier := NewTransformerClassifier(encoder, 3, 0)
	classifier.Train(syntheticExamples(), HeadConfig{Epochs: 60, LearningRate: 0.05, BatchSize: 16, Seed: 3})
	if got := accuracy(classifier, syntheticExamples()); got < 0.9 {
		t.Fatalf("transformer head accuracy = %.3f, want >= 0.9", got)
	}
}

func TestTransformerFineTune(t *testing.T) {
	encoder := tinyEncoder(t)
	classifier := NewTransformerClassifier(encoder, 3, 0)
	classifier.FineTune(syntheticExamples(), 30, 1e-3, 5)
	if got := accuracy(classifier, syntheticExamples()); got < 0.9 {
		t.Fatalf("fine-tuned accuracy = %.3f, want >= 0.9", got)
	}
	for _, example := range syntheticExamples() {
		probabilities := classifier.Predict(example.Tokens)
		var sum float32
		for _, probability := range probabilities {
			if probability < 0 {
				t.Fatalf("negative probability %v", probabilities)
			}
			sum += probability
		}
		if sum < 0.99 || sum > 1.01 {
			t.Fatalf("probabilities sum to %.4f, want ~1", sum)
		}
	}
}

func TestDistillNgramMatchesTeacher(t *testing.T) {
	examples := syntheticExamples()
	encoder := tinyEncoder(t)
	teacher := NewTransformerClassifier(encoder, 3, 0)
	teacher.Train(examples, HeadConfig{Epochs: 60, LearningRate: 0.05, BatchSize: 16, Seed: 3})

	student := DistillNgram(teacher, examples, NgramConfig{Epochs: 15, Seed: 5})
	if got := DistillAccuracy(teacher, student, examples); got < 0.9 {
		t.Fatalf("distillation fidelity = %.3f, want >= 0.9", got)
	}
}

func TestRouterSaveLoadFastPath(t *testing.T) {
	examples := syntheticExamples()
	ngram := TrainNgram(examples, 3, NgramConfig{Epochs: 15, Seed: 11})
	original := &Router{Domains: []string{"alpha", "beta", "gamma"}, Ngram: ngram}

	path := filepath.Join(t.TempDir(), "router.gn")
	if err := original.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := LoadRouter(path, nil)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(loaded.Domains) != 3 || loaded.Domains[1] != "beta" {
		t.Fatalf("domains not restored: %v", loaded.Domains)
	}
	for _, example := range examples {
		before := original.Ngram.Predict(example.Tokens)
		after := loaded.Ngram.Predict(example.Tokens)
		if !equalProbabilities(before, after, 1e-6) {
			t.Fatalf("prediction changed across save/load: %v vs %v", before, after)
		}
	}
}

func TestRouterEscalation(t *testing.T) {
	examples := syntheticExamples()
	encoder := tinyEncoder(t)
	teacher := NewTransformerClassifier(encoder, 3, 0)
	teacher.Train(examples, HeadConfig{Epochs: 60, LearningRate: 0.05, BatchSize: 16, Seed: 3})
	ngram := TrainNgram(examples, 3, NgramConfig{Epochs: 15, Seed: 11})

	router := &Router{Domains: []string{"a", "b", "c"}, Ngram: ngram, Transformer: teacher, Threshold: 0}
	sample := examples[0].Tokens
	if !equalProbabilities(router.Predict(sample), ngram.Predict(sample), 1e-6) {
		t.Fatal("threshold 0 should use the fast path")
	}
	router.Threshold = -1
	if !equalProbabilities(router.Predict(sample), teacher.Predict(sample), 1e-6) {
		t.Fatal("negative threshold should always escalate to the transformer")
	}
}

func equalProbabilities(left, right []float32, tolerance float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if float32(math.Abs(float64(left[index]-right[index]))) > tolerance {
			return false
		}
	}
	return true
}
