package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestDrafterConfigBounded(t *testing.T) {
	backbone := testConfig()
	backbone.SequenceLen = 2048
	config := DrafterConfig(backbone)
	if config.NumLayer != DrafterLayers {
		t.Fatalf("drafter layers = %d, want %d", config.NumLayer, DrafterLayers)
	}
	if config.SequenceLen > 128 {
		t.Fatalf("drafter context = %d, want <= 128", config.SequenceLen)
	}
	if config.VocabSize != backbone.VocabSize || config.HeadDim() != backbone.HeadDim() {
		t.Fatal("drafter must keep the backbone vocab and head geometry")
	}
}

func TestDraftTokensLengthAndRange(t *testing.T) {
	backbone := NewTransformer(testConfig())
	drafter := NewDrafter(backbone)
	drafter.InitWeights(tensors.NewRNG(7))

	context := []int{1, 5, 2, 8, 3, 7, 4, 6}
	drafts := drafter.DraftTokens(context, 4)
	if len(drafts) != 4 {
		t.Fatalf("drafts = %d, want 4", len(drafts))
	}
	for _, token := range drafts {
		if token < 0 || token >= drafter.Config.VocabSize {
			t.Fatalf("draft token %d out of vocab range", token)
		}
	}
}

func TestDraftTokensEmptyContext(t *testing.T) {
	drafter := NewDrafter(NewTransformer(testConfig()))
	drafter.InitWeights(tensors.NewRNG(1))
	if drafts := drafter.DraftTokens(nil, 3); drafts != nil {
		t.Fatalf("empty context should draft nothing, got %v", drafts)
	}
}

func TestDraftTokensBoundedByContext(t *testing.T) {
	backbone := NewTransformer(testConfig())
	drafter := NewDrafter(backbone)
	drafter.InitWeights(tensors.NewRNG(2))

	long := make([]int, 500)
	for index := range long {
		long[index] = index % backbone.Config.VocabSize
	}
	// A long context must still draft (using only the bounded window) rather
	// than panic on the rotary/context limit.
	drafts := drafter.DraftTokens(long, 3)
	if len(drafts) != 3 {
		t.Fatalf("drafts = %d, want 3", len(drafts))
	}
}

func TestDSparkDraftAndConfidence(t *testing.T) {
	backbone := NewTransformer(testConfig())
	dspark := NewDSpark(backbone)
	dspark.InitWeights(tensors.NewRNG(5))

	context := []int{1, 5, 2, 8, 3, 7, 4, 6}
	tokens, confidences := dspark.Draft(context, 5)
	if len(tokens) != 5 || len(confidences) != 5 {
		t.Fatalf("draft = %d tokens / %d confidences, want 5/5", len(tokens), len(confidences))
	}
	for index, token := range tokens {
		if token < 0 || token >= dspark.Drafter.Config.VocabSize {
			t.Fatalf("draft token %d out of range", token)
		}
		if confidences[index] < 0 || confidences[index] > 1 {
			t.Fatalf("confidence %d = %v, want in [0,1]", index, confidences[index])
		}
	}
}

func TestDSparkTrainHeadsStepFinite(t *testing.T) {
	backbone := NewTransformer(testConfig())
	dspark := NewDSpark(backbone)
	dspark.InitWeights(tensors.NewRNG(9))

	inputs := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	targets := tensors.NewInt32sWithData([]int{1, 8}, []int32{2, 3, 4, 5, 6, 7, 8, 9})

	first := dspark.TrainHeadsStep(inputs, targets)
	if math.IsNaN(float64(first)) || math.IsInf(float64(first), 0) {
		t.Fatalf("head loss = %v, want finite", first)
	}
	if dspark.markovUp.Weight.Grad == nil || dspark.confOut.Weight.Grad == nil {
		t.Fatal("head gradients were not accumulated")
	}
	// markovUp is zero-initialized so the first step only populates its
	// gradient; markovDown receives gradient once markovUp is non-zero.
	hasGradient := false
	for _, gradient := range dspark.markovUp.Weight.Grad {
		if gradient != 0 {
			hasGradient = true
		}
	}
	if !hasGradient {
		t.Fatal("markov head received no gradient")
	}
	dspark.ZeroGrad()
	last := dspark.TrainHeadsStep(inputs, targets)
	if last >= first {
		t.Logf("head loss did not decrease on a single step: %v -> %v", first, last)
	}
}

func TestDSparkRoundTrip(t *testing.T) {
	backbone := NewTransformer(testConfig())
	dspark := NewDSpark(backbone)
	dspark.InitWeights(tensors.NewRNG(3))

	config := dspark.Drafter.Config
	loaded := LoadDSpark(config, dspark.NamedParameters())
	if loaded.Drafter.Config.VocabSize != config.VocabSize {
		t.Fatal("loaded DSpark config mismatch")
	}
	for name, parameter := range dspark.NamedParameters() {
		target, ok := loaded.NamedParameters()[name]
		if !ok {
			t.Fatalf("missing parameter %s after load", name)
		}
		for index := range parameter.Data {
			if target.Data[index] != parameter.Data[index] {
				t.Fatalf("parameter %s mismatch at %d", name, index)
			}
		}
	}
}
