package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// smearModel builds a tiny model with a non-zero smear lambda and perturbed
// weights, so the smear recurrence is exercised.
func smearModel() *Transformer {
	config := Config{SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2, EmbedDim: 32, WindowPattern: "L"}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(42))
	model.smearLambda.Data[0] = 0.7
	perturb := tensors.NewRNG(3)
	for _, parameter := range model.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.2
		}
	}
	return model
}

// TestBatchedForwardMatchesSequentialDecode is the regression test for the
// smear recurrence: a batched forward must produce the same logits as decoding
// one token at a time, both from a fresh cache and from a partially filled one
// (the prefix-reuse / speculative-verification case).
func TestBatchedForwardMatchesSequentialDecode(t *testing.T) {
	model := smearModel()
	sequence := []int32{1, 5, 2, 8, 3, 7, 4, 6}

	// Fresh cache: batched prefill vs sequential decode.
	sequentialCache := NewKVBuffer(1, 16, model.Config.NumLayer, model.Config.NumKVHead, model.Config.HeadDim())
	var sequential []float32
	for index, token := range sequence {
		ids := tensors.NewInt32sWithData([]int{1, 1}, []int32{token})
		logits := model.Forward(ids, sequentialCache)
		if index == len(sequence)-1 {
			sequential = append([]float32(nil), logits.Data...)
		}
	}
	batchedCache := NewKVBuffer(1, 16, model.Config.NumLayer, model.Config.NumKVHead, model.Config.HeadDim())
	batched := model.Forward(tensors.NewInt32sWithData([]int{1, len(sequence)}, sequence), batchedCache)
	batchedLast := batched.Data[(len(sequence)-1)*model.Config.VocabSize:]
	assertLogitsClose(t, sequential, batchedLast)

	// Partially filled cache: reuse a prefix, then forward the suffix in one go
	// versus decoding the suffix sequentially.
	split := 4
	prefixCache := NewKVBuffer(1, 16, model.Config.NumLayer, model.Config.NumKVHead, model.Config.HeadDim())
	model.Forward(tensors.NewInt32sWithData([]int{1, split}, sequence[:split]), prefixCache)

	suffixSequentialCache := prefixCache.Clone()
	var suffixSequential []float32
	for index := split; index < len(sequence); index++ {
		ids := tensors.NewInt32sWithData([]int{1, 1}, []int32{sequence[index]})
		logits := model.Forward(ids, suffixSequentialCache)
		if index == len(sequence)-1 {
			suffixSequential = append([]float32(nil), logits.Data...)
		}
	}
	suffixBatchedCache := prefixCache.Clone()
	suffixBatched := model.Forward(tensors.NewInt32sWithData([]int{1, len(sequence) - split}, sequence[split:]), suffixBatchedCache)
	suffixBatchedLast := suffixBatched.Data[(len(sequence)-split-1)*model.Config.VocabSize:]
	assertLogitsClose(t, suffixSequential, suffixBatchedLast)
}

func assertLogitsClose(t *testing.T, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("logit length mismatch: %d != %d", len(want), len(got))
	}
	for index := range want {
		if math.Abs(float64(want[index]-got[index])) > 1e-4 {
			t.Fatalf("logit %d differs: %v != %v", index, want[index], got[index])
		}
	}
}

// TestBackpropDirectionalGradientCheckSmear validates the smear backward pass
// with a non-zero lambda, which the default gradient checks do not exercise
// because InitWeights leaves smearLambda at zero.
func TestBackpropDirectionalGradientCheckSmear(t *testing.T) {
	runDirectionalGradientCheck(t, smearModel())
}
