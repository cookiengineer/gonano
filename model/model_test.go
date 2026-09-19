package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func testConfig() Config {
	return Config{
		SequenceLen:   16,
		VocabSize:     32,
		NumLayer:      2,
		NumHead:       2,
		NumKVHead:     2,
		EmbedDim:      32,
		WindowPattern: "L",
	}
}

func TestConfigForDepth(t *testing.T) {
	config := ConfigForDepth(12, 32000, 64, 128, 2048, "SSSL")
	if config.EmbedDim != 12*64 {
		t.Fatalf("EmbedDim = %d, want %d", config.EmbedDim, 12*64)
	}
	if config.NumHead != config.EmbedDim/128 {
		t.Fatalf("NumHead = %d, want %d", config.NumHead, config.EmbedDim/128)
	}
	if config.EmbedDim%config.NumHead != 0 {
		t.Fatal("EmbedDim must be divisible by NumHead")
	}
}

func TestWindowSizes(t *testing.T) {
	config := Config{
		SequenceLen:   2048,
		VocabSize:     32,
		NumLayer:      4,
		NumHead:       2,
		NumKVHead:     2,
		EmbedDim:      8,
		WindowPattern: "SSSL",
	}
	sizes := config.WindowSizes()
	// short = ceil(2048/4/128)*128 = ceil(4)*128 = 512.
	if sizes[0] != [2]int{512, 0} {
		t.Fatalf("layer 0 window = %v, want [512 0]", sizes[0])
	}
	if sizes[1] != [2]int{512, 0} {
		t.Fatalf("layer 1 window = %v", sizes[1])
	}
	if sizes[2] != [2]int{512, 0} {
		t.Fatalf("layer 2 window = %v", sizes[2])
	}
	// Last layer always full context.
	if sizes[3] != [2]int{2048, 0} {
		t.Fatalf("layer 3 window = %v, want [2048 0]", sizes[3])
	}
}

func TestPaddedVocab(t *testing.T) {
	config := testConfig() // vocab 32
	if config.PaddedVocab() != 64 {
		t.Fatalf("PaddedVocab = %d, want 64", config.PaddedVocab())
	}
	config.VocabSize = 65
	if config.PaddedVocab() != 128 {
		t.Fatalf("PaddedVocab = %d, want 128", config.PaddedVocab())
	}
}

func TestApplyRotaryQuarterTurn(t *testing.T) {
	// headDim = 2 (half = 1): position 1 rotates by 90 degrees.
	cosine := tensors.New(2, 1)
	sine := tensors.New(2, 1)
	cosine.Set2(0, 0, 1)
	cosine.Set2(1, 0, 0)
	sine.Set2(0, 0, 0)
	sine.Set2(1, 0, 1)

	input := tensors.NewWithData([]int{1, 1, 1, 2}, []float32{3, 4})
	output := ApplyRotary(input, cosine, sine, 1)
	// y1 = x1*cos + x2*sin = 3*0 + 4*1 = 4; y2 = -x1*sin + x2*cos = -3*1 + 4*0 = -3.
	if output.Data[0] != 4 || output.Data[1] != -3 {
		t.Fatalf("rotary = %v, want [4 -3]", output.Data)
	}
}

func TestApplyRotaryPreservesNorm(t *testing.T) {
	headDimension := 8
	cosine, sine := precomputeRotary(32, headDimension)
	rng := tensors.NewRNG(3)
	input := tensors.New(2, 4, 2, headDimension)
	tensors.FillNormal(input, rng, 1)

	output := ApplyRotary(input, cosine, sine, 0)
	// Sum of squares must be preserved per (batch, sequence, head) pair.
	for batchIndex := 0; batchIndex < 2; batchIndex++ {
		for sequenceIndex := 0; sequenceIndex < 4; sequenceIndex++ {
			for headIndex := 0; headIndex < 2; headIndex++ {
				var inputNorm, outputNorm float64
				base := ((batchIndex*4+sequenceIndex)*2 + headIndex) * headDimension
				for dimensionIndex := 0; dimensionIndex < headDimension; dimensionIndex++ {
					inputNorm += float64(input.Data[base+dimensionIndex]) * float64(input.Data[base+dimensionIndex])
					outputNorm += float64(output.Data[base+dimensionIndex]) * float64(output.Data[base+dimensionIndex])
				}
				if math.Abs(inputNorm-outputNorm) > 1e-3 {
					t.Fatalf("rotation changed norm: %v -> %v", inputNorm, outputNorm)
				}
			}
		}
	}
}

func buildTestTransformer(t *testing.T) *Transformer {
	t.Helper()
	model := NewTransformer(testConfig())
	model.InitWeights(tensors.NewRNG(42))
	return model
}

func TestForwardShapes(t *testing.T) {
	model := buildTestTransformer(t)
	indexes := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	logits := model.Forward(indexes, nil)
	if logits.Shape[0] != 1 || logits.Shape[1] != 4 || logits.Shape[2] != 32 {
		t.Fatalf("logits shape = %v, want [1 4 32]", logits.Shape)
	}
	for _, value := range logits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("logits contain non-finite value %v", value)
		}
	}
}

func TestForwardDeterministic(t *testing.T) {
	indexes := tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 2, 3, 4, 5, 6})
	firstModel := buildTestTransformer(t)
	secondModel := buildTestTransformer(t)
	firstLogits := firstModel.Forward(indexes, nil)
	secondLogits := secondModel.Forward(indexes, nil)
	for elementIndex := range firstLogits.Data {
		if firstLogits.Data[elementIndex] != secondLogits.Data[elementIndex] {
			t.Fatalf("nondeterministic forward at %d", elementIndex)
		}
	}
}

func TestForwardBatch(t *testing.T) {
	model := buildTestTransformer(t)
	indexes := tensors.NewInt32sWithData([]int{3, 5}, []int32{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})
	logits := model.Forward(indexes, nil)
	if logits.Shape[0] != 3 || logits.Shape[1] != 5 || logits.Shape[2] != 32 {
		t.Fatalf("logits shape = %v", logits.Shape)
	}
}

func TestForwardDecodePath(t *testing.T) {
	model := buildTestTransformer(t)
	cache := NewKVBuffer(1, 16, model.NumLayers(), model.Config.NumKVHead, model.Config.HeadDim())

	prefillIndexes := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	prefillLogits := model.Forward(prefillIndexes, cache)
	if prefillLogits.Shape[0] != 1 || prefillLogits.Shape[1] != 4 {
		t.Fatalf("prefill logits shape = %v", prefillLogits.Shape)
	}
	if cache.Position() != 4 {
		t.Fatalf("cache position = %d, want 4", cache.Position())
	}

	stepIndexes := tensors.NewInt32sWithData([]int{1, 1}, []int32{5})
	decodeLogits := model.Forward(stepIndexes, cache)
	if decodeLogits.Shape[0] != 1 || decodeLogits.Shape[1] != 1 {
		t.Fatalf("decode logits shape = %v", decodeLogits.Shape)
	}
	if cache.Position() != 5 {
		t.Fatalf("cache position = %d, want 5", cache.Position())
	}
}

func TestInitWeightsStats(t *testing.T) {
	config := Config{
		SequenceLen: 16, VocabSize: 64, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 64, WindowPattern: "L",
	}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(7))

	embeddingStd := stdDev(model.tokenEmbedding.Weight.Data)
	if math.Abs(float64(embeddingStd)-0.8) > 0.05 {
		t.Fatalf("wte std = %v, want ~0.8", embeddingStd)
	}
	lmHeadStd := stdDev(model.lmHead.Weight.Data)
	if math.Abs(float64(lmHeadStd)-0.001) > 0.0005 {
		t.Fatalf("lm_head std = %v, want ~0.001", lmHeadStd)
	}
	// resid lambdas decrease from 1.15 to 1.05 across 2 layers.
	if model.residLambdas.Data[0] != 1.15 || model.residLambdas.Data[1] != 1.05 {
		t.Fatalf("resid lambdas = %v, want [1.15 1.05]", model.residLambdas.Data)
	}
	if model.backoutLambda.Data[0] != 0.2 {
		t.Fatalf("backout = %v, want 0.2", model.backoutLambda.Data[0])
	}
	if model.smearLambda.Data[0] != 0 {
		t.Fatalf("smear lambda = %v, want 0", model.smearLambda.Data[0])
	}
}

func stdDev(values []float32) float32 {
	var mean, sumSquaredDeviations float64
	for index, value := range values {
		deviation := float64(value) - mean
		mean += deviation / float64(index+1)
		sumSquaredDeviations += deviation * (float64(value) - mean)
	}
	return float32(math.Sqrt(sumSquaredDeviations / float64(len(values))))
}
