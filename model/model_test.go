package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/model/layers"
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

func TestConfigForDepthRatio(t *testing.T) {
	// depth 12, aspect 64 -> dim 768, 6 heads at headDim 128.
	config := ConfigForDepthRatio(12, 32000, 64, 128, 2048, "SSSL", 3)
	if config.NumHead != 6 {
		t.Fatalf("NumHead = %d, want 6", config.NumHead)
	}
	if config.NumKVHead != 2 {
		t.Fatalf("NumKVHead = %d, want 2", config.NumKVHead)
	}
	// Ratio 1 reproduces MHA and matches ConfigForDepth exactly.
	mha := ConfigForDepth(12, 32000, 64, 128, 2048, "SSSL")
	if mha != ConfigForDepthRatio(12, 32000, 64, 128, 2048, "SSSL", 1) {
		t.Fatal("ratio 1 must equal ConfigForDepth")
	}
}

func TestConfigForDepthRatioRejectsIndivisible(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for indivisible kvHeadRatio")
		}
	}()
	// depth 12 gives 6 heads; ratio 4 does not divide 6.
	ConfigForDepthRatio(12, 32000, 64, 128, 2048, "SSSL", 4)
}

func TestConfigForDepthRatioRejectsZero(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for kvHeadRatio < 1")
		}
	}()
	ConfigForDepthRatio(12, 32000, 64, 128, 2048, "SSSL", 0)
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

func TestAttentionSplitCount(t *testing.T) {
	cases := []struct {
		keyLength, baseTasks, workers, want int
	}{
		{128, 2, 16, 1},   // below the key threshold
		{4096, 16, 16, 1}, // (batch, head) tasks already fill the cores
		{4096, 2, 16, 8},  // 16/2
		{4096, 1, 16, 16}, // capped at 16
		{4096, 1, 64, 16}, // hard cap
		{700, 1, 16, 5},   // key length limits the shard count (700/128)
	}
	for _, testCase := range cases {
		if got := attentionSplitCount(testCase.keyLength, testCase.baseTasks, testCase.workers); got != testCase.want {
			t.Errorf("attentionSplitCount(%d, %d, %d) = %d, want %d",
				testCase.keyLength, testCase.baseTasks, testCase.workers, got, testCase.want)
		}
	}
}

func TestAttentionForwardCombinedMatchesFull(t *testing.T) {
	queryLength, keyLength, headDim := 4, 40, 8
	rng := tensors.NewRNG(11)
	query := tensors.New(1, 1, 1, queryLength*headDim)
	tensors.FillNormal(query, rng, 1)
	key := tensors.New(1, 1, 1, keyLength*headDim)
	tensors.FillNormal(key, rng, 1)
	value := tensors.New(1, 1, 1, keyLength*headDim)
	tensors.FillNormal(value, rng, 1)

	full := make([]float32, queryLength*headDim)
	attentionForwardCombined(query.Data, key.Data, value.Data, full, nil, queryLength, keyLength, headDim, 0, -1, 1)

	split := make([]float32, queryLength*headDim)
	attentionForwardCombined(query.Data, key.Data, value.Data, split, nil, queryLength, keyLength, headDim, 0, -1, 5)

	for index := range full {
		if math.Abs(float64(full[index]-split[index])) > 1e-4 {
			t.Fatalf("split attention differs at %d: full=%v split=%v", index, full[index], split[index])
		}
	}
}

func TestRotaryDimension(t *testing.T) {
	full := Config{EmbedDim: 32, NumHead: 4} // headDim 8
	if got := full.RotaryDimension(); got != 8 {
		t.Fatalf("RotaryDimension default = %d, want 8", got)
	}
	partial := Config{EmbedDim: 32, NumHead: 4, RotaryDims: 4}
	if got := partial.RotaryDimension(); got != 4 {
		t.Fatalf("RotaryDimension = %d, want 4", got)
	}
}

func TestRotaryDimensionRejectsInvalid(t *testing.T) {
	cases := []Config{
		{EmbedDim: 32, NumHead: 4, RotaryDims: 10}, // > headDim 8
		{EmbedDim: 32, NumHead: 4, RotaryDims: 3},  // odd
	}
	for _, config := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("expected panic for RotaryDims=%d", config.RotaryDims)
				}
			}()
			config.RotaryDimension()
		}()
	}
}

func TestApplyRotaryPartialLeavesLeadingChannels(t *testing.T) {
	// headDim = 4, rotary = 2, so the leading 2 channels must be untouched.
	cosine := tensors.New(2, 1)
	sine := tensors.New(2, 1)
	cosine.Set2(0, 0, 1)
	cosine.Set2(1, 0, 0)
	sine.Set2(0, 0, 0)
	sine.Set2(1, 0, 1)

	input := tensors.NewWithData([]int{1, 1, 1, 4}, []float32{10, 20, 3, 4})
	output := ApplyRotary(input, cosine, sine, 1)
	// Leading channels unchanged; last two rotated by 90 degrees at position 1.
	want := []float32{10, 20, 4, -3}
	for index := range want {
		if output.Data[index] != want[index] {
			t.Fatalf("partial rotary = %v, want %v", output.Data, want)
		}
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

func TestInt8QuantizationModes(t *testing.T) {
	config := testConfig()
	config.QAT = "int8"
	transformer := NewTransformer(config)
	if transformer.lmHead.QuantMode != layers.QuantInt8 {
		t.Fatal("lm_head should be int8-quantized")
	}
	if transformer.blocks[0].attention.queryProjection.QuantMode != layers.QuantInt8 {
		t.Fatal("c_q should be int8-quantized")
	}
	if transformer.blocks[0].mlp.inputProjection.QuantMode != layers.QuantInt8 {
		t.Fatal("mlp c_fc should be int8-quantized")
	}
}

func TestQATCompressedTrainStepFinite(t *testing.T) {
	config := testConfig()
	config.QAT = "int8"
	config.CompressionRatio = 2
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	indexes, targets := tinyData()
	logits, context := transformer.TrainForward(indexes)
	for _, value := range logits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite logit %v", value)
		}
	}
	flattened := logits.Reshape(indexes.Numel(), config.VocabSize)
	_, valid := tensors.CrossEntropyPerPosition(flattened, targets.Reshape(indexes.Numel()), -1)
	gradLogits := tensors.CrossEntropyGrad(flattened, targets.Reshape(indexes.Numel()), -1, 1/float32(valid))
	transformer.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], config.VocabSize))
	for _, parameter := range transformer.Parameters() {
		for _, value := range parameter.Grad {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("non-finite gradient")
			}
		}
	}
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
