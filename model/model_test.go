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

// TestSampleMaskingBlocksCrossDocumentAttention verifies that a token in one
// segment cannot attend to tokens in a previous segment: perturbing a
// segment-0 token must not change the logits of segment-1 positions when
// segments are supplied, but must change them without the mask.
func TestSampleMaskingBlocksCrossDocumentAttention(t *testing.T) {
	model := perturbModel(NewTransformer(testConfig()))
	// Disable the smear recurrence so only attention can move information
	// across positions.
	model.smearLambda.Data[0] = 0

	segments := tensors.NewInt32sWithData([]int{1, 6}, []int32{0, 0, 0, 1, 1, 1})
	base := []int32{1, 5, 2, 8, 3, 7}
	changed := []int32{4, 5, 2, 8, 3, 7}
	vocab := model.Config.VocabSize

	logitsBase, _ := model.TrainForwardSegments(tensors.NewInt32sWithData([]int{1, 6}, base), segments)
	logitsChanged, _ := model.TrainForwardSegments(tensors.NewInt32sWithData([]int{1, 6}, changed), segments)
	for position := 3; position < 6; position++ {
		for vocabIndex := 0; vocabIndex < vocab; vocabIndex++ {
			index := position*vocab + vocabIndex
			if math.Abs(float64(logitsBase.Data[index]-logitsChanged.Data[index])) > 1e-5 {
				t.Fatalf("masked logit at position %d/%d changed across segments: %v vs %v", position, vocabIndex, logitsBase.Data[index], logitsChanged.Data[index])
			}
		}
	}

	// Without the mask the same perturbation must reach later positions.
	unmaskedBase, _ := model.TrainForward(tensors.NewInt32sWithData([]int{1, 6}, base))
	unmaskedChanged, _ := model.TrainForward(tensors.NewInt32sWithData([]int{1, 6}, changed))
	propagated := false
	var maxDiff float64
	for position := 3; position < 6 && !propagated; position++ {
		for vocabIndex := 0; vocabIndex < vocab; vocabIndex++ {
			index := position*vocab + vocabIndex
			diff := math.Abs(float64(unmaskedBase.Data[index] - unmaskedChanged.Data[index]))
			if diff > maxDiff {
				maxDiff = diff
			}
			if diff > 1e-5 {
				propagated = true
				break
			}
		}
	}
	if !propagated {
		t.Fatalf("without a segment mask the perturbation should propagate to later positions (max diff %v)", maxDiff)
	}
}

// TestCompressedSampleMaskingBlocksCrossDocumentAttention is the compressed /
// sparse / SWA counterpart of the dense sample-masking test: perturbing a
// segment-0 token must not change segment-1 logits.
func TestCompressedSampleMaskingBlocksCrossDocumentAttention(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"compression", func(c *Config) { c.CompressionRatio = 2 }},
		{"sparse", func(c *Config) { c.CompressionRatio = 2; c.SparseTopK = 2; c.IndexerPool = 2 }},
		{"swa", func(c *Config) { c.CompressionRatio = 2; c.SWAWindow = 3 }},
		{"sparse+swa", func(c *Config) { c.CompressionRatio = 2; c.SparseTopK = 2; c.IndexerPool = 2; c.SWAWindow = 3 }},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := testConfig()
			testCase.mutate(&config)
			config.Validate()
			model := perturbModel(NewTransformer(config))
			model.smearLambda.Data[0] = 0

			segments := tensors.NewInt32sWithData([]int{1, 6}, []int32{0, 0, 0, 1, 1, 1})
			base := []int32{1, 5, 2, 8, 3, 7}
			changed := []int32{4, 5, 2, 8, 3, 7}
			vocab := config.VocabSize
			logitsBase, _ := model.TrainForwardSegments(tensors.NewInt32sWithData([]int{1, 6}, base), segments)
			logitsChanged, _ := model.TrainForwardSegments(tensors.NewInt32sWithData([]int{1, 6}, changed), segments)
			for position := 3; position < 6; position++ {
				for vocabIndex := 0; vocabIndex < vocab; vocabIndex++ {
					index := position*vocab + vocabIndex
					if math.Abs(float64(logitsBase.Data[index]-logitsChanged.Data[index])) > 1e-4 {
						t.Fatalf("masked logit at position %d/%d changed across segments: %v vs %v", position, vocabIndex, logitsBase.Data[index], logitsChanged.Data[index])
					}
				}
			}
		})
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

func TestSparseSelectAllMatchesDense(t *testing.T) {
	config := testConfig()
	config.CompressionRatio = 2
	config.SparseTopK = 1000 // far more than the block count: selects every allowed block
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	indexes, _ := tinyData()

	sparseLogits, _ := transformer.TrainForward(indexes)
	for _, block := range transformer.blocks {
		block.attention.indexer = nil
		block.attention.sparseTopK = 0
	}
	denseLogits, _ := transformer.TrainForward(indexes)

	for index := range sparseLogits.Data {
		if math.Abs(float64(sparseLogits.Data[index]-denseLogits.Data[index])) > 1e-4 {
			t.Fatalf("sparse-with-all-blocks differs from dense at %d: %v vs %v", index, sparseLogits.Data[index], denseLogits.Data[index])
		}
	}
}

func TestSparseCompressedTrainStepFinite(t *testing.T) {
	config := testConfig()
	config.CompressionRatio = 2
	config.SparseTopK = 1
	config.IndexerDim = 4
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

func TestSparseIndexerReceivesDistillationGradient(t *testing.T) {
	config := testConfig()
	config.CompressionRatio = 2
	config.SparseTopK = 1
	config.IndexerDim = 4
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	indexes, targets := tinyData()

	transformer.ZeroGrad()
	logits, context := transformer.TrainForward(indexes)
	flattened := logits.Reshape(indexes.Numel(), config.VocabSize)
	_, valid := tensors.CrossEntropyPerPosition(flattened, targets.Reshape(indexes.Numel()), -1)
	gradLogits := tensors.CrossEntropyGrad(flattened, targets.Reshape(indexes.Numel()), -1, 1/float32(valid))
	transformer.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], config.VocabSize))

	indexer := transformer.blocks[0].attention.indexer
	nonZero := false
	for _, parameter := range indexer.Parameters() {
		for _, value := range parameter.Grad {
			if value != 0 {
				nonZero = true
			}
		}
	}
	if !nonZero {
		t.Fatal("indexer parameters received no distillation gradient")
	}
}

func TestReusePatternValidation(t *testing.T) {
	cases := []struct {
		name    string
		pattern string
		ratio   int
	}{
		{"reuse without compression", "FRUU", 0},
		{"must start with full", "RU", 2},
		{"invalid character", "FXU", 2},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			config := testConfig()
			config.CompressionRatio = testCase.ratio
			config.ReusePattern = testCase.pattern
			NewTransformer(config)
		})
	}
}

func TestReusePatternBuilds(t *testing.T) {
	config := testConfig()
	config.NumLayer = 4
	config.CompressionRatio = 2
	config.SparseTopK = 2
	config.IndexerDim = 4
	config.ReusePattern = "FRUU"
	transformer := NewTransformer(config)

	modes := []ReuseMode{ReuseFull, ReuseReindex, ReuseReuse, ReuseReuse}
	for layerIndex, want := range modes {
		attention := transformer.blocks[layerIndex].attention
		if attention.reuseMode != want {
			t.Fatalf("layer %d mode = %d, want %d", layerIndex, attention.reuseMode, want)
		}
		if attention.producer != 0 {
			t.Fatalf("layer %d producer = %d, want 0", layerIndex, attention.producer)
		}
	}
}

func TestReuseProducerFollowsLaterFullLayers(t *testing.T) {
	// Pattern F F R U: layers 0 and 1 are full; layer 1 produces for layers 2-3.
	config := testConfig()
	config.NumLayer = 4
	config.CompressionRatio = 2
	config.SparseTopK = 2
	config.IndexerDim = 4
	config.ReusePattern = "FFRU"
	transformer := NewTransformer(config)

	if producer := transformer.blocks[2].attention.producer; producer != 1 {
		t.Fatalf("layer 2 producer = %d, want 1", producer)
	}
	if producer := transformer.blocks[3].attention.producer; producer != 1 {
		t.Fatalf("layer 3 producer = %d, want 1", producer)
	}
}

func TestReuseCompressedTrainStepFinite(t *testing.T) {
	config := testConfig()
	config.NumLayer = 4
	config.CompressionRatio = 2
	config.SparseTopK = 2
	config.IndexerDim = 4
	config.ReusePattern = "FRUU"
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

// TestReuseMultiGroupTrainStepFinite exercises patterns with more than one full
// producer group, ensuring each group's share state is independent.
func TestReuseMultiGroupTrainStepFinite(t *testing.T) {
	for _, pattern := range []string{"FFRU", "FRFR", "FRUR"} {
		t.Run(pattern, func(t *testing.T) {
			config := testConfig()
			config.NumLayer = 4
			config.CompressionRatio = 2
			config.SparseTopK = 2
			config.IndexerDim = 4
			config.ReusePattern = pattern
			transformer := NewTransformer(config)
			transformer.InitWeights(tensors.NewRNG(42))
			indexes, targets := tinyData()

			transformer.ZeroGrad()
			logits, context := transformer.TrainForward(indexes)
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
		})
	}
}

// TestHierarchicalPoolTrainStepFinite exercises the shared candidate pool: a
// Full layer publishes the pool and a Reindex layer scores only within it,
// during both the forward and backward pass.
func TestHierarchicalPoolTrainStepFinite(t *testing.T) {
	for _, pattern := range []string{"FRUU", "FR"} {
		t.Run(pattern, func(t *testing.T) {
			config := testConfig()
			config.NumLayer = 4
			config.CompressionRatio = 2
			config.SparseTopK = 2
			config.IndexerDim = 4
			config.IndexerPool = 2
			config.IndexerCandidates = 4
			config.ReusePattern = pattern
			transformer := NewTransformer(config)
			transformer.InitWeights(tensors.NewRNG(42))
			indexes, targets := tinyData()

			transformer.ZeroGrad()
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
		})
	}
}

// TestHierarchicalPoolInferenceFinite runs prefill and decode through the
// inference pool path: the Full layer publishes the pool and the Reindex layer
// consumes it.
func TestHierarchicalPoolInferenceFinite(t *testing.T) {
	config := testConfig()
	config.NumLayer = 4
	config.CompressionRatio = 2
	config.SparseTopK = 2
	config.IndexerDim = 4
	config.IndexerPool = 2
	config.IndexerCandidates = 4
	config.ReusePattern = "FRUU"
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))

	maximum := 16
	cache := NewKVBuffer(1, maximum, config.NumLayer, config.NumKVHead, config.HeadDim())
	compressionMask := make([]bool, config.NumLayer)
	indexerMask := make([]bool, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		compressionMask[layer] = config.OwnsCompressed(layer)
		indexerMask[layer] = config.OwnsIndexer(layer)
	}
	cache.EnableCompressionLayers(config.Compression(), config.EmbedDim, config.NumKVHead*config.HeadDim(), maximum/config.Compression()+1, compressionMask)
	cache.EnableIndexerKeysLayers(config.IndexerDim, indexerMask)

	prefill := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	logits := transformer.Forward(prefill, cache)
	for _, value := range logits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite prefill logit %v", value)
		}
	}
	for step := 0; step < 4; step++ {
		next := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(step + 1)})
		logits = transformer.Forward(next, cache)
		for _, value := range logits.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("non-finite decode logit %v", value)
			}
		}
	}
}

// TestReuseLayersReceiveQueryGradient verifies reuse layers still train their
// own query projection even though they borrow the producing layer's KV. The
// fixture perturbs the zero-initialized projections so the gradient signal
// reaches the attention input.
func TestReuseLayersReceiveQueryGradient(t *testing.T) {
	transformer := tinyReuseModel()
	indexes, targets := tinyData()

	analyticGrads(transformer, indexes, targets)

	for _, layerIndex := range []int{1, 2, 3} {
		gradient := transformer.blocks[layerIndex].attention.queryProjection.Weight.Grad
		nonZero := false
		for _, value := range gradient {
			if value != 0 {
				nonZero = true
				break
			}
		}
		if !nonZero {
			t.Fatalf("layer %d query projection received no gradient", layerIndex)
		}
	}
}

func TestReuseReducesParameters(t *testing.T) {
	base := testConfig()
	base.NumLayer = 4
	base.CompressionRatio = 2
	base.SparseTopK = 2
	base.IndexerDim = 4
	full := NewTransformer(base)

	reuseConfig := base
	reuseConfig.ReusePattern = "FRUU"
	reuse := NewTransformer(reuseConfig)

	if reuse.TotalParams() >= full.TotalParams() {
		t.Fatalf("reuse params %d must be fewer than full params %d", reuse.TotalParams(), full.TotalParams())
	}
	if reuse.blocks[0].attention.compressor == nil || reuse.blocks[0].attention.indexer == nil {
		t.Fatal("full layer should own a compressor and indexer")
	}
	if reuse.blocks[1].attention.compressor != nil || reuse.blocks[1].attention.indexer == nil {
		t.Fatal("reindex layer should own an indexer but no compressor")
	}
	for _, layerIndex := range []int{2, 3} {
		if reuse.blocks[layerIndex].attention.compressor != nil || reuse.blocks[layerIndex].attention.indexer != nil {
			t.Fatalf("reuse layer %d should own neither compressor nor indexer", layerIndex)
		}
	}
}

func TestReuseCacheAllocation(t *testing.T) {
	config := testConfig()
	config.NumLayer = 4
	config.CompressionRatio = 2
	config.SparseTopK = 2
	config.IndexerDim = 4
	config.ReusePattern = "FRUU"

	cache := NewKVBuffer(1, 16, config.NumLayer, config.NumKVHead, config.HeadDim())
	compressionMask := make([]bool, config.NumLayer)
	indexerMask := make([]bool, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		compressionMask[layer] = config.OwnsCompressed(layer)
		indexerMask[layer] = config.OwnsIndexer(layer)
	}
	cache.EnableCompressionLayers(2, config.EmbedDim, config.NumKVHead*config.HeadDim(), 8, compressionMask)
	cache.EnableIndexerKeysLayers(config.IndexerDim, indexerMask)

	if cache.compressedKey[0] == nil {
		t.Fatal("full layer should own compressed KV")
	}
	for _, layerIndex := range []int{1, 2, 3} {
		if cache.compressedKey[layerIndex] != nil {
			t.Fatalf("layer %d should not own compressed KV", layerIndex)
		}
	}
	if cache.indexerKey[0] == nil || cache.indexerKey[1] == nil {
		t.Fatal("full and reindex layers should cache indexer keys")
	}
	if cache.indexerKey[2] != nil || cache.indexerKey[3] != nil {
		t.Fatal("reuse layers should not cache indexer keys")
	}
}

func TestReuseReducesCompressedCacheBytes(t *testing.T) {
	config := testConfig()
	config.NumLayer = 4
	config.CompressionRatio = 2
	config.SparseTopK = 2
	config.IndexerDim = 4

	newCache := func(pattern string) *KVBuffer {
		current := config
		current.ReusePattern = pattern
		cache := NewKVBuffer(1, 16, current.NumLayer, current.NumKVHead, current.HeadDim())
		compressionMask := make([]bool, current.NumLayer)
		indexerMask := make([]bool, current.NumLayer)
		for layer := 0; layer < current.NumLayer; layer++ {
			compressionMask[layer] = current.OwnsCompressed(layer)
			indexerMask[layer] = current.OwnsIndexer(layer)
		}
		cache.EnableCompressionLayers(2, current.EmbedDim, current.NumKVHead*current.HeadDim(), 8, compressionMask)
		cache.EnableIndexerKeysLayers(current.IndexerDim, indexerMask)
		return cache
	}

	full := newCache("")
	reuse := newCache("FRUU")
	if reuse.CompressedBytesAllocated() >= full.CompressedBytesAllocated() {
		t.Fatalf("reuse cache %d bytes must be fewer than full %d", reuse.CompressedBytesAllocated(), full.CompressedBytesAllocated())
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

// TestForwardLowRankDecodePath runs prefill and decode with the low-rank query
// and KV latent projections combined with dense compression and sliding-window
// attention.
func TestForwardLowRankDecodePath(t *testing.T) {
	config := testConfig()
	config.CompressionRatio = 2
	config.SWAWindow = 3
	config.QueryCompressionDim = 8
	config.KVLatentDim = 8
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))

	maximum := 16
	cache := NewKVBuffer(1, maximum, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableCompression(config.Compression(), config.EmbedDim, config.NumKVHead*config.HeadDim(), maximum/config.Compression()+1)

	prefill := tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 2, 3, 4, 5, 6})
	logits := transformer.Forward(prefill, cache)
	assertFiniteTensor(t, logits)
	for step := 0; step < 4; step++ {
		next := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(step + 1)})
		logits = transformer.Forward(next, cache)
		assertFiniteTensor(t, logits)
	}
}

func assertFiniteTensor(t *testing.T, values *tensors.Tensor) {
	t.Helper()
	for _, value := range values.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite value %v", value)
		}
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
