package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// cedBaseConfig is a small model that satisfies the CED requirements
// (compression and a local sliding-window branch) with a four-layer split.
func cedBaseConfig() Config {
	return Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 4, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 3, CED: true,
	}
}

// tinyCEDModel is a dense CED model: layers 0-1 are the encoder and layers 2-3
// the decoder, whose global keys/values are projected from the encoder's final
// hidden state.
func tinyCEDModel() *Transformer {
	model := NewTransformer(cedBaseConfig())
	model.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
		}
	}
	return model
}

// tinyCEDReuseModel exercises CED with a per-half reuse pattern: within each
// half layer 0 is full and layer 1 reindexes, so each decoder group's producer
// is a decoder layer.
func tinyCEDReuseModel() *Transformer {
	config := cedBaseConfig()
	config.ReusePattern = "FR"
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
		}
	}
	return model
}

func TestCEDConfigValidation(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"without compression", func(config *Config) { config.CompressionRatio = 0 }},
		{"without window", func(config *Config) { config.SWAWindow = 0 }},
		{"single layer", func(config *Config) { config.NumLayer = 1 }},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			config := cedBaseConfig()
			testCase.mutate(&config)
			NewTransformer(config)
		})
	}
}

func TestCEDBuilds(t *testing.T) {
	config := cedBaseConfig()
	transformer := NewTransformer(config)
	split := config.CEDSplit()
	if split != 2 {
		t.Fatalf("CEDSplit = %d, want 2", split)
	}
	for layerIndex := 0; layerIndex < config.NumLayer; layerIndex++ {
		attention := transformer.blocks[layerIndex].attention
		want := layerIndex >= split
		if attention.ced != want {
			t.Fatalf("layer %d ced = %v, want %v", layerIndex, attention.ced, want)
		}
	}
}

// TestCEDGlobalKvComesFromEncoder verifies that a decoder layer's compressor
// reads the encoder's final hidden state rather than its own input.
func TestCEDGlobalKvComesFromEncoder(t *testing.T) {
	transformer := tinyCEDModel()
	indexes, _ := tinyData()
	_, context := transformer.TrainForward(indexes)

	split := transformer.Config.CEDSplit()
	encoderHidden := context.previousActivations[split]
	for layerIndex := split; layerIndex < transformer.Config.NumLayer; layerIndex++ {
		compression := context.blockContexts[layerIndex].attentionContext.compression
		if compression == nil {
			t.Fatalf("decoder layer %d has no compressed context", layerIndex)
		}
		if compression.input != encoderHidden {
			t.Fatalf("decoder layer %d compressor input is not the encoder hidden state", layerIndex)
		}
	}
	// Encoder layers keep using their own hidden state.
	if context.blockContexts[0].attentionContext.compression.input == encoderHidden {
		t.Fatal("encoder layer should not use the encoder hidden state as compressor input")
	}
}

// TestBackpropDirectionalGradientCheckCED verifies the analytic backward
// through the CED split, including the routing of the decoder's global KV
// gradients to the encoder hidden state.
func TestBackpropDirectionalGradientCheckCED(t *testing.T) {
	runDirectionalGradientCheck(t, tinyCEDModel())
}

// TestBackpropDirectionalGradientCheckCEDReuse additionally covers the
// per-half cross-layer reuse path under CED.
func TestBackpropDirectionalGradientCheckCEDReuse(t *testing.T) {
	runDirectionalGradientCheck(t, tinyCEDReuseModel())
}

func TestCEDTrainStepFinite(t *testing.T) {
	for _, sparseTopK := range []int{0, 2} {
		config := cedBaseConfig()
		config.SparseTopK = sparseTopK
		config.IndexerDim = 4
		transformer := NewTransformer(config)
		transformer.InitWeights(tensors.NewRNG(42))
		indexes, targets := tinyData()

		transformer.ZeroGrad()
		logits, context := transformer.TrainForward(indexes)
		for _, value := range logits.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("sparseTopK=%d: non-finite logit %v", sparseTopK, value)
			}
		}
		flattened := logits.Reshape(indexes.Numel(), config.VocabSize)
		_, valid := tensors.CrossEntropyPerPosition(flattened, targets.Reshape(indexes.Numel()), -1)
		gradLogits := tensors.CrossEntropyGrad(flattened, targets.Reshape(indexes.Numel()), -1, 1/float32(valid))
		transformer.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], config.VocabSize))
		for _, parameter := range transformer.Parameters() {
			for _, value := range parameter.Grad {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("sparseTopK=%d: non-finite gradient", sparseTopK)
				}
			}
		}
	}
}

// TestCEDReuseProducerStaysInHalf verifies that the per-half restart keeps a
// decoder group's producer inside the decoder.
func TestCEDReuseProducerStaysInHalf(t *testing.T) {
	config := cedBaseConfig()
	config.ReusePattern = "FR"
	transformer := NewTransformer(config)

	modes := []ReuseMode{ReuseFull, ReuseReindex, ReuseFull, ReuseReindex}
	for layerIndex, want := range modes {
		if mode := transformer.blocks[layerIndex].attention.reuseMode; mode != want {
			t.Fatalf("layer %d mode = %d, want %d", layerIndex, mode, want)
		}
	}
	if producer := transformer.blocks[1].attention.producer; producer != 0 {
		t.Fatalf("encoder reindex producer = %d, want 0", producer)
	}
	if producer := transformer.blocks[3].attention.producer; producer != 2 {
		t.Fatalf("decoder reindex producer = %d, want 2", producer)
	}
}

// enableCEDCache allocates the compression state (and indexer keys) for a test
// KV cache, mirroring the inference engine's allocation masks.
func enableCEDCache(cache *KVBuffer, config Config, maximumSequenceLength int) {
	ratio := config.Compression()
	kvWidth := config.NumKVHead * config.HeadDim()
	cache.EnableCompression(ratio, config.EmbedDim, kvWidth, maximumSequenceLength/ratio+1)
	if config.SparseTopK > 0 {
		dim, heads := config.indexerDefaults()
		cache.EnableIndexerKeys(dim * heads)
	}
}

// TestCEDPrefillMatchesFullWhenWindowCoversPrompt verifies the encoder-only
// prefill and bounded replay against the ordinary full forward when the replay
// window covers the whole prompt (so no truncation occurs).
func TestCEDPrefillMatchesFullWhenWindowCoversPrompt(t *testing.T) {
	for _, sparseTopK := range []int{0, 2} {
		config := cedBaseConfig()
		config.SWAWindow = 16
		config.SparseTopK = sparseTopK
		config.IndexerDim = 4
		transformer := NewTransformer(config)
		transformer.InitWeights(tensors.NewRNG(42))
		perturb := tensors.NewRNG(123)
		for _, parameter := range transformer.Parameters() {
			for elementIndex := range parameter.Data {
				parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
			}
		}
		prompt := []int32{1, 5, 2, 8, 3, 7}
		indexes := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)

		fullCache := NewKVBuffer(1, len(prompt), config.NumLayer, config.NumKVHead, config.HeadDim())
		enableCEDCache(fullCache, config, len(prompt))
		fullLogits := transformer.Forward(indexes, fullCache)

		cedCache := NewKVBuffer(1, len(prompt), config.NumLayer, config.NumKVHead, config.HeadDim())
		enableCEDCache(cedCache, config, len(prompt))
		cedLogits := transformer.PrefillCED(indexes, cedCache)

		if cedLogits.Shape[1] != len(prompt) {
			t.Fatalf("sparseTopK=%d: replay logits rows = %d, want %d", sparseTopK, cedLogits.Shape[1], len(prompt))
		}
		rows := fullLogits.Shape[1]
		fullFlat := fullLogits.Reshape(rows, config.VocabSize)
		cedFlat := cedLogits.Reshape(cedLogits.Shape[1], config.VocabSize)
		fullLast := fullFlat.Data[(rows-1)*config.VocabSize : rows*config.VocabSize]
		cedLast := cedFlat.Data[(cedLogits.Shape[1]-1)*config.VocabSize:]
		for index := range fullLast {
			scale := float32(math.Abs(float64(fullLast[index]))) + 1e-3
			if diff := float32(math.Abs(float64(fullLast[index] - cedLast[index]))); diff > 1e-2*scale {
				t.Fatalf("sparseTopK=%d: CED prefill differs from full forward at %d: full=%v ced=%v", sparseTopK, index, fullLast[index], cedLast[index])
			}
		}
	}
}

// TestCEDPrefillBoundedReplayFinite runs the bounded replay with a window
// smaller than the prompt and checks the cache is left ready for decoding.
func TestCEDPrefillBoundedReplayFinite(t *testing.T) {
	config := cedBaseConfig()
	config.SequenceLen = 16
	prompt := []int32{1, 5, 2, 8, 3, 7, 4, 6, 2, 9, 1, 3}
	indexes := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)

	transformer := tinyCEDModel()
	cache := NewKVBuffer(1, len(prompt)+2, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableCEDCache(cache, config, len(prompt)+2)
	logits := transformer.PrefillCED(indexes, cache)
	for _, value := range logits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite prefill logit %v", value)
		}
	}
	if cache.Position() != len(prompt) {
		t.Fatalf("cache position = %d, want %d", cache.Position(), len(prompt))
	}

	// A one-token decode step after the prefill must produce finite logits.
	next := tensors.NewInt32sWithData([]int{1, 1}, []int32{4})
	decodeLogits := transformer.Forward(next, cache)
	for _, value := range decodeLogits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite decode logit %v", value)
		}
	}
}

// TestCEDPrefillFlops checks that CED reduces prefill compute (the decoder is
// only replayed) while leaving decode compute unchanged.
func TestCEDPrefillFlops(t *testing.T) {
	withCED := NewTransformer(cedBaseConfig())
	withoutCEDConfig := cedBaseConfig()
	withoutCEDConfig.CED = false
	withoutCED := NewTransformer(withoutCEDConfig)

	if withCED.EstimatePrefillFlops(64) >= withoutCED.EstimatePrefillFlops(64) {
		t.Fatalf("CED prefill FLOPs %v must be less than full %v", withCED.EstimatePrefillFlops(64), withoutCED.EstimatePrefillFlops(64))
	}
	if withCED.EstimateDecodeFlops(64) != withoutCED.EstimateDecodeFlops(64) {
		t.Fatal("CED must not change decode FLOPs")
	}
}

// TestCEDEncoderReceivesGlobalGradient verifies that the decoder's global KV
// projections route gradient into the encoder's final layer.
func TestCEDEncoderReceivesGlobalGradient(t *testing.T) {
	transformer := tinyCEDModel()
	indexes, targets := tinyData()
	analyticGrads(transformer, indexes, targets)

	split := transformer.Config.CEDSplit()
	gradient := transformer.blocks[split-1].mlp.outputProjection.Weight.Grad
	nonZero := false
	for _, value := range gradient {
		if value != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("last encoder layer received no gradient from the decoder global KV")
	}
}
