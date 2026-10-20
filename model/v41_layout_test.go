package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func layeredV41Config() Config {
	config := ConfigForDepth(6, 16, 16, 32, 256, "SSSL")
	config.ApplyPreset(PresetFlash)
	config.Validate()
	return config
}

func layeredCompressionMask(config Config) []bool {
	mask := make([]bool, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		mask[layer] = config.OwnsCompressed(layer)
	}
	return mask
}

func layeredIndexerMask(config Config) []bool {
	mask := make([]bool, config.NumLayer)
	for layer := 0; layer < config.NumLayer; layer++ {
		mask[layer] = config.OwnsIndexer(layer)
	}
	return mask
}

func TestV41LayerLayoutBuilds(t *testing.T) {
	config := layeredV41Config()
	split := config.CEDSplit()
	if split != 3 || config.PrefillLayer != 3 {
		t.Fatalf("split = %d prefill = %d, want 3/3", split, config.PrefillLayer)
	}
	transformer := NewTransformer(config)
	for layer := 0; layer < config.NumLayer; layer++ {
		attention := transformer.blocks[layer].attention
		if layer < split {
			if attention.usesCompression() {
				t.Fatalf("encoder layer %d unexpectedly uses compression", layer)
			}
			if config.HasCompressionAt(layer) || config.CompressesAt(layer) {
				t.Fatalf("encoder layer %d should be SWA-only", layer)
			}
			continue
		}
		if !attention.usesCompression() {
			t.Fatalf("decoder layer %d should use compression", layer)
		}
		if got, want := attention.compressionRatio, config.LayerRatioAt(layer); got != want {
			t.Fatalf("layer %d ratio = %d, want %d", layer, got, want)
		}
	}
	// The C1 boundary is packed but does not merge rows.
	if !config.HasCompressionAt(split) || config.CompressesAt(split) {
		t.Fatalf("boundary layer should be C1")
	}
	if transformer.blocks[split].attention.compressor == nil {
		t.Fatal("C1 boundary should own a compressor")
	}
	// FRU cycles over compressed layers 3,4,5: full, reindex, reuse.
	if mode := transformer.blocks[5].attention.reuseMode; mode != ReuseReuse {
		t.Fatalf("layer 5 reuse mode = %d, want reuse", mode)
	}
	if producer := transformer.blocks[5].attention.producer; producer != split {
		t.Fatalf("layer 5 producer = %d, want %d", producer, split)
	}
	if got := transformer.windowSizes[0][0]; got != 128 {
		t.Fatalf("encoder window = %d, want 128", got)
	}
}

func TestV41LayeredTrainStepFinite(t *testing.T) {
	config := layeredV41Config()
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
				t.Fatal("non-finite gradient in layered V4.1 model")
			}
		}
	}
}

func TestV41LayeredCacheRoundTrip(t *testing.T) {
	config := layeredV41Config()
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	indexes := tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 5, 2, 8, 3, 7})
	cache := NewKVBuffer(1, 8, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
	cache.EnableCompressionLayout(config.LayerRatios(), config.EmbedDim, config.NumKVHead*config.HeadDim(),
		config.CompressedMaxBlocks(8), layeredCompressionMask(config))
	dim, heads := config.indexerDefaults()
	cache.EnableIndexerKeysLayers(dim*heads, layeredIndexerMask(config))
	transformer.Forward(indexes, cache)

	for layer := 0; layer < config.NumLayer; layer++ {
		if got, want := cache.CompressionEnabledAt(layer), config.HasCompressionAt(layer); got != want {
			t.Fatalf("layer %d compression enabled = %v, want %v", layer, got, want)
		}
		if got, want := cache.CompressionRatioAt(layer), config.LayerRatioAt(layer); got != want {
			t.Fatalf("layer %d cache ratio = %d, want %d", layer, got, want)
		}
	}

	data, err := cache.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	decoded, err := UnmarshalKVBuffer(data)
	if err != nil {
		t.Fatalf("UnmarshalKVBuffer: %v", err)
	}
	for layer := 0; layer < config.NumLayer; layer++ {
		if got, want := decoded.CompressionRatioAt(layer), config.LayerRatioAt(layer); got != want {
			t.Fatalf("decoded layer %d ratio = %d, want %d", layer, got, want)
		}
	}
}

func TestLegacyUniformCompressionRemainsUniform(t *testing.T) {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 1,
		EmbedDim: 16, WindowPattern: "L", CompressionRatio: 2,
	}
	if config.LayeredCompression() {
		t.Fatal("uniform config should not report a layered layout")
	}
	for layer := 0; layer < config.NumLayer; layer++ {
		if got := config.LayerRatioAt(layer); got != 2 {
			t.Fatalf("layer %d ratio = %d, want 2", layer, got)
		}
	}
	cache := NewKVBuffer(1, 8, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableCompression(2, config.EmbedDim, config.NumKVHead*config.HeadDim(), 4)
	if !cache.CompressionEnabled() || cache.CompressionRatio() != 2 {
		t.Fatalf("legacy cache compression = %v/%d", cache.CompressionEnabled(), cache.CompressionRatio())
	}
}

func newV41Cache(config Config, capacity int) *KVBuffer {
	cache := NewKVBuffer(1, capacity, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
	cache.EnableCompressionLayout(config.LayerRatios(), config.EmbedDim, config.NumKVHead*config.HeadDim(),
		config.CompressedMaxBlocks(capacity), layeredCompressionMask(config))
	dim, heads := config.indexerDefaults()
	cache.EnableIndexerKeysLayers(dim*heads, layeredIndexerMask(config))
	return cache
}

func TestV41CEDPrefillMatchesFullForward(t *testing.T) {
	config := layeredV41Config()
	config.SequenceLen = 16
	config.SWAWindow = 16
	config.Validate()
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range transformer.Parameters() {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
		}
	}
	prompt := []int32{1, 5, 2, 8, 3, 7, 4, 6}
	indexes := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)

	fullCache := newV41Cache(config, len(prompt))
	fullLogits := transformer.Forward(indexes, fullCache)

	cedCache := newV41Cache(config, len(prompt))
	cedLogits := transformer.PrefillCED(indexes, cedCache)
	if cedLogits.Shape[1] != len(prompt) {
		t.Fatalf("replay rows = %d, want %d", cedLogits.Shape[1], len(prompt))
	}
	for index := range fullLogits.Data {
		delta := fullLogits.Data[index] - cedLogits.Data[index]
		if delta < 0 {
			delta = -delta
		}
		if delta > 1e-4 {
			t.Fatalf("logit %d differs: full=%v ced=%v", index, fullLogits.Data[index], cedLogits.Data[index])
		}
	}
}
