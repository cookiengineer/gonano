package model

import (
	"fmt"
	"math"
	"testing"

	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

func mlaTestConfig() Config {
	config := testConfig()
	config.MLALatent = 8
	config.MLARotaryDims = 8
	return config
}

func assertMLAShape(t *testing.T, weight *tensors.Tensor, want ...int) {
	t.Helper()
	if len(weight.Shape) != len(want) {
		t.Fatalf("rank = %d, want %d", len(weight.Shape), len(want))
	}
	for index, dimension := range want {
		if weight.Shape[index] != dimension {
			t.Fatalf("shape %v, want %v", weight.Shape, want)
		}
	}
}

func TestMLAConfigValidation(t *testing.T) {
	transformer := NewTransformer(mlaTestConfig())
	if !transformer.Config.MLAEnabled() || transformer.Config.MLARank() != 8 {
		t.Fatal("MLA config not recognized")
	}
	if got := transformer.Config.MLAContentDim(); got != transformer.Config.HeadDim()-8 {
		t.Fatalf("content dim = %d, want %d", got, transformer.Config.HeadDim()-8)
	}

	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"compression", func(config *Config) { config.CompressionRatio = 2 }},
		{"ced", func(config *Config) { config.CED = true; config.CompressionRatio = 2; config.SWAWindow = 3 }},
		{"sparse", func(config *Config) { config.SparseTopK = 2; config.CompressionRatio = 2 }},
		{"reuse", func(config *Config) { config.ReusePattern = "F"; config.CompressionRatio = 2 }},
		{"low-rank", func(config *Config) { config.KVLatentDim = 8 }},
		{"head-wise-muon", func(config *Config) { config.HeadWiseMuon = true }},
		{"full-rotary", func(config *Config) { config.MLARotaryDims = config.HeadDim() }},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected panic")
				}
			}()
			config := mlaTestConfig()
			testCase.mutate(&config)
			NewTransformer(config)
		})
	}
}

func TestMLARotaryDimensionDefault(t *testing.T) {
	config := testConfig() // headDim 16
	config.MLALatent = 4
	if got := config.MLARotaryDimension(); got != 8 {
		t.Fatalf("default MLA rotary = %d, want 8 (headDim/2)", got)
	}
}

func TestMLAParameterShapes(t *testing.T) {
	config := mlaTestConfig()
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))

	headDimension := config.HeadDim()
	ropeDim := config.MLARotaryDimension()
	contentDim := headDimension - ropeDim
	latent := config.MLALatent

	for layer, block := range transformer.blocks {
		mla := block.attention.mla
		if mla == nil {
			t.Fatalf("layer %d has no MLA parameters", layer)
		}
		if block.attention.queryProjection != nil || block.attention.keyProjection != nil || block.attention.valueProjection != nil {
			t.Fatalf("layer %d allocated standard projections alongside MLA", layer)
		}
		assertMLAShape(t, mla.QueryDown.Weight, latent, config.EmbedDim)
		assertMLAShape(t, mla.QueryUp.Weight, config.NumHead*contentDim, latent)
		assertMLAShape(t, mla.QueryRope.Weight, config.NumHead*ropeDim, config.EmbedDim)
		assertMLAShape(t, mla.KVDown.Weight, latent, config.EmbedDim)
		assertMLAShape(t, mla.KeyUp.Weight, config.NumKVHead*contentDim, latent)
		assertMLAShape(t, mla.ValueUp.Weight, config.NumKVHead*headDimension, latent)
		assertMLAShape(t, mla.KeyRope.Weight, config.NumKVHead*ropeDim, config.EmbedDim)
	}

	names := transformer.NamedParameters()
	for layer := 0; layer < config.NumLayer; layer++ {
		for _, suffix := range []string{
			"mla_q_down.weight", "mla_q_up.weight", "mla_q_rope.weight",
			"mla_kv_down.weight", "mla_k_up.weight", "mla_v_up.weight", "mla_k_rope.weight",
		} {
			name := fmt.Sprintf("transformer.h.%d.attn.%s", layer, suffix)
			if _, ok := names[name]; !ok {
				t.Fatalf("missing named parameter %s", name)
			}
		}
		if _, ok := names[fmt.Sprintf("transformer.h.%d.attn.c_q.weight", layer)]; ok {
			t.Fatalf("layer %d unexpectedly has a standard query weight", layer)
		}
	}

	// MatmulParams counts the MLA weights plus the output, MLP, ve-gate, lm_head,
	// and smear gate.
	expected := 0
	for _, block := range transformer.blocks {
		expected += block.attention.mla.numParameters()
		expected += block.attention.outputProjection.Weight.Numel()
		if block.attention.valueEmbeddingGate != nil {
			expected += block.attention.valueEmbeddingGate.Weight.Numel()
		}
		expected += block.mlp.inputProjection.Weight.Numel()
		expected += block.mlp.outputProjection.Weight.Numel()
	}
	expected += transformer.lmHead.Weight.Numel()
	expected += transformer.smearGate.Weight.Numel()
	if got := transformer.MatmulParams(); got != expected {
		t.Fatalf("MatmulParams = %d, want %d", got, expected)
	}
	if transformer.NumScalingParams().Total != transformer.TotalParams() {
		t.Fatalf("scaling total %d != TotalParams %d", transformer.NumScalingParams().Total, transformer.TotalParams())
	}

	for _, parameter := range transformer.Parameters() {
		for _, value := range parameter.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("non-finite MLA parameter after init")
			}
		}
	}
}

func mlaTinyModel() *Transformer {
	transformer := NewTransformer(mlaTestConfig())
	transformer.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range transformer.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.1
		}
	}
	return transformer
}

func TestMLATrainStepFinite(t *testing.T) {
	transformer := mlaTinyModel()
	indexes, targets := tinyData()
	transformer.ZeroGrad()
	logits, context := transformer.TrainForward(indexes)
	for _, value := range logits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite MLA logit %v", value)
		}
	}
	flattened := logits.Reshape(indexes.Numel(), transformer.Config.VocabSize)
	_, valid := tensors.CrossEntropyPerPosition(flattened, targets.Reshape(indexes.Numel()), -1)
	gradLogits := tensors.CrossEntropyGrad(flattened, targets.Reshape(indexes.Numel()), -1, 1/float32(valid))
	transformer.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], transformer.Config.VocabSize))

	nonZero := false
	for _, parameter := range transformer.Parameters() {
		for _, value := range parameter.Grad {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("non-finite MLA gradient")
			}
			if value != 0 {
				nonZero = true
			}
		}
	}
	if !nonZero {
		t.Fatal("MLA backward produced no gradients")
	}
}

func TestBackpropDirectionalGradientCheckMLA(t *testing.T) {
	runDirectionalGradientCheck(t, mlaTinyModel())
}

func TestBackpropPerElementMLA(t *testing.T) {
	transformer := mlaTinyModel()
	indexes, targets := tinyData()
	analyticGrads(transformer, indexes, targets)

	checks := []struct {
		name         string
		parameter    *tensors.Tensor
		elementIndex int
	}{
		{"mla_q_down", transformer.blocks[0].attention.mla.QueryDown.Weight, 3},
		{"mla_q_up", transformer.blocks[0].attention.mla.QueryUp.Weight, 5},
		{"mla_q_rope", transformer.blocks[0].attention.mla.QueryRope.Weight, 7},
		{"mla_kv_down", transformer.blocks[0].attention.mla.KVDown.Weight, 1},
		{"mla_k_up", transformer.blocks[0].attention.mla.KeyUp.Weight, 9},
		{"mla_v_up", transformer.blocks[0].attention.mla.ValueUp.Weight, 2},
		{"mla_k_rope", transformer.blocks[0].attention.mla.KeyRope.Weight, 4},
	}

	epsilon := float32(1e-3)
	for _, check := range checks {
		analytic := check.parameter.Grad[check.elementIndex]
		original := check.parameter.Data[check.elementIndex]
		check.parameter.Data[check.elementIndex] = original + epsilon
		lossPlus := meanLoss(transformer, indexes, targets)
		check.parameter.Data[check.elementIndex] = original - epsilon
		lossMinus := meanLoss(transformer, indexes, targets)
		check.parameter.Data[check.elementIndex] = original
		numeric := (lossPlus - lossMinus) / (2 * epsilon)

		scale := math.Abs(float64(analytic))
		if math.Abs(float64(numeric)) > scale {
			scale = math.Abs(float64(numeric))
		}
		scale = math.Max(scale, 1e-3)
		if math.Abs(float64(numeric-analytic)) > 0.5*scale {
			t.Errorf("%s[%d]: analytic=%v numeric=%v", check.name, check.elementIndex, analytic, numeric)
		}
	}
}

func TestMLATrainOverfitsTiny(t *testing.T) {
	transformer := mlaTinyModel()
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	optimizerImpl := optimizer.NewMuonAdamW(groups)

	indexes, targets := tinyData()
	vocabulary := transformer.Config.VocabSize
	step := func() float32 {
		transformer.ZeroGrad()
		logits, context := transformer.TrainForward(indexes)
		flat := logits.Reshape(indexes.Numel(), vocabulary)
		targetFlat := targets.Reshape(indexes.Numel())
		loss := tensors.CrossEntropy(flat, targetFlat, -1)
		_, valid := tensors.CrossEntropyPerPosition(flat, targetFlat, -1)
		gradient := tensors.CrossEntropyGrad(flat, targetFlat, -1, 1/float32(valid))
		transformer.TrainBackward(context, gradient.Reshape(indexes.Shape[0], indexes.Shape[1], vocabulary))
		return loss
	}

	first := step()
	optimizerImpl.Step()
	optimizerImpl.ZeroGrad()
	var last float32
	for stepIndex := 0; stepIndex < 60; stepIndex++ {
		last = step()
		optimizerImpl.Step()
		optimizerImpl.ZeroGrad()
	}
	if math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
		t.Fatalf("MLA loss became non-finite: %v", last)
	}
	if last >= first {
		t.Fatalf("MLA loss did not decrease: %v -> %v", first, last)
	}
}

func newMLACache(config Config, capacity int) *KVBuffer {
	cache := NewKVBuffer(1, capacity, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
	return cache
}

// TestMLAInferenceMatchesTraining verifies that the absorbed inference path
// computes the same function as the non-absorbed training construction.
func TestMLAInferenceMatchesTraining(t *testing.T) {
	transformer := mlaTinyModel()
	config := transformer.Config
	ids := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})

	trainLogits, _ := transformer.TrainForward(ids)
	cache := newMLACache(config, 16)
	forwardLogits := transformer.Forward(ids, cache)

	for index := range trainLogits.Data {
		if math.Abs(float64(trainLogits.Data[index]-forwardLogits.Data[index])) > 1e-3 {
			t.Fatalf("logit %d: training=%v inference=%v", index, trainLogits.Data[index], forwardLogits.Data[index])
		}
	}
}

// TestMLADecodeMatchesPrefill verifies the latent KV cache: decoding a token
// incrementally matches prefilling the whole sequence.
func TestMLADecodeMatchesPrefill(t *testing.T) {
	transformer := mlaTinyModel()
	config := transformer.Config
	tokens := []int32{1, 5, 2, 8, 3, 7, 4, 6}
	vocabulary := config.VocabSize

	prefillCache := newMLACache(config, 16)
	prefilled := transformer.Forward(tensors.NewInt32sWithData([]int{1, len(tokens)}, tokens), prefillCache)
	prefillLast := prefilled.Data[(len(tokens)-1)*vocabulary:]

	decodeCache := newMLACache(config, 16)
	transformer.Forward(tensors.NewInt32sWithData([]int{1, len(tokens) - 1}, tokens[:len(tokens)-1]), decodeCache)
	decoded := transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, tokens[len(tokens)-1:]), decodeCache)
	decodeLast := decoded.Data

	for index := range prefillLast {
		if math.Abs(float64(prefillLast[index]-decodeLast[index])) > 1e-3 {
			t.Fatalf("logit %d: prefill=%v decode=%v", index, prefillLast[index], decodeLast[index])
		}
	}
}

func TestMLACacheCodecRoundTrip(t *testing.T) {
	config := mlaTestConfig()
	transformer := mlaTinyModel()
	ids := tensors.NewInt32sWithData([]int{1, 5}, []int32{1, 2, 3, 4, 5})
	cache := newMLACache(config, 16)
	transformer.Forward(ids, cache)

	data, err := cache.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := UnmarshalKVBuffer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !decoded.MLAEnabled() {
		t.Fatal("decoded cache lost MLA buffers")
	}
	if decoded.MLALatentWidth() != config.MLALatent || decoded.MLARopeKeyWidth() != config.NumKVHead*config.MLARotaryDimension() {
		t.Fatal("MLA widths mismatch after round-trip")
	}
	position := cache.Position()
	for layer := 0; layer < config.NumLayer; layer++ {
		original := cache.MLALatent(layer, 0, position)
		restored := decoded.MLALatent(layer, 0, position)
		for index := range original {
			if original[index] != restored[index] {
				t.Fatalf("latent mismatch at layer %d index %d", layer, index)
			}
		}
		original = cache.MLARopeKey(layer, 0, position)
		restored = decoded.MLARopeKey(layer, 0, position)
		for index := range original {
			if original[index] != restored[index] {
				t.Fatalf("rope key mismatch at layer %d index %d", layer, index)
			}
		}
	}
}

func TestMLAKVBytesReduced(t *testing.T) {
	full := NewTransformer(testConfig())
	mla := NewTransformer(mlaTestConfig())
	if mla.KVBytesPerToken() >= full.KVBytesPerToken() {
		t.Fatalf("MLA KV bytes %d must be fewer than full %d", mla.KVBytesPerToken(), full.KVBytesPerToken())
	}
	if mla.KVReadBytes(8) >= full.KVReadBytes(8) {
		t.Fatal("MLA KV read bytes must be fewer than full")
	}
}

func mlaGQAConfig() Config {
	config := testConfig()
	config.NumKVHead = 1 // grouped-query attention
	config.MLALatent = 8
	config.MLARotaryDims = 8
	return config
}

func TestMLAWithGroupedQueryAttention(t *testing.T) {
	config := mlaGQAConfig()
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(7)
	for _, parameter := range transformer.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.1
		}
	}

	ids := tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 5, 2, 8, 3, 7})
	trainLogits, _ := transformer.TrainForward(ids)
	cache := newMLACache(config, 16)
	forwardLogits := transformer.Forward(ids, cache)
	for index := range trainLogits.Data {
		if math.Abs(float64(trainLogits.Data[index]-forwardLogits.Data[index])) > 1e-3 {
			t.Fatalf("GQA MLA logit %d: training=%v inference=%v", index, trainLogits.Data[index], forwardLogits.Data[index])
		}
	}
	runDirectionalGradientCheck(t, transformer)
}

// TestMLAForwardNilCache verifies that a cache-less full forward (used by
// evaluation and distillation) works and matches the cached forward.
func TestMLAForwardNilCache(t *testing.T) {
	transformer := mlaTinyModel()
	ids := tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 5, 2, 8, 3, 7})

	nilLogits := transformer.Forward(ids, nil)
	cache := newMLACache(transformer.Config, 16)
	cacheLogits := transformer.Forward(ids, cache)
	for index := range nilLogits.Data {
		if math.Abs(float64(nilLogits.Data[index]-cacheLogits.Data[index])) > 1e-5 {
			t.Fatalf("nil-cache logit %d: nil=%v cached=%v", index, nilLogits.Data[index], cacheLogits.Data[index])
		}
	}
}
