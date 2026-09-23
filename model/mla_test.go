package model

import (
	"fmt"
	"math"
	"testing"

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
