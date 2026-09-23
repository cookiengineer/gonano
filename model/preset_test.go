package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func presetTestConfig(preset Preset) Config {
	return ConfigForPreset(preset, 2, 16, 16, 32, 32, "SSSL")
}

func TestParsePreset(t *testing.T) {
	cases := map[string]Preset{
		"":        PresetFlash,
		"flash":   PresetFlash,
		"FLASH":   PresetFlash,
		" latent": PresetLatent,
		"dense":   PresetDense,
	}
	for name, want := range cases {
		got, err := ParsePreset(name)
		if err != nil {
			t.Fatalf("ParsePreset(%q): %v", name, err)
		}
		if got != want {
			t.Fatalf("ParsePreset(%q) = %q, want %q", name, got, want)
		}
	}
	if _, err := ParsePreset("nope"); err == nil {
		t.Fatal("ParsePreset should reject an unknown preset")
	}
}

func TestPresetUsesSinkhorn(t *testing.T) {
	if !PresetFlash.UsesSinkhorn() || !PresetLatent.UsesSinkhorn() {
		t.Fatal("flash and latent presets should use Sinkhorn")
	}
	if PresetDense.UsesSinkhorn() {
		t.Fatal("dense preset should not use Sinkhorn")
	}
}

func TestPresetDenseIsNoOp(t *testing.T) {
	before := ConfigForDepth(2, 16, 16, 32, 32, "SSSL")
	after := before
	after.ApplyPreset(PresetDense)
	if after != before {
		t.Fatalf("dense preset changed the config: %+v -> %+v", before, after)
	}
	if before.MoEEnabled() {
		t.Fatal("dense config should not enable MoE")
	}
}

func TestPresetFlashEnablesLongContextStack(t *testing.T) {
	config := ConfigForDepth(6, 16, 16, 32, 128, "SSSL")
	config.ApplyPreset(PresetFlash)
	config.Validate()
	if config.CompressionRatio != 4 || config.Compression() != 4 {
		t.Fatalf("compression = %d", config.CompressionRatio)
	}
	if config.SparseTopK != 8 || config.IndexerPool != 8 {
		t.Fatalf("sparse = %d pool = %d, want 8/8", config.SparseTopK, config.IndexerPool)
	}
	if config.ReusePattern != "FRU" {
		t.Fatalf("reuse pattern = %q, want FRU", config.ReusePattern)
	}
	if config.SWAWindow != 128 || !config.CED {
		t.Fatalf("SWA = %d CED = %v, want 128/true", config.SWAWindow, config.CED)
	}
	if !config.HeadWiseMuon {
		t.Fatal("flash preset should enable head-wise Muon")
	}
	if !config.MoEEnabled() {
		t.Fatal("flash preset should enable MoE")
	}
	if config.NumKVHead == config.NumHead && config.NumHead%2 == 0 {
		t.Fatal("flash preset should enable grouped-query attention")
	}
}

func TestPresetFlashSkipsCEDForSingleLayer(t *testing.T) {
	config := ConfigForDepth(1, 16, 16, 32, 32, "SSSL")
	config.ApplyPreset(PresetFlash)
	config.Validate()
	if config.CED {
		t.Fatal("single-layer flash config must not enable CED")
	}
	if !config.MoEEnabled() {
		t.Fatal("single-layer flash config should still enable MoE")
	}
}

func TestPresetLatentEnablesMLA(t *testing.T) {
	config := ConfigForDepth(2, 16, 16, 32, 32, "SSSL")
	config.ApplyPreset(PresetLatent)
	config.Validate()
	if !config.MLAEnabled() {
		t.Fatal("latent preset should enable MLA")
	}
	if config.HeadWiseMuon {
		t.Fatal("latent preset must not enable head-wise Muon (incompatible with MLA)")
	}
	if config.Compression() > 1 || config.CED || config.SparseTopK > 0 || config.ReusePattern != "" {
		t.Fatal("latent preset must leave the compression stack off")
	}
	if !config.MoEEnabled() {
		t.Fatal("latent preset should enable MoE")
	}
}

// TestPresetForwardAndTrainStep checks every preset builds a model that can
// forward and backpropagate finitely and set up its optimizer groups.
func TestPresetForwardAndTrainStep(t *testing.T) {
	for _, preset := range []Preset{PresetFlash, PresetLatent, PresetDense} {
		t.Run(string(preset), func(t *testing.T) {
			config := presetTestConfig(preset)
			model := NewTransformer(config)
			model.InitWeights(tensors.NewRNG(1))

			indexes, targets := tinyData()
			logits, context := model.TrainForward(indexes)
			for index, value := range logits.Data {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("non-finite logit at %d", index)
				}
			}
			flat := logits.Reshape(indexes.Numel(), config.VocabSize)
			gradLogits := tensors.CrossEntropyGrad(flat, targets.Reshape(indexes.Numel()), -1, 1)
			model.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], config.VocabSize))

			groups := model.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, preset.UsesSinkhorn())
			if len(groups) == 0 {
				t.Fatal("preset produced no optimizer groups")
			}
		})
	}
}

// TestConfigForPresetDefaultsToFlash documents that the zero preset name maps
// to the recommended architecture.
func TestConfigForPresetDefaultsToFlash(t *testing.T) {
	preset, err := ParsePreset("")
	if err != nil {
		t.Fatal(err)
	}
	if preset != DefaultPreset {
		t.Fatalf("default preset = %q, want %q", preset, DefaultPreset)
	}
	config := ConfigForPreset(preset, 2, 16, 16, 32, 32, "SSSL")
	if !config.MoEEnabled() || config.Compression() <= 1 || !config.CED {
		t.Fatal("default preset should be the flash long-context stack")
	}
}
