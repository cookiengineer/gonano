package checkpoint

import (
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// moeConfig returns a tiny MoE config with non-default clamp/scale so the
// metadata round-trip is exercised.
func moeConfig() model.Config {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
		NumExperts: 4, NumExpertsPerToken: 2,
		ExpertHiddenDim: 16, SharedExpertHiddenDim: 16,
		MoEClamp: 7.5, MoEScale: 2.5,
	}
	config.ApplyMoEDefaults()
	return config
}

// TestSaveLoadRoundtripMoE verifies that a MoE model round-trips through the
// native .gn checkpoint and reproduces the same forward pass.
func TestSaveLoadRoundtripMoE(tester *testing.T) {
	config := moeConfig()
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	params := transformer.NamedParameters()

	path := filepath.Join(tester.TempDir(), "model_000001.gn")
	if err := Save(path, Meta{Step: 1, ModelConfig: config}, params); err != nil {
		tester.Fatalf("Save: %v", err)
	}
	meta, gotParams, err := Load(path)
	if err != nil {
		tester.Fatalf("Load: %v", err)
	}
	if meta.ModelConfig != config {
		tester.Fatalf("config = %+v, want %+v", meta.ModelConfig, config)
	}
	if len(gotParams) != len(params) {
		tester.Fatalf("params = %d, want %d", len(gotParams), len(params))
	}
	reloaded := LoadModel(meta, gotParams)

	inputIDs := tensors.NewInt32sWithData([]int{1, 5}, []int32{1, 2, 3, 4, 5})
	original := transformer.Forward(inputIDs, nil)
	rebuilt := reloaded.Forward(inputIDs, nil)
	for index := range original.Data {
		if original.Data[index] != rebuilt.Data[index] {
			tester.Fatalf("forward mismatch at %d after reload", index)
		}
	}
}

// TestExportGGUFMoE verifies that the MoE architecture metadata survives a GGUF
// round-trip and that the rebuilt model forwards identically.
func TestExportGGUFMoE(tester *testing.T) {
	config := moeConfig()
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(7))

	path := filepath.Join(tester.TempDir(), "model.gguf")
	if err := ExportGGUF(path, Meta{Step: 3, ModelConfig: config}, transformer.NamedParameters()); err != nil {
		tester.Fatalf("ExportGGUF: %v", err)
	}

	meta, params, err := LoadGGUF(path)
	if err != nil {
		tester.Fatalf("LoadGGUF: %v", err)
	}
	if meta.ModelConfig != config {
		tester.Fatalf("config = %+v, want %+v", meta.ModelConfig, config)
	}
	if meta.ModelConfig.NumExperts != 4 || meta.ModelConfig.NumExpertsPerToken != 2 {
		tester.Fatalf("MoE routing metadata lost")
	}
	if meta.ModelConfig.MoEClamp != 7.5 || meta.ModelConfig.MoEScale != 2.5 {
		tester.Fatalf("MoE clamp/scale metadata lost: clamp=%v scale=%v", meta.ModelConfig.MoEClamp, meta.ModelConfig.MoEScale)
	}

	reloaded := LoadModel(meta, params)
	inputIDs := tensors.NewInt32sWithData([]int{1, 5}, []int32{1, 2, 3, 4, 5})
	original := transformer.Forward(inputIDs, nil)
	rebuilt := reloaded.Forward(inputIDs, nil)
	for index := range original.Data {
		if original.Data[index] != rebuilt.Data[index] {
			tester.Fatalf("forward mismatch at %d after GGUF round-trip", index)
		}
	}
}

// TestGGUFWithoutMoEMetadataStillLoads verifies backward compatibility: a GGUF
// file written before the MoE metadata keys existed still parses.
func TestGGUFWithoutMoEMetadataStillLoads(tester *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(9))
	path := filepath.Join(tester.TempDir(), "dense.gguf")
	if err := ExportGGUF(path, Meta{Step: 2, ModelConfig: config}, transformer.NamedParameters()); err != nil {
		tester.Fatalf("ExportGGUF: %v", err)
	}
	meta, params, err := LoadGGUF(path)
	if err != nil {
		tester.Fatalf("LoadGGUF: %v", err)
	}
	if meta.ModelConfig.MoEEnabled() {
		tester.Fatal("dense config should not report MoE enabled")
	}
	reloaded := LoadModel(meta, params)
	if reloaded.Config.NumExperts != 0 {
		tester.Fatal("reloaded dense model should have no experts")
	}
}
