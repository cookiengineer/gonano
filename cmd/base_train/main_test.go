package main

import (
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/model/checkpoint"
	"github.com/cookiengineer/gonano/tensors"
)

// newTestCheckpoint writes a tiny checkpoint and returns its path and config.
func newTestCheckpoint(t *testing.T, config model.Config, step int) string {
	t.Helper()
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(1))
	path := filepath.Join(t.TempDir(), "model.gn")
	if err := checkpoint.Save(path, checkpoint.Meta{Step: step, ModelConfig: config}, transformer.NamedParameters()); err != nil {
		t.Fatalf("Save: %v", err)
	}
	return path
}

func TestLoadInitialCheckpoint(t *testing.T) {
	configuration := model.ConfigForPreset(model.PresetDense, 2, 256, 64, 64, 64, "SSSL")
	path := newTestCheckpoint(t, configuration, 7)

	meta, params, err := loadInitialCheckpoint(path, configuration.VocabSize)
	if err != nil {
		t.Fatalf("loadInitialCheckpoint: %v", err)
	}
	if meta.Step != 7 {
		t.Errorf("step = %d, want 7", meta.Step)
	}
	if meta.ModelConfig != configuration {
		t.Errorf("config mismatch:\n got %+v\nwant %+v", meta.ModelConfig, configuration)
	}
	if len(params) == 0 {
		t.Fatal("no parameters loaded")
	}
	if reconstructed := checkpoint.LoadModel(meta, params); reconstructed == nil {
		t.Fatal("LoadModel returned nil")
	}
}

func TestLoadInitialCheckpointVocabMismatch(t *testing.T) {
	configuration := model.ConfigForPreset(model.PresetDense, 2, 256, 64, 64, 64, "SSSL")
	path := newTestCheckpoint(t, configuration, 3)

	// The tokenizer has 64 more tokens than the checkpoint was trained with.
	if _, _, err := loadInitialCheckpoint(path, configuration.VocabSize+64); err == nil {
		t.Fatal("expected vocab mismatch error")
	}
}

func TestLoadInitialCheckpointMissing(t *testing.T) {
	if _, _, err := loadInitialCheckpoint(filepath.Join(t.TempDir(), "missing.gn"), 256); err == nil {
		t.Fatal("expected error for a missing checkpoint")
	}
}
