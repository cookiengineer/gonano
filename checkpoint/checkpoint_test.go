package checkpoint

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
)

func TestSaveLoadRoundtrip(t *testing.T) {
	cfg := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(42))
	params := m.NamedParameters()

	val := float32(0.5)
	meta := Meta{
		Step:        123,
		ValBPB:      &val,
		ModelConfig: cfg,
		LoopState:   map[string]any{"min_val_bpb": 0.4},
	}
	path := filepath.Join(t.TempDir(), "model_000123.gn")
	if err := Save(path, meta, params); err != nil {
		t.Fatalf("Save: %v", err)
	}

	gotMeta, gotParams, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if gotMeta.Step != 123 {
		t.Fatalf("step = %d, want 123", gotMeta.Step)
	}
	if gotMeta.ModelConfig != cfg {
		t.Fatalf("config = %+v, want %+v", gotMeta.ModelConfig, cfg)
	}
	if len(gotParams) != len(params) {
		t.Fatalf("params = %d, want %d", len(gotParams), len(params))
	}
	for name, p := range params {
		got := gotParams[name]
		if got == nil {
			t.Fatalf("missing param %s", name)
		}
		for i := range p.Data {
			if got.Data[i] != p.Data[i] {
				t.Fatalf("param %s[%d] = %v, want %v", name, i, got.Data[i], p.Data[i])
			}
		}
	}
}

func TestLoadModel(t *testing.T) {
	cfg := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(7))
	params := m.NamedParameters()

	path := filepath.Join(t.TempDir(), "model_000001.gn")
	if err := Save(path, Meta{Step: 1, ModelConfig: cfg}, params); err != nil {
		t.Fatalf("Save: %v", err)
	}
	meta, gotParams, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	rebuilt := LoadModel(meta, gotParams)
	idx := tensor.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	a := m.Forward(idx, nil)
	b := rebuilt.Forward(idx, nil)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("forward mismatch at %d after reload", i)
		}
	}
}

func TestFindLastStep(t *testing.T) {
	dir := t.TempDir()
	for _, step := range []int{1, 5, 42} {
		if err := os.WriteFile(ModelPath(dir, step), []byte("x"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	last, err := FindLastStep(dir)
	if err != nil {
		t.Fatalf("FindLastStep: %v", err)
	}
	if last != 42 {
		t.Fatalf("last = %d, want 42", last)
	}
}

func TestLoadRejectsNonCheckpoint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "junk.gn")
	if err := os.WriteFile(path, []byte("not a checkpoint"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(path); err == nil {
		t.Fatal("expected error for non-checkpoint file")
	}
}
