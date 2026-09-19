package checkpoint

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

func TestSaveLoadRoundtrip(tester *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	params := transformer.NamedParameters()

	value := float32(0.5)
	meta := Meta{
		Step:        123,
		ValBPB:      &value,
		ModelConfig: config,
		LoopState:   map[string]any{"min_val_bpb": 0.4},
	}
	path := filepath.Join(tester.TempDir(), "model_000123.gn")
	if err := Save(path, meta, params); err != nil {
		tester.Fatalf("Save: %v", err)
	}

	gotMeta, gotParams, err := Load(path)
	if err != nil {
		tester.Fatalf("Load: %v", err)
	}
	if gotMeta.Step != 123 {
		tester.Fatalf("step = %d, want 123", gotMeta.Step)
	}
	if gotMeta.ModelConfig != config {
		tester.Fatalf("config = %+v, want %+v", gotMeta.ModelConfig, config)
	}
	if len(gotParams) != len(params) {
		tester.Fatalf("params = %d, want %d", len(gotParams), len(params))
	}
	for name, param := range params {
		got := gotParams[name]
		if got == nil {
			tester.Fatalf("missing param %s", name)
		}
		for index := range param.Data {
			if got.Data[index] != param.Data[index] {
				tester.Fatalf("param %s[%d] = %v, want %v", name, index, got.Data[index], param.Data[index])
			}
		}
	}
}

func TestLoadModel(tester *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(7))
	params := transformer.NamedParameters()

	path := filepath.Join(tester.TempDir(), "model_000001.gn")
	if err := Save(path, Meta{Step: 1, ModelConfig: config}, params); err != nil {
		tester.Fatalf("Save: %v", err)
	}
	meta, gotParams, err := Load(path)
	if err != nil {
		tester.Fatalf("Load: %v", err)
	}
	reloaded := LoadModel(meta, gotParams)
	inputIDs := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	original := transformer.Forward(inputIDs, nil)
	rebuilt := reloaded.Forward(inputIDs, nil)
	for index := range original.Data {
		if original.Data[index] != rebuilt.Data[index] {
			tester.Fatalf("forward mismatch at %d after reload", index)
		}
	}
}

func TestLoadModelPacksInt8QAT(tester *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", QAT: "int8",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(7))

	path := filepath.Join(tester.TempDir(), "model_000001.gn")
	if err := Save(path, Meta{Step: 1, ModelConfig: config}, transformer.NamedParameters()); err != nil {
		tester.Fatalf("Save: %v", err)
	}
	meta, gotParams, err := Load(path)
	if err != nil {
		tester.Fatalf("Load: %v", err)
	}
	reloaded := LoadModel(meta, gotParams)
	if !reloaded.Int8InferenceEnabled() {
		tester.Fatal("a QAT int8 checkpoint should be packed for int8 inference on load")
	}
}

func TestFindLastStep(tester *testing.T) {
	dir := tester.TempDir()
	for _, step := range []int{1, 5, 42} {
		if err := os.WriteFile(ModelPath(dir, step), []byte("x"), 0o644); err != nil {
			tester.Fatalf("write: %v", err)
		}
	}
	lastStep, err := FindLastStep(dir)
	if err != nil {
		tester.Fatalf("FindLastStep: %v", err)
	}
	if lastStep != 42 {
		tester.Fatalf("last = %d, want 42", lastStep)
	}
}

func TestLoadRejectsNonCheckpoint(tester *testing.T) {
	path := filepath.Join(tester.TempDir(), "junk.gn")
	if err := os.WriteFile(path, []byte("not a checkpoint"), 0o644); err != nil {
		tester.Fatal(err)
	}
	if _, _, err := Load(path); err == nil {
		tester.Fatal("expected error for non-checkpoint file")
	}
}
