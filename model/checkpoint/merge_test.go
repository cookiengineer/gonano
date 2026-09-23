package checkpoint

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

func mergeTestConfig() model.Config {
	return model.Config{SequenceLen: 16, VocabSize: 16, NumLayer: 1, NumHead: 2, NumKVHead: 2, EmbedDim: 32, WindowPattern: "L"}
}

func mergeTestMap(values ...map[string][]float32) map[string]*tensors.Tensor {
	out := make(map[string]*tensors.Tensor)
	for _, value := range values {
		for name, data := range value {
			out[name] = tensors.NewWithData([]int{len(data)}, append([]float32(nil), data...))
		}
	}
	return out
}

func TestMergeParametersUniform(t *testing.T) {
	first := mergeTestMap(map[string][]float32{"w": {1, 3}, "b": {2, 4}})
	second := mergeTestMap(map[string][]float32{"w": {3, 1}, "b": {4, 2}})
	merged, err := MergeParameters([]map[string]*tensors.Tensor{first, second}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := merged["w"].Data; got[0] != 2 || got[1] != 2 {
		t.Fatalf("w = %v, want [2 2]", got)
	}
	if got := merged["b"].Data; got[0] != 3 || got[1] != 3 {
		t.Fatalf("b = %v, want [3 3]", got)
	}
	// Inputs must not be mutated.
	if first["w"].Data[0] != 1 {
		t.Fatal("merge mutated its input")
	}
}

func TestMergeParametersWeighted(t *testing.T) {
	first := mergeTestMap(map[string][]float32{"w": {0, 0}})
	second := mergeTestMap(map[string][]float32{"w": {10, 20}})
	merged, err := MergeParameters([]map[string]*tensors.Tensor{first, second}, []float32{1, 3})
	if err != nil {
		t.Fatal(err)
	}
	if got := merged["w"].Data; math.Abs(float64(got[0]-7.5)) > 1e-6 || math.Abs(float64(got[1]-15)) > 1e-6 {
		t.Fatalf("weighted w = %v, want [7.5 15]", got)
	}
}

func TestMergeParametersShapeMismatch(t *testing.T) {
	first := map[string]*tensors.Tensor{"w": tensors.New(2)}
	second := map[string]*tensors.Tensor{"w": tensors.New(3)}
	if _, err := MergeParameters([]map[string]*tensors.Tensor{first, second}, nil); err == nil {
		t.Fatal("expected a shape mismatch error")
	}
}

func TestMergeParametersMissing(t *testing.T) {
	first := map[string]*tensors.Tensor{"w": tensors.New(2), "b": tensors.New(2)}
	second := map[string]*tensors.Tensor{"w": tensors.New(2)}
	if _, err := MergeParameters([]map[string]*tensors.Tensor{first, second}, nil); err == nil {
		t.Fatal("expected a missing-parameter error")
	}
}

func TestMergeFilesRoundTrip(t *testing.T) {
	directory := t.TempDir()
	config := mergeTestConfig()
	firstPath := filepath.Join(directory, "a.gn")
	secondPath := filepath.Join(directory, "b.gn")
	outPath := filepath.Join(directory, "merged.gn")

	firstParams := map[string]*tensors.Tensor{"w": tensors.NewWithData([]int{2}, []float32{1, 5})}
	secondParams := map[string]*tensors.Tensor{"w": tensors.NewWithData([]int{2}, []float32{3, 1})}
	if err := Save(firstPath, Meta{ModelConfig: config}, firstParams); err != nil {
		t.Fatal(err)
	}
	if err := Save(secondPath, Meta{ModelConfig: config}, secondParams); err != nil {
		t.Fatal(err)
	}
	if err := MergeFiles([]string{firstPath, secondPath}, nil, outPath); err != nil {
		t.Fatal(err)
	}
	meta, merged, err := Load(outPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := merged["w"].Data; got[0] != 2 || got[1] != 3 {
		t.Fatalf("merged w = %v, want [2 3]", got)
	}
	sources, ok := meta.UserConfig["merged_from"].([]any)
	if !ok || len(sources) != 2 {
		t.Fatalf("merged_from metadata = %v", meta.UserConfig["merged_from"])
	}
}
