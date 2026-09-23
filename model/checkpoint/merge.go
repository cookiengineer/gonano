package checkpoint

import (
	"fmt"

	"github.com/cookiengineer/gonano/tensors"
)

// MergeParameters averages parameter maps from several checkpoints with the
// given weights. All maps must contain exactly the same parameter names with
// identical shapes. A nil or empty weights slice uses uniform weights; weights
// are normalized to sum to one. It returns fresh tensors and does not alias the
// inputs.
func MergeParameters(models []map[string]*tensors.Tensor, weights []float32) (map[string]*tensors.Tensor, error) {
	if len(models) == 0 {
		return nil, fmt.Errorf("checkpoint: no models to merge")
	}
	if len(weights) == 0 {
		weights = make([]float32, len(models))
		for index := range weights {
			weights[index] = 1
		}
	}
	if len(weights) != len(models) {
		return nil, fmt.Errorf("checkpoint: %d weights for %d models", len(weights), len(models))
	}
	var total float32
	for _, weight := range weights {
		total += weight
	}
	if total == 0 {
		return nil, fmt.Errorf("checkpoint: merge weights sum to zero")
	}

	reference := models[0]
	for modelIndex := 1; modelIndex < len(models); modelIndex++ {
		candidate := models[modelIndex]
		if len(candidate) != len(reference) {
			return nil, fmt.Errorf("checkpoint: model %d has %d parameters, want %d", modelIndex, len(candidate), len(reference))
		}
		for name, parameter := range reference {
			other, ok := candidate[name]
			if !ok {
				return nil, fmt.Errorf("checkpoint: model %d is missing parameter %q", modelIndex, name)
			}
			if !sameShape(parameter.Shape, other.Shape) {
				return nil, fmt.Errorf("checkpoint: parameter %q shape differs (%v vs %v)", name, parameter.Shape, other.Shape)
			}
		}
	}

	merged := make(map[string]*tensors.Tensor, len(reference))
	for name, parameter := range reference {
		out := tensors.New(parameter.Shape...)
		for modelIndex, model := range models {
			scale := weights[modelIndex] / total
			source := model[name].Data
			for element := range out.Data {
				out.Data[element] += scale * source[element]
			}
		}
		merged[name] = out
	}
	return merged, nil
}

// MergeFiles loads several checkpoints, verifies they share an architecture,
// averages their parameters, and writes the merged checkpoint to outPath. The
// merged metadata records the source paths in its user config (the lightweight
// reinitialization step DeepSeek-V4.1 §5.1.2 applies between successive RL
// runs).
func MergeFiles(paths []string, weights []float32, outPath string) error {
	if len(paths) == 0 {
		return fmt.Errorf("checkpoint: no checkpoints to merge")
	}
	metas := make([]Meta, len(paths))
	parameterSets := make([]map[string]*tensors.Tensor, len(paths))
	for index, path := range paths {
		meta, parameters, err := Load(path)
		if err != nil {
			return err
		}
		metas[index] = meta
		parameterSets[index] = parameters
	}
	for index := 1; index < len(paths); index++ {
		if metas[index].ModelConfig != metas[0].ModelConfig {
			return fmt.Errorf("checkpoint: %s and %s have different model configs", paths[0], paths[index])
		}
	}
	merged, err := MergeParameters(parameterSets, weights)
	if err != nil {
		return err
	}
	meta := metas[0]
	meta.Step = 0
	if meta.UserConfig == nil {
		meta.UserConfig = make(map[string]any)
	}
	meta.UserConfig["merged_from"] = append([]string(nil), paths...)
	return Save(outPath, meta, merged)
}

// sameShape reports whether two shapes are element-wise equal.
func sameShape(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
