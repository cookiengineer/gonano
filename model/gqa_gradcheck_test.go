package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// TestBackpropGradientCheckWithGroupedQueryAttention exercises the training
// backward when several query heads share one key/value head. Before the
// kernel contract existed, this path overwrote (instead of accumulated) the
// shared key/value gradients, so a grouped-query model could not pass a
// gradient check.
func TestBackpropGradientCheckWithGroupedQueryAttention(t *testing.T) {
	configuration := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 4, NumKVHead: 1,
		EmbedDim: 32, WindowPattern: "L",
	}
	model := NewTransformer(configuration)
	model.InitWeights(tensors.NewRNG(42))
	// Zero-initialized projections starve the gradient at init; perturb every
	// weight so the check sees signal through all layers.
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.1
		}
	}

	indexes, targets := tinyData()
	analyticGrads(model, indexes, targets)

	rng := tensors.NewRNG(999)
	parameters := model.Parameters()
	directions := make([][]float32, len(parameters))
	var analytic float64
	for parameterIndex, parameter := range parameters {
		directions[parameterIndex] = make([]float32, parameter.Numel())
		for elementIndex := range parameter.Data {
			value := rng.NormFloat32()
			directions[parameterIndex][elementIndex] = value
			analytic += float64(parameter.Grad[elementIndex]) * float64(value)
		}
	}

	epsilon := float32(1e-3)
	for parameterIndex, parameter := range parameters {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += epsilon * directions[parameterIndex][elementIndex]
		}
	}
	lossPlus := meanLoss(model, indexes, targets)
	for parameterIndex, parameter := range parameters {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] -= 2 * epsilon * directions[parameterIndex][elementIndex]
		}
	}
	lossMinus := meanLoss(model, indexes, targets)
	for parameterIndex, parameter := range parameters {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += epsilon * directions[parameterIndex][elementIndex]
		}
	}
	numeric := float64(lossPlus-lossMinus) / (2 * float64(epsilon))

	scale := math.Abs(analytic) + 1e-6
	if math.Abs(numeric-analytic) > 5e-2*scale {
		t.Fatalf("grouped-query directional gradient mismatch: analytic=%v numeric=%v", analytic, numeric)
	}
}
