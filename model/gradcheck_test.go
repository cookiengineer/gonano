package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// Gradient-check the analytic backprop against finite differences. A
// directional check (projecting the full gradient onto a random direction) is
// far more robust to fp32 finite-difference noise than per-element checks.

func tinyTrainModel() *Transformer {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(42))
	// nanochat initializes projection layers to zero, which zeroes the
	// gradient flowing into the trunk at init. Perturb all weights so the
	// gradient check sees non-trivial signal through every layer.
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
		}
	}
	return model
}

func tinyCompressedModel() *Transformer {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2,
	}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
		}
	}
	return model
}

// tinyReuseModel exercises cross-layer dense-compressed reuse: layer 0 is full
// and layers 1-3 reuse its compressed KV. Dense compression is used because
// sparse top-k selection is non-differentiable, which would invalidate a
// finite-difference gradient check.
func tinyReuseModel() *Transformer {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 4, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, ReusePattern: "FRUU",
	}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
		}
	}
	return model
}

func tinyData() (*tensors.Int32s, *tensors.Int32s) {
	indexes := tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 5, 2, 8, 3, 7})
	targets := tensors.NewInt32sWithData([]int{1, 6}, []int32{5, 2, 8, 3, 7, 4})
	return indexes, targets
}

func meanLoss(model *Transformer, indexes, targets *tensors.Int32s) float32 {
	logits, _ := model.TrainForward(indexes)
	flattened := logits.Reshape(indexes.Numel(), model.Config.VocabSize)
	targetFlat := targets.Reshape(indexes.Numel())
	return tensors.CrossEntropy(flattened, targetFlat, -1)
}

func analyticGrads(model *Transformer, indexes, targets *tensors.Int32s) {
	model.ZeroGrad()
	logits, context := model.TrainForward(indexes)
	flattened := logits.Reshape(indexes.Numel(), model.Config.VocabSize)
	targetFlat := targets.Reshape(indexes.Numel())
	_, valid := tensors.CrossEntropyPerPosition(flattened, targetFlat, -1)
	gradLogits := tensors.CrossEntropyGrad(flattened, targetFlat, -1, 1/float32(valid))
	gradLogits = gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], model.Config.VocabSize)
	model.TrainBackward(context, gradLogits)
}

func TestBackpropDirectionalGradientCheck(t *testing.T) {
	runDirectionalGradientCheck(t, tinyTrainModel())
}

func TestBackpropDirectionalGradientCheckCompressed(t *testing.T) {
	runDirectionalGradientCheck(t, tinyCompressedModel())
}

// TestBackpropDirectionalGradientCheckReuse verifies that the compressed
// key/value gradients contributed by reindex/reuse layers reach the producing
// layer's compressor.
func TestBackpropDirectionalGradientCheckReuse(t *testing.T) {
	runDirectionalGradientCheck(t, tinyReuseModel())
}

func runDirectionalGradientCheck(t *testing.T, model *Transformer) {
	indexes, targets := tinyData()
	analyticGrads(model, indexes, targets)

	// Build a random direction and compute the analytic projection.
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
		t.Fatalf("directional gradient mismatch: analytic=%v numeric=%v", analytic, numeric)
	}
}

func TestBackpropPerElementGradientCheck(t *testing.T) {
	model := tinyTrainModel()
	indexes, targets := tinyData()
	analyticGrads(model, indexes, targets)

	checks := []struct {
		name         string
		parameter    *tensors.Tensor
		elementIndex int
	}{
		{"wte", model.tokenEmbedding.Weight, 100},
		{"lm_head", model.lmHead.Weight, 50},
		{"attn.c_q", model.blocks[0].attention.queryProjection.Weight, 0},
		{"attn.c_k", model.blocks[0].attention.keyProjection.Weight, 3},
		{"attn.c_v", model.blocks[0].attention.valueProjection.Weight, 7},
		{"attn.c_proj", model.blocks[0].attention.outputProjection.Weight, 2},
		{"mlp.c_fc", model.blocks[0].mlp.inputProjection.Weight, 5},
		{"mlp.c_proj", model.blocks[0].mlp.outputProjection.Weight, 9},
		{"resid_lambdas", model.residLambdas, 0},
		{"x0_lambdas", model.x0Lambdas, 1},
		{"smear_gate", model.smearGate.Weight, 0},
		{"smear_lambda", model.smearLambda, 0},
		{"backout_lambda", model.backoutLambda, 0},
		{"value_embeds.1", model.valueEmbeds[1].Weight, 11},
		{"ve_gate.1", model.blocks[1].attention.valueEmbeddingGate.Weight, 4},
	}

	epsilon := float32(1e-3)
	for _, check := range checks {
		analytic := check.parameter.Grad[check.elementIndex]
		original := check.parameter.Data[check.elementIndex]
		check.parameter.Data[check.elementIndex] = original + epsilon
		lossPlus := meanLoss(model, indexes, targets)
		check.parameter.Data[check.elementIndex] = original - epsilon
		lossMinus := meanLoss(model, indexes, targets)
		check.parameter.Data[check.elementIndex] = original
		numeric := (lossPlus - lossMinus) / (2 * epsilon)

		// fp32 finite differences are noisy for small gradients; use a
		// relative tolerance anchored to the larger magnitude.
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
