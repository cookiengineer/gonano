package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensor"
)

// Gradient-check the analytic backprop against finite differences. A
// directional check (projecting the full gradient onto a random direction) is
// far more robust to fp32 finite-difference noise than per-element checks.

func tinyTrainModel() *Transformer {
	cfg := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(42))
	// nanochat initializes projection layers to zero, which zeroes the
	// gradient flowing into the trunk at init. Perturb all weights so the
	// gradient check sees non-trivial signal through every layer.
	perturb := tensor.NewRNG(123)
	for _, p := range m.Parameters() {
		for i := range p.Data {
			p.Data[i] += perturb.NormFloat32() * 0.1
		}
	}
	return m
}

func tinyData() (*tensor.Int32s, *tensor.Int32s) {
	idx := tensor.NewInt32sWithData([]int{1, 6}, []int32{1, 5, 2, 8, 3, 7})
	targets := tensor.NewInt32sWithData([]int{1, 6}, []int32{5, 2, 8, 3, 7, 4})
	return idx, targets
}

func meanLoss(m *Transformer, idx, targets *tensor.Int32s) float32 {
	logits, _ := m.TrainForward(idx)
	flat := logits.Reshape(idx.Numel(), m.Config.VocabSize)
	tflat := targets.Reshape(idx.Numel())
	return tensor.CrossEntropy(flat, tflat, -1)
}

func analyticGrads(m *Transformer, idx, targets *tensor.Int32s) {
	m.ZeroGrad()
	logits, ctx := m.TrainForward(idx)
	flat := logits.Reshape(idx.Numel(), m.Config.VocabSize)
	tflat := targets.Reshape(idx.Numel())
	_, valid := tensor.CrossEntropyPerPosition(flat, tflat, -1)
	gradLogits := tensor.CrossEntropyGrad(flat, tflat, -1, 1/float32(valid))
	gradLogits = gradLogits.Reshape(idx.Shape[0], idx.Shape[1], m.Config.VocabSize)
	m.TrainBackward(ctx, gradLogits)
}

func TestBackpropDirectionalGradientCheck(t *testing.T) {
	m := tinyTrainModel()
	idx, targets := tinyData()
	analyticGrads(m, idx, targets)

	// Build a random direction and compute the analytic projection.
	rng := tensor.NewRNG(999)
	params := m.Parameters()
	directions := make([][]float32, len(params))
	var analytic float64
	for i, p := range params {
		directions[i] = make([]float32, p.Numel())
		for j := range p.Data {
			v := rng.NormFloat32()
			directions[i][j] = v
			analytic += float64(p.Grad[j]) * float64(v)
		}
	}

	eps := float32(1e-3)
	for i, p := range params {
		for j := range p.Data {
			p.Data[j] += eps * directions[i][j]
		}
	}
	lplus := meanLoss(m, idx, targets)
	for i, p := range params {
		for j := range p.Data {
			p.Data[j] -= 2 * eps * directions[i][j]
		}
	}
	lminus := meanLoss(m, idx, targets)
	for i, p := range params {
		for j := range p.Data {
			p.Data[j] += eps * directions[i][j]
		}
	}
	numeric := float64(lplus-lminus) / (2 * float64(eps))

	scale := math.Abs(analytic) + 1e-6
	if math.Abs(numeric-analytic) > 5e-2*scale {
		t.Fatalf("directional gradient mismatch: analytic=%v numeric=%v", analytic, numeric)
	}
}

func TestBackpropPerElementGradientCheck(t *testing.T) {
	m := tinyTrainModel()
	idx, targets := tinyData()
	analyticGrads(m, idx, targets)

	checks := []struct {
		name string
		p    *tensor.Tensor
		idx  int
	}{
		{"wte", m.wte.Weight, 100},
		{"lm_head", m.lm.Weight, 50},
		{"attn.c_q", m.h[0].Attn.cq.Weight, 0},
		{"attn.c_k", m.h[0].Attn.ck.Weight, 3},
		{"attn.c_v", m.h[0].Attn.cv.Weight, 7},
		{"attn.c_proj", m.h[0].Attn.cproj.Weight, 2},
		{"mlp.c_fc", m.h[0].MLP.CFc.Weight, 5},
		{"mlp.c_proj", m.h[0].MLP.CProj.Weight, 9},
		{"resid_lambdas", m.residLambdas, 0},
		{"x0_lambdas", m.x0Lambdas, 1},
		{"smear_gate", m.smearGate.Weight, 0},
		{"smear_lambda", m.smearLambda, 0},
		{"backout_lambda", m.backoutLambda, 0},
		{"value_embeds.1", m.valueEmbeds[1].Weight, 11},
		{"ve_gate.1", m.h[1].Attn.veGate.Weight, 4},
	}

	eps := float32(1e-3)
	for _, ck := range checks {
		analytic := ck.p.Grad[ck.idx]
		orig := ck.p.Data[ck.idx]
		ck.p.Data[ck.idx] = orig + eps
		lplus := meanLoss(m, idx, targets)
		ck.p.Data[ck.idx] = orig - eps
		lminus := meanLoss(m, idx, targets)
		ck.p.Data[ck.idx] = orig
		numeric := (lplus - lminus) / (2 * eps)

		// fp32 finite differences are noisy for small gradients; use a
		// relative tolerance anchored to the larger magnitude.
		scale := math.Abs(float64(analytic))
		if math.Abs(float64(numeric)) > scale {
			scale = math.Abs(float64(numeric))
		}
		scale = math.Max(scale, 1e-3)
		if math.Abs(float64(numeric-analytic)) > 0.5*scale {
			t.Errorf("%s[%d]: analytic=%v numeric=%v", ck.name, ck.idx, analytic, numeric)
		}
	}
}
