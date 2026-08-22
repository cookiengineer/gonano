package train

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optim"
	"github.com/cookiengineer/gonano/tensor"
)

// PolicyGradientStep performs one REINFORCE/GRPO-style policy-gradient update.
// inputs/targets are a batch of rollouts (shape [B,T]); advantages is one
// advantage per sequence; norm is the normalization denominator
// (num_valid_tokens * num_passes * examples_per_rank). It returns the policy
// gradient objective (for logging).
func PolicyGradientStep(m *model.Transformer, inputs, targets *tensor.Int32s, advantages []float32, norm float32) float32 {
	b, t := inputs.Shape[0], inputs.Shape[1]
	vocab := m.Config.VocabSize

	logits, ctx := m.TrainForward(inputs) // [B,T,vocab]
	flat := logits.Reshape(b*t, vocab)
	tflat := targets.Reshape(b * t)

	// Per-token negative log-likelihood (logp = -nll).
	nll, _ := tensor.CrossEntropyPerPosition(flat, tflat, -1)

	// Objective: loss = -sum(logp * adv) / norm = sum(nll * adv) / norm.
	var loss float64
	for i := 0; i < b; i++ {
		for j := 0; j < t; j++ {
			idx := i*t + j
			if tflat.Data[idx] == -1 {
				continue
			}
			loss += float64(nll[idx]) * float64(advantages[i]) / float64(norm)
		}
	}

	// Gradient wrt logits = (softmax - onehot) * adv / norm per token.
	probs := tensor.SoftmaxLastDim(flat) // [b*t, vocab]
	grad := tensor.New(b*t, vocab)
	for i := 0; i < b; i++ {
		scale := advantages[i] / norm
		for j := 0; j < t; j++ {
			y := tflat.Data[i*t+j]
			if y == -1 {
				continue
			}
			base := (i*t + j) * vocab
			for k := 0; k < vocab; k++ {
				g := probs.Data[base+k]
				if int32(k) == y {
					g -= 1
				}
				grad.Data[base+k] = g * scale
			}
		}
	}

	m.TrainBackward(ctx, grad.Reshape(b, t, vocab))
	return float32(loss)
}

// RLStep runs a policy-gradient update and an optimizer step.
func RLStep(m *model.Transformer, opt *optim.MuonAdamW, inputs, targets *tensor.Int32s, advantages []float32, norm float32) float32 {
	loss := PolicyGradientStep(m, inputs, targets, advantages, norm)
	opt.Step()
	opt.ZeroGrad()
	return loss
}
