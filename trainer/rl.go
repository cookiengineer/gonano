package trainer

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// PolicyGradientStep performs one REINFORCE/GRPO-style policy-gradient update.
// inputs/targets are a batch of rollouts (shape [B,T]); advantages is one
// advantage per sequence; norm is the normalization denominator
// (num_valid_tokens * num_passes * examples_per_rank). It returns the policy
// gradient objective (for logging).
func PolicyGradientStep(transformer *model.Transformer, inputs, targets *tensors.Int32s, advantages []float32, norm float32) float32 {
	batchSize, sequenceLen := inputs.Shape[0], inputs.Shape[1]
	vocabSize := transformer.Config.VocabSize

	logits, trainContext := transformer.TrainForward(inputs) // [B,T,vocab]
	flat := logits.Reshape(batchSize*sequenceLen, vocabSize)
	targetsFlat := targets.Reshape(batchSize * sequenceLen)

	// Per-token negative log-likelihood (logp = -nll).
	negativeLogLikelihood, _ := tensors.CrossEntropyPerPosition(flat, targetsFlat, -1)

	// Objective: loss = -sum(logp * adv) / norm = sum(nll * adv) / norm.
	var loss float64
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for position := 0; position < sequenceLen; position++ {
			flatIndex := batchIndex*sequenceLen + position
			if targetsFlat.Data[flatIndex] == -1 {
				continue
			}
			loss += float64(negativeLogLikelihood[flatIndex]) * float64(advantages[batchIndex]) / float64(norm)
		}
	}

	// Gradient wrt logits = (softmax - onehot) * adv / norm per token.
	probabilities := tensors.SoftmaxLastDim(flat) // [b*t, vocab]
	gradient := tensors.New(batchSize*sequenceLen, vocabSize)
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		advantageScale := advantages[batchIndex] / norm
		for position := 0; position < sequenceLen; position++ {
			target := targetsFlat.Data[batchIndex*sequenceLen+position]
			if target == -1 {
				continue
			}
			baseOffset := (batchIndex*sequenceLen + position) * vocabSize
			for vocabIndex := 0; vocabIndex < vocabSize; vocabIndex++ {
				gradientValue := probabilities.Data[baseOffset+vocabIndex]
				if int32(vocabIndex) == target {
					gradientValue -= 1
				}
				gradient.Data[baseOffset+vocabIndex] = gradientValue * advantageScale
			}
		}
	}

	transformer.TrainBackward(trainContext, gradient.Reshape(batchSize, sequenceLen, vocabSize))
	return float32(loss)
}

// RLStep runs a policy-gradient update and an optimizer step.
func RLStep(transformer *model.Transformer, optimizer *optimizer.MuonAdamW, inputs, targets *tensors.Int32s, advantages []float32, norm float32) float32 {
	loss := PolicyGradientStep(transformer, inputs, targets, advantages, norm)
	optimizer.Step()
	optimizer.ZeroGrad()
	return loss
}
