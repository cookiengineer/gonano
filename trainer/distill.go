package trainer

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// DistillStep performs one on-policy distillation micro-batch: the frozen
// teacher scores the inputs, the student is trained to match the teacher's full
// vocabulary distribution (DeepSeek-V4.1 §5.2.4), and the resulting gradient is
// backpropagated into the student. It returns the mean per-token loss.
func (trainer *Trainer) DistillStep(teacher *model.Transformer, inputs, mask *tensors.Int32s, temperature float32) float32 {
	vocabSize := trainer.Model.Config.VocabSize
	flatRows := inputs.Numel()

	accumulation := trainer.GradAccumSteps
	if accumulation < 1 {
		accumulation = 1
	}
	trainer.Model.SetAuxiliaryGradientScale(1 / float32(accumulation))

	teacherLogits := teacher.Forward(inputs, nil) // [B,T,vocab], no gradient
	logits, trainContext := trainer.Model.TrainForward(inputs)
	studentFlat := logits.Reshape(flatRows, vocabSize)
	teacherFlat := teacherLogits.Reshape(flatRows, vocabSize)

	var flatMask *tensors.Int32s
	if mask != nil {
		flatMask = mask.Reshape(flatRows)
	}
	losses, validTokens := tensors.DistillationLossPerPosition(studentFlat, teacherFlat, temperature, flatMask)
	if validTokens == 0 {
		validTokens = 1
	}
	scale := 1.0 / (float32(validTokens) * float32(trainer.GradAccumSteps))
	gradient := tensors.DistillationGrad(studentFlat, teacherFlat, temperature, flatMask, scale)
	trainer.Model.TrainBackward(trainContext, gradient.Reshape(inputs.Shape[0], inputs.Shape[1], vocabSize))

	var sum float64
	for _, loss := range losses {
		sum += float64(loss)
	}
	return float32(sum / float64(validTokens))
}

// Distill runs on-policy distillation for numIterations steps, or until next
// returns ok == false. next yields one micro-batch of inputs and an optional
// mask; a non-zero mask entry marks a position supervised by the teacher, and a
// nil mask supervises every position. It returns the per-step losses.
func Distill(student, teacher *model.Transformer, groups []optimizer.ParamGroup, next func() (*tensors.Int32s, *tensors.Int32s, bool), numIterations int, temperature float32, evaluateFunc func(step int, loss float32)) []float32 {
	trainer := NewTrainer(student, groups, 1)
	trainer.WarmupSteps = 0
	trainer.WarmdownRatio = 0.5
	trainer.FinalLRFrac = 0

	var losses []float32
	for step := 0; ; step++ {
		if numIterations > 0 && step >= numIterations {
			break
		}
		inputs, mask, ok := next()
		if !ok {
			break
		}
		loss := trainer.DistillStep(teacher, inputs, mask, temperature)
		trainer.StepOptimizer(step, max(1, numIterations))
		losses = append(losses, loss)
		if evaluateFunc != nil {
			evaluateFunc(step, loss)
		}
	}
	return losses
}
