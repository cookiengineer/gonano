package trainer

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// Trainer drives a single training run: forward/backward over micro-batches
// with gradient accumulation, then optimizer steps with LR/momentum/weight-
// decay scheduling.
type Trainer struct {
	Model          *model.Transformer
	Optim          *optimizer.MuonAdamW
	GradAccumSteps int

	// schedule configuration
	WarmupSteps   int
	WarmdownRatio float32
	FinalLRFrac   float32
	WeightDecay   float32 // base weight decay (Muon)

	initialLRs []float32
	microStep  int
}

// NewTrainer builds a Trainer from the model and optimizer groups.
func NewTrainer(transformer *model.Transformer, groups []optimizer.ParamGroup, gradAccumSteps int) *Trainer {
	trainer := &Trainer{
		Model:          transformer,
		Optim:          optimizer.NewMuonAdamW(groups),
		GradAccumSteps: gradAccumSteps,
		WarmupSteps:    40,
		WarmdownRatio:  0.65,
		FinalLRFrac:    0.05,
		initialLRs:     make([]float32, len(groups)),
	}
	for index := range groups {
		trainer.initialLRs[index] = groups[index].LR
	}
	return trainer
}

// TrainStep runs one forward/backward micro-batch and returns the mean loss.
func (trainer *Trainer) TrainStep(inputs, targets *tensors.Int32s) float32 {
	logits, trainContext := trainer.Model.TrainForward(inputs)
	flat := logits.Reshape(inputs.Numel(), trainer.Model.Config.VocabSize)
	targetsFlat := targets.Reshape(inputs.Numel())

	loss := tensors.CrossEntropy(flat, targetsFlat, -1)
	_, validTokens := tensors.CrossEntropyPerPosition(flat, targetsFlat, -1)
	if validTokens == 0 {
		validTokens = 1
	}
	scale := 1.0 / (float32(validTokens) * float32(trainer.GradAccumSteps))
	gradLogits := tensors.CrossEntropyGrad(flat, targetsFlat, -1, scale)
	gradLogits = gradLogits.Reshape(inputs.Shape[0], inputs.Shape[1], trainer.Model.Config.VocabSize)
	trainer.Model.TrainBackward(trainContext, gradLogits)
	return loss
}

// StepOptimizer applies the current schedules and steps the optimizer. step is
// the global training step. Call it after every GradAccumSteps micro-batches.
func (trainer *Trainer) StepOptimizer(step, numIterations int) {
	lrMultiplier := LRMultiplier(step, numIterations, trainer.WarmupSteps, trainer.WarmdownRatio, trainer.FinalLRFrac)
	momentum := MuonMomentum(step, numIterations, trainer.WarmdownRatio)
	weightDecay := WeightDecayCosine(step, numIterations, trainer.WeightDecay)

	for index := range trainer.Optim.Groups {
		group := &trainer.Optim.Groups[index]
		group.LR = trainer.initialLRs[index] * lrMultiplier
		switch group.Kind {
		case optimizer.KindMuon:
			group.Momentum = momentum
			group.WeightDecay = weightDecay
		case optimizer.KindSinkhorn:
			// The Sinkhorn-balanced update uses the Muon momentum schedule but
			// applies no weight decay (DeepSeek-V4.1 §2.5).
			group.Momentum = momentum
			group.WeightDecay = 0
		}
	}
	trainer.Optim.Step()
	trainer.Optim.ZeroGrad()
}
