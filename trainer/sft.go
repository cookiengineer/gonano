package trainer

import (
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optimizer"
)

// TrainSFT runs a supervised fine-tuning loop over the given conversation
// loader. It trains for numIterations steps (or until the loader is exhausted
// if numIterations <= 0) and returns the per-step losses.
func TrainSFT(transformer *model.Transformer, groups []optimizer.ParamGroup, loader *data.SFTLoader, numIterations int, evaluateFunc func(step int, loss float32)) []float32 {
	trainer := NewTrainer(transformer, groups, 1)
	trainer.WarmupSteps = 0
	trainer.WarmdownRatio = 0.5
	trainer.FinalLRFrac = 0

	var losses []float32
	for step := 0; ; step++ {
		if numIterations > 0 && step >= numIterations {
			break
		}
		inputs, targets, ok := loader.Next()
		if !ok {
			break
		}
		loss := trainer.TrainStep(inputs, targets)
		trainer.StepOptimizer(step, max(1, numIterations))
		losses = append(losses, loss)
		if evaluateFunc != nil {
			evaluateFunc(step, loss)
		}
	}
	return losses
}
