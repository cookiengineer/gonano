package train

import (
	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optim"
)

// TrainSFT runs a supervised fine-tuning loop over the given conversation
// loader. It trains for numIterations steps (or until the loader is exhausted
// if numIterations <= 0) and returns the per-step losses.
func TrainSFT(m *model.Transformer, groups []optim.ParamGroup, loader *data.SFTLoader, numIterations int, evalFn func(step int, loss float32)) []float32 {
	tr := NewTrainer(m, groups, 1)
	tr.WarmupSteps = 0
	tr.WarmdownRatio = 0.5
	tr.FinalLRFrac = 0

	var losses []float32
	for step := 0; ; step++ {
		if numIterations > 0 && step >= numIterations {
			break
		}
		x, y, ok := loader.Next()
		if !ok {
			break
		}
		loss := tr.TrainStep(x, y)
		tr.StepOptimizer(step, max(1, numIterations))
		losses = append(losses, loss)
		if evalFn != nil {
			evalFn(step, loss)
		}
	}
	return losses
}
