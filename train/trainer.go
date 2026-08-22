package train

import (
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optim"
	"github.com/cookiengineer/gonano/tensor"
)

// Trainer drives a single training run: forward/backward over micro-batches
// with gradient accumulation, then optimizer steps with LR/momentum/weight-
// decay scheduling.
type Trainer struct {
	Model          *model.Transformer
	Optim          *optim.MuonAdamW
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
func NewTrainer(m *model.Transformer, groups []optim.ParamGroup, gradAccumSteps int) *Trainer {
	t := &Trainer{
		Model:          m,
		Optim:          optim.NewMuonAdamW(groups),
		GradAccumSteps: gradAccumSteps,
		WarmupSteps:    40,
		WarmdownRatio:  0.65,
		FinalLRFrac:    0.05,
		initialLRs:     make([]float32, len(groups)),
	}
	for i := range groups {
		t.initialLRs[i] = groups[i].LR
	}
	return t
}

// TrainStep runs one forward/backward micro-batch and returns the mean loss.
func (t *Trainer) TrainStep(x, y *tensor.Int32s) float32 {
	logits, ctx := t.Model.TrainForward(x)
	flat := logits.Reshape(x.Numel(), t.Model.Config.VocabSize)
	tflat := y.Reshape(x.Numel())

	loss := tensor.CrossEntropy(flat, tflat, -1)
	_, valid := tensor.CrossEntropyPerPosition(flat, tflat, -1)
	if valid == 0 {
		valid = 1
	}
	scale := 1.0 / (float32(valid) * float32(t.GradAccumSteps))
	gradLogits := tensor.CrossEntropyGrad(flat, tflat, -1, scale)
	gradLogits = gradLogits.Reshape(x.Shape[0], x.Shape[1], t.Model.Config.VocabSize)
	t.Model.TrainBackward(ctx, gradLogits)
	return loss
}

// StepOptimizer applies the current schedules and steps the optimizer. step is
// the global training step. Call it after every GradAccumSteps micro-batches.
func (t *Trainer) StepOptimizer(step, numIterations int) {
	lrm := LRMultiplier(step, numIterations, t.WarmupSteps, t.WarmdownRatio, t.FinalLRFrac)
	momentum := MuonMomentum(step, numIterations, t.WarmdownRatio)
	wd := WeightDecayCosine(step, numIterations, t.WeightDecay)

	for i := range t.Optim.Groups {
		g := &t.Optim.Groups[i]
		g.LR = t.initialLRs[i] * lrm
		if g.Kind == optim.KindMuon {
			g.Momentum = momentum
			g.WeightDecay = wd
		}
	}
	t.Optim.Step()
	t.Optim.ZeroGrad()
}
