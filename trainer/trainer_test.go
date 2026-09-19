package trainer

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

func TestLRMultiplier(tester *testing.T) {
	// warmup 10, warmdown ratio 0.5, numIterations 100.
	warmup := LRMultiplier(0, 100, 10, 0.5, 0.05)
	if math.Abs(float64(warmup-0.1)) > 1e-5 {
		tester.Fatalf("warmup[0] = %v, want 0.1", warmup)
	}
	if mid := LRMultiplier(50, 100, 10, 0.5, 0.05); mid != 1.0 {
		tester.Fatalf("mid = %v, want 1.0", mid)
	}
	if end := LRMultiplier(100, 100, 10, 0.5, 0.05); math.Abs(float64(end-0.05)) > 1e-5 {
		tester.Fatalf("end = %v, want 0.05", end)
	}
}

func TestMuonMomentum(tester *testing.T) {
	start := MuonMomentum(0, 1000, 0.5)
	if math.Abs(float64(start-0.85)) > 1e-5 {
		tester.Fatalf("momentum[0] = %v, want 0.85", start)
	}
	if mid := MuonMomentum(500, 1000, 0.5); math.Abs(float64(mid-0.97)) > 1e-5 {
		tester.Fatalf("momentum[500] = %v, want 0.97", mid)
	}
}

func TestWeightDecayCosine(tester *testing.T) {
	if weightDecay := WeightDecayCosine(0, 100, 0.3); math.Abs(float64(weightDecay-0.3)) > 1e-5 {
		tester.Fatalf("wd[0] = %v, want 0.3", weightDecay)
	}
	if weightDecay := WeightDecayCosine(100, 100, 0.3); math.Abs(float64(weightDecay)) > 1e-5 {
		tester.Fatalf("wd[end] = %v, want ~0", weightDecay)
	}
}

func TestDeriveHyperparams(tester *testing.T) {
	hyperparams := DeriveHyperparams(20, 32000, 12, 0.28)
	if hyperparams.TotalBatchSize <= 0 {
		tester.Fatalf("batch size = %d, want > 0", hyperparams.TotalBatchSize)
	}
	if hyperparams.NumIterations <= 0 {
		tester.Fatalf("iterations = %d, want > 0", hyperparams.NumIterations)
	}
	if hyperparams.WeightDecay <= 0 {
		tester.Fatalf("weight decay = %v, want > 0", hyperparams.WeightDecay)
	}
	if hyperparams.BatchLRScale <= 0 {
		tester.Fatalf("batch lr scale = %v, want > 0", hyperparams.BatchLRScale)
	}
}

func TestTrainerOverfitsTiny(tester *testing.T) {
	config := model.Config{
		SequenceLen: 8, VocabSize: 16, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(0))
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1)
	trainer := NewTrainer(transformer, groups, 1)
	trainer.WarmupSteps = 2
	trainer.WarmdownRatio = 0.0

	inputs := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	targets := tensors.NewInt32sWithData([]int{1, 8}, []int32{2, 3, 4, 5, 6, 7, 8, 9})

	first := trainer.TrainStep(inputs, targets)
	trainer.StepOptimizer(0, 100)
	var last float32
	for step := 1; step < 100; step++ {
		last = trainer.TrainStep(inputs, targets)
		trainer.StepOptimizer(step, 100)
	}
	if math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
		tester.Fatalf("loss became non-finite: %v", last)
	}
	if last >= first {
		tester.Fatalf("loss did not decrease: %v -> %v", first, last)
	}
}
