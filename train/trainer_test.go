package train

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
)

func TestLRMultiplier(t *testing.T) {
	// warmup 10, warmdown ratio 0.5, numIterations 100.
	warmup := LRMultiplier(0, 100, 10, 0.5, 0.05)
	if math.Abs(float64(warmup-0.1)) > 1e-5 {
		t.Fatalf("warmup[0] = %v, want 0.1", warmup)
	}
	if mid := LRMultiplier(50, 100, 10, 0.5, 0.05); mid != 1.0 {
		t.Fatalf("mid = %v, want 1.0", mid)
	}
	if end := LRMultiplier(100, 100, 10, 0.5, 0.05); math.Abs(float64(end-0.05)) > 1e-5 {
		t.Fatalf("end = %v, want 0.05", end)
	}
}

func TestMuonMomentum(t *testing.T) {
	start := MuonMomentum(0, 1000, 0.5)
	if math.Abs(float64(start-0.85)) > 1e-5 {
		t.Fatalf("momentum[0] = %v, want 0.85", start)
	}
	if mid := MuonMomentum(500, 1000, 0.5); math.Abs(float64(mid-0.97)) > 1e-5 {
		t.Fatalf("momentum[500] = %v, want 0.97", mid)
	}
}

func TestWeightDecayCosine(t *testing.T) {
	if wd := WeightDecayCosine(0, 100, 0.3); math.Abs(float64(wd-0.3)) > 1e-5 {
		t.Fatalf("wd[0] = %v, want 0.3", wd)
	}
	if wd := WeightDecayCosine(100, 100, 0.3); math.Abs(float64(wd)) > 1e-5 {
		t.Fatalf("wd[end] = %v, want ~0", wd)
	}
}

func TestDeriveHyperparams(t *testing.T) {
	hp := DeriveHyperparams(20, 32000, 12, 0.28)
	if hp.TotalBatchSize <= 0 {
		t.Fatalf("batch size = %d, want > 0", hp.TotalBatchSize)
	}
	if hp.NumIterations <= 0 {
		t.Fatalf("iterations = %d, want > 0", hp.NumIterations)
	}
	if hp.WeightDecay <= 0 {
		t.Fatalf("weight decay = %v, want > 0", hp.WeightDecay)
	}
	if hp.BatchLRScale <= 0 {
		t.Fatalf("batch lr scale = %v, want > 0", hp.BatchLRScale)
	}
}

func TestTrainerOverfitsTiny(t *testing.T) {
	cfg := model.Config{
		SequenceLen: 8, VocabSize: 16, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(0))
	groups := m.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1)
	tr := NewTrainer(m, groups, 1)
	tr.WarmupSteps = 2
	tr.WarmdownRatio = 0.0

	x := tensor.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	y := tensor.NewInt32sWithData([]int{1, 8}, []int32{2, 3, 4, 5, 6, 7, 8, 9})

	first := tr.TrainStep(x, y)
	tr.StepOptimizer(0, 100)
	var last float32
	for step := 1; step < 100; step++ {
		last = tr.TrainStep(x, y)
		tr.StepOptimizer(step, 100)
	}
	if math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
		t.Fatalf("loss became non-finite: %v", last)
	}
	if last >= first {
		t.Fatalf("loss did not decrease: %v -> %v", first, last)
	}
}
