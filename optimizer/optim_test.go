package optimizer

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestAdamWExactValue(test *testing.T) {
	param := tensors.NewWithData([]int{1}, []float32{1})
	param.Grad = []float32{1}
	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind:   KindAdamW,
		Params: []*tensors.Tensor{param},
		LR:     0.01,
		Beta1:  0.9,
		Beta2:  0.999,
		Eps:    1e-8,
	}})
	optimizer.Step()
	// Hand-computed: param = 1 - (0.01/0.1) * 0.1 / 1 = 0.99.
	want := float32(0.99)
	if math.Abs(float64(param.Data[0]-want)) > 1e-5 {
		test.Fatalf("adamw param = %v, want %v", param.Data[0], want)
	}
}

func TestAdamWReducesQuadraticLoss(test *testing.T) {
	param := tensors.New(100)
	for index := range param.Data {
		param.Data[index] = 1
	}
	// loss = ||param||^2, grad = 2*param.
	setQuadGrad(param)
	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind: KindAdamW, Params: []*tensors.Tensor{param},
		LR: 0.05, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
	}})
	before := computeNormSq(param)
	for index := 0; index < 10; index++ {
		setQuadGrad(param)
		optimizer.Step()
	}
	after := computeNormSq(param)
	if after >= before {
		test.Fatalf("adamw did not reduce loss: %v -> %v", before, after)
	}
}

func setQuadGrad(param *tensors.Tensor) {
	param.EnsureGrad()
	for index := range param.Data {
		param.Grad[index] = 2 * param.Data[index]
	}
}

func computeNormSq(param *tensors.Tensor) float32 {
	var sum float64
	for _, value := range param.Data {
		sum += float64(value) * float64(value)
	}
	return float32(sum)
}

func TestMuonStepFiniteAndUpdatesState(test *testing.T) {
	rng := tensors.NewRNG(1)
	param := tensors.New(6, 3) // tall matrix
	tensors.FillNormal(param, rng, 0.1)
	param.EnsureGrad()
	tensors.FillNormal(tensors.NewWithData([]int{6, 3}, param.Grad), rng, 0.1)

	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind: KindMuon, Params: []*tensors.Tensor{param},
		LR: 0.01, Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9,
	}})
	before := param.Clone()
	optimizer.Step()

	state := optimizer.muonStates[param]
	for index, value := range param.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			test.Fatalf("muon produced non-finite param at %d", index)
		}
		if param.Data[index] == before.Data[index] {
			test.Fatalf("param did not change at %d", index)
		}
	}
	if state.momentum == nil || state.secondMom == nil {
		test.Fatal("muon state not allocated")
	}
	for _, value := range state.secondMom.Data {
		if value <= 0 || math.IsNaN(float64(value)) {
			test.Fatalf("second moment = %v, want positive finite", value)
		}
	}
}

func TestMuonWideMatrix(test *testing.T) {
	rng := tensors.NewRNG(2)
	param := tensors.New(3, 8) // wide matrix (rows < columns)
	tensors.FillNormal(param, rng, 0.1)
	param.EnsureGrad()
	tensors.FillNormal(tensors.NewWithData([]int{3, 8}, param.Grad), rng, 0.1)

	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind: KindMuon, Params: []*tensors.Tensor{param},
		LR: 0.01, Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9,
	}})
	optimizer.Step()
	for _, value := range param.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			test.Fatalf("wide muon produced non-finite param")
		}
	}
	if len(optimizer.muonStates[param].secondMom.Data) != 8 {
		test.Fatalf("wide second moment shape = %d, want 8 (per column)", len(optimizer.muonStates[param].secondMom.Data))
	}
}

func TestMuonAdamWRouting(test *testing.T) {
	matrixParam := tensors.New(4, 4) // matrix -> Muon
	vectorParam := tensors.New(4)    // vector -> AdamW
	for index := range matrixParam.Data {
		matrixParam.Data[index] = 0.5
	}
	for index := range vectorParam.Data {
		vectorParam.Data[index] = 0.5
	}
	matrixParam.Grad = make([]float32, matrixParam.Numel())
	vectorParam.Grad = make([]float32, vectorParam.Numel())
	for index := range matrixParam.Grad {
		matrixParam.Grad[index] = 1
	}
	for index := range vectorParam.Grad {
		vectorParam.Grad[index] = 1
	}

	optimizer := NewMuonAdamW([]ParamGroup{
		{Kind: KindMuon, Params: []*tensors.Tensor{matrixParam}, LR: 0.01, Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9},
		{Kind: KindAdamW, Params: []*tensors.Tensor{vectorParam}, LR: 0.01, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8},
	})
	optimizer.Step()
	// Both must have been updated and have state.
	if _, ok := optimizer.muonStates[matrixParam]; !ok {
		test.Fatal("matrix param should be routed to Muon")
	}
	if _, ok := optimizer.adamStates[vectorParam]; !ok {
		test.Fatal("vector param should be routed to AdamW")
	}
}

func TestZeroGrad(test *testing.T) {
	param := tensors.New(3)
	param.Grad = []float32{1, 2, 3}
	optimizer := NewMuonAdamW([]ParamGroup{{Kind: KindAdamW, Params: []*tensors.Tensor{param}, LR: 0.01, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}})
	optimizer.ZeroGrad()
	for _, value := range param.Grad {
		if value != 0 {
			test.Fatalf("grad = %v, want 0", value)
		}
	}
}
