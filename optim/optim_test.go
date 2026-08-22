package optim

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensor"
)

func TestAdamWExactValue(t *testing.T) {
	p := tensor.NewWithData([]int{1}, []float32{1})
	p.Grad = []float32{1}
	o := NewMuonAdamW([]ParamGroup{{
		Kind:    KindAdamW,
		Params:  []*tensor.Tensor{p},
		LR:      0.01,
		Beta1:   0.9,
		Beta2:   0.999,
		Eps:     1e-8,
	}})
	o.Step()
	// Hand-computed: p = 1 - (0.01/0.1) * 0.1 / 1 = 0.99.
	want := float32(0.99)
	if math.Abs(float64(p.Data[0]-want)) > 1e-5 {
		t.Fatalf("adamw p = %v, want %v", p.Data[0], want)
	}
}

func TestAdamWReducesQuadraticLoss(t *testing.T) {
	p := tensor.New(100)
	for i := range p.Data {
		p.Data[i] = 1
	}
	// loss = ||p||^2, grad = 2*p.
	setQuadGrad(p)
	o := NewMuonAdamW([]ParamGroup{{
		Kind: KindAdamW, Params: []*tensor.Tensor{p},
		LR: 0.05, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8,
	}})
	before := normSq(p)
	for i := 0; i < 10; i++ {
		setQuadGrad(p)
		o.Step()
	}
	after := normSq(p)
	if after >= before {
		t.Fatalf("adamw did not reduce loss: %v -> %v", before, after)
	}
}

func setQuadGrad(p *tensor.Tensor) {
	p.EnsureGrad()
	for i := range p.Data {
		p.Grad[i] = 2 * p.Data[i]
	}
}

func normSq(p *tensor.Tensor) float32 {
	var s float64
	for _, v := range p.Data {
		s += float64(v) * float64(v)
	}
	return float32(s)
}

func TestMuonStepFiniteAndUpdatesState(t *testing.T) {
	rng := tensor.NewRNG(1)
	p := tensor.New(6, 3) // tall matrix
	tensor.FillNormal(p, rng, 0.1)
	p.EnsureGrad()
	tensor.FillNormal(tensor.NewWithData([]int{6, 3}, p.Grad), rng, 0.1)

	o := NewMuonAdamW([]ParamGroup{{
		Kind: KindMuon, Params: []*tensor.Tensor{p},
		LR: 0.01, Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9,
	}})
	before := p.Clone()
	o.Step()

	st := o.muonStates[p]
	for i, v := range p.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("muon produced non-finite param at %d", i)
		}
		if p.Data[i] == before.Data[i] {
			t.Fatalf("param did not change at %d", i)
		}
	}
	if st.momentum == nil || st.secondMom == nil {
		t.Fatal("muon state not allocated")
	}
	for _, v := range st.secondMom.Data {
		if v <= 0 || math.IsNaN(float64(v)) {
			t.Fatalf("second moment = %v, want positive finite", v)
		}
	}
}

func TestMuonWideMatrix(t *testing.T) {
	rng := tensor.NewRNG(2)
	p := tensor.New(3, 8) // wide matrix (m < n)
	tensor.FillNormal(p, rng, 0.1)
	p.EnsureGrad()
	tensor.FillNormal(tensor.NewWithData([]int{3, 8}, p.Grad), rng, 0.1)

	o := NewMuonAdamW([]ParamGroup{{
		Kind: KindMuon, Params: []*tensor.Tensor{p},
		LR: 0.01, Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9,
	}})
	o.Step()
	for _, v := range p.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("wide muon produced non-finite param")
		}
	}
	if len(o.muonStates[p].secondMom.Data) != 8 {
		t.Fatalf("wide second moment shape = %d, want 8 (per column)", len(o.muonStates[p].secondMom.Data))
	}
}

func TestMuonAdamWRouting(t *testing.T) {
	mp := tensor.New(4, 4) // matrix -> Muon
	sp := tensor.New(4)    // vector -> AdamW
	for i := range mp.Data {
		mp.Data[i] = 0.5
	}
	for i := range sp.Data {
		sp.Data[i] = 0.5
	}
	mp.Grad = make([]float32, mp.Numel())
	sp.Grad = make([]float32, sp.Numel())
	for i := range mp.Grad {
		mp.Grad[i] = 1
	}
	for i := range sp.Grad {
		sp.Grad[i] = 1
	}

	o := NewMuonAdamW([]ParamGroup{
		{Kind: KindMuon, Params: []*tensor.Tensor{mp}, LR: 0.01, Momentum: 0.95, NSSteps: 5, MuonBeta2: 0.9},
		{Kind: KindAdamW, Params: []*tensor.Tensor{sp}, LR: 0.01, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8},
	})
	o.Step()
	// Both must have been updated and have state.
	if _, ok := o.muonStates[mp]; !ok {
		t.Fatal("matrix param should be routed to Muon")
	}
	if _, ok := o.adamStates[sp]; !ok {
		t.Fatal("vector param should be routed to AdamW")
	}
}

func TestZeroGrad(t *testing.T) {
	p := tensor.New(3)
	p.Grad = []float32{1, 2, 3}
	o := NewMuonAdamW([]ParamGroup{{Kind: KindAdamW, Params: []*tensor.Tensor{p}, LR: 0.01, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}})
	o.ZeroGrad()
	for _, v := range p.Grad {
		if v != 0 {
			t.Fatalf("grad = %v, want 0", v)
		}
	}
}
