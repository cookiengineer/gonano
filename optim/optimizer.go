// Package optim implements the optimizers used by gonano: fused AdamW and the
// Muon orthogonalized-momentum optimizer, combined into a MuonAdamW that
// routes parameters to the appropriate optimizer by kind.
//
// This is the single-process (non-distributed) implementation: gradient
// accumulation and parameter updates run entirely in memory, with the heavy
// Muon matrix operations parallelized by the tensor package.
package optim

import (
	"math"

	"github.com/cookiengineer/gonano/tensor"
)

// Kind selects which optimizer a parameter group uses.
type Kind int

const (
	// KindAdamW routes a group to the AdamW optimizer (embeddings, scalars,
	// the unembedding layer, and other non-matrix parameters).
	KindAdamW Kind = iota
	// KindMuon routes a group to the Muon optimizer (2D matrix parameters).
	KindMuon
)

// ParamGroup describes a group of parameters that share optimizer
// hyperparameters.
type ParamGroup struct {
	Kind   Kind
	Params []*tensor.Tensor

	// AdamW hyperparameters.
	LR          float32
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32

	// Muon hyperparameters.
	Momentum  float32
	NSSteps   int
	MuonBeta2 float32
}

// adamwState is the per-parameter optimizer state for AdamW.
type adamwState struct {
	expAvg   []float32
	expAvgSq []float32
	step     int
}

// muonState is the per-parameter optimizer state for Muon.
type muonState struct {
	momentum  *tensor.Tensor
	secondMom *tensor.Tensor // factored: [m] or [n] depending on reduction dim
}

// MuonAdamW is the combined Muon + AdamW optimizer.
type MuonAdamW struct {
	Groups     []ParamGroup
	step       int
	adamStates map[*tensor.Tensor]*adamwState
	muonStates map[*tensor.Tensor]*muonState
}

// NewMuonAdamW builds the optimizer with the given parameter groups.
func NewMuonAdamW(groups []ParamGroup) *MuonAdamW {
	o := &MuonAdamW{
		Groups:     groups,
		adamStates: make(map[*tensor.Tensor]*adamwState),
		muonStates: make(map[*tensor.Tensor]*muonState),
	}
	for _, g := range groups {
		for _, p := range g.Params {
			switch g.Kind {
			case KindAdamW:
				o.adamStates[p] = &adamwState{expAvg: make([]float32, p.Numel()), expAvgSq: make([]float32, p.Numel())}
			case KindMuon:
				m, n := p.Shape[0], p.Shape[1]
				// The second-moment buffer is factored along the reduction dim:
				// per-row when m >= n, per-column otherwise.
				redDim := m
				if m < n {
					redDim = n
				}
				o.muonStates[p] = &muonState{
					momentum:  tensor.New(p.Shape...),
					secondMom: tensor.New(redDim),
				}
			}
		}
	}
	return o
}

// Step performs one optimizer step, consuming the accumulated gradients in
// each parameter's Grad buffer. It returns the total number of parameters.
func (o *MuonAdamW) Step() {
	o.step++
	for _, g := range o.Groups {
		switch g.Kind {
		case KindAdamW:
			for _, p := range g.Params {
				o.adamwStep(g, p)
			}
		case KindMuon:
			for _, p := range g.Params {
				o.muonStep(g, p)
			}
		}
	}
}

// ZeroGrad zeroes the gradients of all parameters.
func (o *MuonAdamW) ZeroGrad() {
	for _, g := range o.Groups {
		for _, p := range g.Params {
			p.ZeroGrad()
		}
	}
}

// adamwStep applies one fused AdamW update to p using its Grad.
func (o *MuonAdamW) adamwStep(g ParamGroup, p *tensor.Tensor) {
	st := o.adamStates[p]
	st.step++
	p.EnsureGrad()
	beta1 := float64(g.Beta1)
	beta2 := float64(g.Beta2)
	eps := float64(g.Eps)
	wd := float64(g.WeightDecay)
	lr := float64(g.LR)

	bias1 := 1 - math.Pow(beta1, float64(st.step))
	bias2 := 1 - math.Pow(beta2, float64(st.step))
	decay := 1 - lr*wd
	stepSize := lr / bias1

	pd := p.Data
	gd := p.Grad
	ma := st.expAvg
	mv := st.expAvgSq
	for i := range pd {
		pd[i] *= float32(decay)
		ma[i] = float32(beta1*float64(ma[i]) + (1-beta1)*float64(gd[i]))
		mv[i] = float32(beta2*float64(mv[i]) + (1-beta2)*float64(gd[i])*float64(gd[i]))
		denom := math.Sqrt(float64(mv[i])/bias2) + eps
		pd[i] -= float32(stepSize * float64(ma[i]) / denom)
	}
}
