// Package optimizer implements the optimizers used by gonano: fused AdamW and
// the Muon orthogonalized-momentum optimizer, combined into a MuonAdamW that
// routes parameters to the appropriate optimizer by kind.
//
// This is the single-process (non-distributed) implementation: gradient
// accumulation and parameter updates run entirely in memory, with the heavy
// Muon matrix operations parallelized by the tensors package.
package optimizer

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// Kind selects which optimizer a parameter group uses.
type Kind int

const (
	// KindAdamW routes a group to the AdamW optimizer (scalars, normalization
	// weights, and other non-matrix parameters).
	KindAdamW Kind = iota
	// KindMuon routes a group to the Muon optimizer (2D matrix parameters).
	KindMuon
	// KindSinkhorn routes a group to the Sinkhorn-balanced momentum update
	// (DeepSeek-V4.1 §2.5, Algorithm 1), used for the large embedding tables
	// and the prediction head.
	KindSinkhorn
)

// ParamGroup describes a group of parameters that share optimizer
// hyperparameters.
type ParamGroup struct {
	Kind   Kind
	Params []*tensors.Tensor

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

	// Sinkhorn hyperparameters (DeepSeek-V4.1 §2.5, Algorithm 1). Gamma is the
	// learning-rate correction factor, SinkhornSteps the odd number of
	// alternating normalizations, SinkhornTau the near-zero row-mask threshold,
	// and SinkhornEps the normalization epsilon.
	Gamma         float32
	SinkhornSteps int
	SinkhornTau   float32
	SinkhornEps   float32
}

// adamwState is the per-parameter optimizer state for AdamW.
type adamwState struct {
	expAvg   []float32
	expAvgSq []float32
	step     int
}

// muonState is the per-parameter optimizer state for Muon.
type muonState struct {
	momentum  *tensors.Tensor
	secondMom *tensors.Tensor // factored: [rows] or [columns] depending on reduction dim
}

// MuonAdamW is the combined Muon + AdamW optimizer.
type MuonAdamW struct {
	Groups         []ParamGroup
	step           int
	adamStates     map[*tensors.Tensor]*adamwState
	muonStates     map[*tensors.Tensor]*muonState
	sinkhornStates map[*tensors.Tensor]*sinkhornState
}

// NewMuonAdamW builds the optimizer with the given parameter groups.
func NewMuonAdamW(groups []ParamGroup) *MuonAdamW {
	optimizer := &MuonAdamW{
		Groups:         groups,
		adamStates:     make(map[*tensors.Tensor]*adamwState),
		muonStates:     make(map[*tensors.Tensor]*muonState),
		sinkhornStates: make(map[*tensors.Tensor]*sinkhornState),
	}
	for _, group := range groups {
		for _, param := range group.Params {
			switch group.Kind {
			case KindAdamW:
				optimizer.adamStates[param] = &adamwState{expAvg: make([]float32, param.Numel()), expAvgSq: make([]float32, param.Numel())}
			case KindMuon:
				rows, columns := param.Shape[0], param.Shape[1]
				// The second-moment buffer is factored along the reduction dim:
				// per-row when rows >= columns, per-column otherwise.
				reduceDim := rows
				if rows < columns {
					reduceDim = columns
				}
				optimizer.muonStates[param] = &muonState{
					momentum:  tensors.New(param.Shape...),
					secondMom: tensors.New(reduceDim),
				}
			case KindSinkhorn:
				optimizer.sinkhornStates[param] = &sinkhornState{
					momentum: tensors.New(param.Shape...),
				}
			}
		}
	}
	return optimizer
}

// Step performs one optimizer step, consuming the accumulated gradients in
// each parameter's Grad buffer.
func (optimizer *MuonAdamW) Step() {
	optimizer.step++
	for _, group := range optimizer.Groups {
		switch group.Kind {
		case KindAdamW:
			for _, param := range group.Params {
				optimizer.adamwStep(group, param)
			}
		case KindMuon:
			for _, param := range group.Params {
				optimizer.muonStep(group, param)
			}
		case KindSinkhorn:
			for _, param := range group.Params {
				optimizer.sinkhornStep(group, param)
			}
		}
	}
}

// ZeroGrad zeroes the gradients of all parameters.
func (optimizer *MuonAdamW) ZeroGrad() {
	for _, group := range optimizer.Groups {
		for _, param := range group.Params {
			param.ZeroGrad()
		}
	}
}

// adamwStep applies one fused AdamW update to param using its Grad.
func (optimizer *MuonAdamW) adamwStep(group ParamGroup, param *tensors.Tensor) {
	state := optimizer.adamStates[param]
	state.step++
	param.EnsureGrad()
	beta1 := float64(group.Beta1)
	beta2 := float64(group.Beta2)
	eps := float64(group.Eps)
	weightDecay := float64(group.WeightDecay)
	learningRate := float64(group.LR)

	bias1 := 1 - math.Pow(beta1, float64(state.step))
	bias2 := 1 - math.Pow(beta2, float64(state.step))
	decay := 1 - learningRate*weightDecay
	stepSize := learningRate / bias1

	paramData := param.Data
	gradData := param.Grad
	expAvg := state.expAvg
	expAvgSq := state.expAvgSq
	for index := range paramData {
		paramData[index] *= float32(decay)
		expAvg[index] = float32(beta1*float64(expAvg[index]) + (1-beta1)*float64(gradData[index]))
		expAvgSq[index] = float32(beta2*float64(expAvgSq[index]) + (1-beta2)*float64(gradData[index])*float64(gradData[index]))
		denom := math.Sqrt(float64(expAvgSq[index])/bias2) + eps
		paramData[index] -= float32(stepSize * float64(expAvg[index]) / denom)
	}
}
