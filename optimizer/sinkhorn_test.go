package optimizer

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// TestSinkhornSingleRowNormalization checks the K=1 update against a
// hand-computed row normalization: unit row [0.6, 0.8] times sqrt(columns).
func TestSinkhornSingleRowNormalization(t *testing.T) {
	param := tensors.New(1, 2)
	param.Grad = []float32{3, 4}
	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind: KindSinkhorn, Params: []*tensors.Tensor{param},
		LR: 1, Momentum: 0, Gamma: 1, SinkhornSteps: 1, SinkhornTau: 0, SinkhornEps: 1e-20,
	}})
	optimizer.Step()
	want0 := -float32(math.Sqrt(2) * 0.6)
	want1 := -float32(math.Sqrt(2) * 0.8)
	if math.Abs(float64(param.Data[0]-want0)) > 1e-5 || math.Abs(float64(param.Data[1]-want1)) > 1e-5 {
		t.Fatalf("sinkhorn param = %v, want [%v %v]", param.Data, want0, want1)
	}
}

// TestSinkhornMasksNearZeroRows checks that a row whose norm is below tau times
// the mean row norm is masked to zero and left unchanged.
func TestSinkhornMasksNearZeroRows(t *testing.T) {
	param := tensors.New(2, 2)
	param.Grad = []float32{3, 4, 0, 0}
	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind: KindSinkhorn, Params: []*tensors.Tensor{param},
		LR: 1, Momentum: 0, Gamma: 1, SinkhornSteps: 1, SinkhornTau: 0.5, SinkhornEps: 1e-20,
	}})
	optimizer.Step()
	if param.Data[2] != 0 || param.Data[3] != 0 {
		t.Fatalf("near-zero row not masked: %v", param.Data[2:])
	}
	if param.Data[0] == 0 || param.Data[1] == 0 {
		t.Fatalf("active row was not updated")
	}
}

// TestSinkhornRowsHaveUnitRMS verifies the paper's unit-row-RMS property: after
// an odd number of alternating normalizations and the sqrt(n) rescale, every
// non-zero row of the update has RMS one.
func TestSinkhornRowsHaveUnitRMS(t *testing.T) {
	rows, columns := 4, 3
	param := tensors.New(rows, columns)
	param.Grad = []float32{1e-3, 2e-3, 3e-3, 4, 5, 6, 7, 8, 9, 0.1, 0.2, 0.3}
	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind: KindSinkhorn, Params: []*tensors.Tensor{param},
		LR: 1, Momentum: 0, Gamma: 1, SinkhornSteps: 11, SinkhornTau: 0, SinkhornEps: 1e-20,
	}})
	optimizer.Step()
	for row := 0; row < rows; row++ {
		var sum float64
		for column := 0; column < columns; column++ {
			delta := float64(param.Data[row*columns+column])
			sum += delta * delta
		}
		rms := math.Sqrt(sum / float64(columns))
		if math.Abs(rms-1) > 1e-3 {
			t.Fatalf("row %d update RMS = %v, want 1", row, rms)
		}
	}
}

// TestSinkhornStateAllocatedAndFinite runs a full step with the paper defaults
// and checks the state allocation, finiteness, and that the parameter changed.
func TestSinkhornStateAllocatedAndFinite(t *testing.T) {
	param := tensors.New(6, 3)
	tensors.FillNormal(param, tensors.NewRNG(1), 0.1)
	param.EnsureGrad()
	tensors.FillNormal(tensors.NewWithData([]int{6, 3}, param.Grad), tensors.NewRNG(2), 0.1)

	optimizer := NewMuonAdamW([]ParamGroup{{
		Kind: KindSinkhorn, Params: []*tensors.Tensor{param},
		LR: 0.01, Momentum: 0.9, Gamma: 0.18, SinkhornSteps: 11, SinkhornTau: 1e-3, SinkhornEps: 1e-20,
	}})
	before := param.Clone()
	optimizer.Step()

	state, ok := optimizer.sinkhornStates[param]
	if !ok || state.momentum == nil {
		t.Fatal("sinkhorn state not allocated")
	}
	// Only one buffer is allocated, versus AdamW's two.
	if len(state.momentum.Data) != param.Numel() {
		t.Fatalf("momentum size = %d, want %d", len(state.momentum.Data), param.Numel())
	}
	changed := false
	for index := range param.Data {
		if math.IsNaN(float64(param.Data[index])) || math.IsInf(float64(param.Data[index]), 0) {
			t.Fatalf("non-finite param at %d", index)
		}
		if param.Data[index] != before.Data[index] {
			changed = true
		}
	}
	if !changed {
		t.Fatal("sinkhorn did not update the parameter")
	}
	for index, value := range state.momentum.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite momentum at %d", index)
		}
	}
}
