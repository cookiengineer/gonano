package trainer

import (
	"math"
	"testing"
)

func TestEffortPenaltyCoefficientDecreasesWithEffort(t *testing.T) {
	config := EffortPenaltyConfig{K0: 1, Lambda: 1, MinEffort: 50, MeanDeltaEffort: 25}
	low := EffortPenaltyCoefficient(50, config)
	mid := EffortPenaltyCoefficient(75, config)
	high := EffortPenaltyCoefficient(100, config)
	if !(low > mid && mid > high) {
		t.Fatalf("penalty coefficient must decrease with effort: %v %v %v", low, mid, high)
	}
	// tau = 25, so increasing effort by 25 multiplies the coefficient by e^-1.
	if math.Abs(float64(mid/low)-math.Exp(-1)) > 1e-4 {
		t.Fatalf("coefficient ratio = %v, want e^-1", mid/low)
	}
}

func TestExponentialTokenPenalty(t *testing.T) {
	config := EffortPenaltyConfig{K0: 1, Lambda: 1, MinEffort: 50, MeanDeltaEffort: 25, Cap: 0.5, Norm: 100}
	short := ExponentialTokenPenalty(50, 10, config)
	long := ExponentialTokenPenalty(50, 90, config)
	if !(short > long && long < 0) {
		t.Fatalf("penalty must be negative and decrease with length: short=%v long=%v", short, long)
	}
	// The cap saturates the deduction at -Cap.
	capped := ExponentialTokenPenalty(50, 100000, config)
	if capped != -0.5 {
		t.Fatalf("capped penalty = %v, want -0.5", capped)
	}
	// Higher effort reduces the penalty for the same length.
	if ExponentialTokenPenalty(100, 50, config) <= ExponentialTokenPenalty(50, 50, config) {
		t.Fatal("higher effort must reduce the length penalty")
	}
	if ExponentialTokenPenalty(50, 0, config) != 0 {
		t.Fatal("zero-length penalty must be zero")
	}
}
