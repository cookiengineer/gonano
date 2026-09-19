package trainer

import "math"

// Learning-rate, momentum, and weight-decay schedules, matching nanochat's
// base_train.py.

// LRMultiplier returns the learning-rate multiplier at step: linear warmup,
// constant, then linear warmdown to finalLRFrac.
func LRMultiplier(step, numIterations, warmupSteps int, warmdownRatio, finalLRFrac float32) float32 {
	warmdownIterations := int(float32(numIterations) * warmdownRatio)
	if step < warmupSteps {
		return float32(step+1) / float32(warmupSteps)
	}
	if step <= numIterations-warmdownIterations {
		return 1.0
	}
	progress := float32(numIterations-step) / float32(warmdownIterations)
	return progress*1.0 + (1-progress)*finalLRFrac
}

// MuonMomentum returns the Muon momentum at step: 0.85 -> 0.97 over the first
// 400 steps, then 0.97 -> 0.90 during warmdown.
func MuonMomentum(step, numIterations int, warmdownRatio float32) float32 {
	warmdownIterations := int(float32(numIterations) * warmdownRatio)
	warmdownStart := numIterations - warmdownIterations
	if step < 400 {
		fraction := float32(step) / 400
		return (1-fraction)*0.85 + fraction*0.97
	}
	if step >= warmdownStart {
		progress := float32(step-warmdownStart) / float32(warmdownIterations)
		return 0.97*(1-progress) + 0.90*progress
	}
	return 0.97
}

// WeightDecayCosine returns the Muon weight decay at step (cosine decay to 0).
func WeightDecayCosine(step, numIterations int, baseWeightDecay float32) float32 {
	return baseWeightDecay * 0.5 * float32(1+math.Cos(math.Pi*float64(step)/float64(numIterations)))
}
