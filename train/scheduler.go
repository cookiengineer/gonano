package train

import "math"

// Learning-rate, momentum, and weight-decay schedules, matching nanochat's
// base_train.py.

// LRMultiplier returns the learning-rate multiplier at step: linear warmup,
// constant, then linear warmdown to finalLRFrac.
func LRMultiplier(step, numIterations, warmupSteps int, warmdownRatio, finalLRFrac float32) float32 {
	warmdownIters := int(float32(numIterations) * warmdownRatio)
	if step < warmupSteps {
		return float32(step+1) / float32(warmupSteps)
	}
	if step <= numIterations-warmdownIters {
		return 1.0
	}
	progress := float32(numIterations-step) / float32(warmdownIters)
	return progress*1.0 + (1-progress)*finalLRFrac
}

// MuonMomentum returns the Muon momentum at step: 0.85 -> 0.97 over the first
// 400 steps, then 0.97 -> 0.90 during warmdown.
func MuonMomentum(step, numIterations int, warmdownRatio float32) float32 {
	warmdownIters := int(float32(numIterations) * warmdownRatio)
	warmdownStart := numIterations - warmdownIters
	if step < 400 {
		frac := float32(step) / 400
		return (1-frac)*0.85 + frac*0.97
	}
	if step >= warmdownStart {
		progress := float32(step-warmdownStart) / float32(warmdownIters)
		return 0.97*(1-progress) + 0.90*progress
	}
	return 0.97
}

// WeightDecayCosine returns the Muon weight decay at step (cosine decay to 0).
func WeightDecayCosine(step, numIterations int, base float32) float32 {
	return base * 0.5 * float32(1+math.Cos(math.Pi*float64(step)/float64(numIterations)))
}
