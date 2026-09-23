package trainer

import "math"

// EffortPenaltyConfig parameterizes the exponential reasoning-length penalty
// used to condition a model on a scalar effort level b in [1,100]
// (DeepSeek-V4.1 §5.1.4).
type EffortPenaltyConfig struct {
	// K0 is the basic token-penalty coefficient at the minimum effort level.
	K0 float32
	// Lambda controls the rate of penalty decay; tau = Lambda * MeanDeltaEffort.
	Lambda float32
	// Cap is the maximum deduction applied to one trajectory (C_max).
	Cap float32
	// Norm is the reference reasoning length (L_norm).
	Norm float32
	// MinEffort is b_min, the smallest training effort level.
	MinEffort int
	// MeanDeltaEffort is the average spacing between consecutive training
	// effort levels.
	MeanDeltaEffort float32
}

// EffortPenaltyCoefficient returns k(b) = K0 * exp(-(b - b_min) / tau) with
// tau = Lambda * MeanDeltaEffort. A non-positive tau disables the decay.
func EffortPenaltyCoefficient(effort int, config EffortPenaltyConfig) float32 {
	tau := config.Lambda * config.MeanDeltaEffort
	if tau <= 0 {
		return config.K0
	}
	delta := float64(effort-config.MinEffort) / float64(tau)
	return config.K0 * float32(math.Exp(-delta))
}

// ExponentialTokenPenalty returns the length-penalty reward term
//
//	r_len = -min(Cap, k(b) * length / Norm)
//
// for a completion of the given reasoning-token length at effort level b. It
// returns zero when Norm or Cap is non-positive.
func ExponentialTokenPenalty(effort, length int, config EffortPenaltyConfig) float32 {
	if config.Norm <= 0 || config.Cap <= 0 || length <= 0 {
		return 0
	}
	penalty := EffortPenaltyCoefficient(effort, config) * float32(length) / config.Norm
	if penalty > config.Cap {
		penalty = config.Cap
	}
	return -penalty
}
