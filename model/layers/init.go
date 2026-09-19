package layers

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// Weight initializers matching nanochat's init_weights exactly.

// InitNormal fills weight with independent N(0, std) samples.
func InitNormal(weight *tensors.Tensor, rng *tensors.RNG, std float32) {
	tensors.FillNormal(weight, rng, std)
}

// InitUniform fills weight with independent uniform samples in [lo, hi].
func InitUniform(weight *tensors.Tensor, rng *tensors.RNG, lo, hi float32) {
	tensors.FillUniform(weight, rng, lo, hi)
}

// InitZeros fills weight with zeros.
func InitZeros(weight *tensors.Tensor) {
	tensors.FillZeros(weight)
}

// InitValue fills weight with a constant.
func InitValue(weight *tensors.Tensor, value float32) {
	weight.Set(value)
}

// stdToUniformBound converts a desired standard deviation to the half-width of
// a uniform distribution with the same standard deviation. The uniform
// distribution on [-bound, bound] has std = bound / sqrt(3), so
// bound = sqrt(3) * std.
func stdToUniformBound(std float64) float64 {
	return math.Sqrt(3.0) * std
}
