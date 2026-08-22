package nn

import (
	"math"

	"github.com/cookiengineer/gonano/tensor"
)

// Weight initializers matching nanochat's init_weights exactly.

// InitNormal fills w with independent N(0, std) samples.
func InitNormal(w *tensor.Tensor, rng *tensor.RNG, std float32) {
	tensor.FillNormal(w, rng, std)
}

// InitUniform fills w with independent uniform samples in [lo, hi].
func InitUniform(w *tensor.Tensor, rng *tensor.RNG, lo, hi float32) {
	tensor.FillUniform(w, rng, lo, hi)
}

// InitZeros fills w with zeros.
func InitZeros(w *tensor.Tensor) {
	tensor.FillZeros(w)
}

// InitValue fills w with a constant.
func InitValue(w *tensor.Tensor, v float32) {
	w.Set(v)
}

// stdToUniformBound converts a desired standard deviation to the half-width of
// a uniform distribution with the same standard deviation. The uniform
// distribution on [-b, b] has std = b / sqrt(3), so b = sqrt(3) * std.
func stdToUniformBound(std float64) float64 {
	return math.Sqrt(3.0) * std
}
