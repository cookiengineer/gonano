package tensors

import (
	"math"
	"math/rand/v2"
)

// RNG is a deterministic random number generator used for weight
// initialization, sampling, and data shuffling. It wraps math/rand/v2's PCG
// generator, which is a high-quality, zero-dependency source.
type RNG struct {
	source *rand.Rand
}

// NewRNG returns a generator seeded deterministically from seed.
func NewRNG(seed uint64) *RNG {
	return &RNG{source: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))}
}

// Float32 returns a uniform float32 in [0, 1).
func (generator *RNG) Float32() float32 { return generator.source.Float32() }

// Float64 returns a uniform float64 in [0, 1).
func (generator *RNG) Float64() float64 { return generator.source.Float64() }

// NormFloat32 returns a standard normal float32 (mean 0, standard deviation 1).
func (generator *RNG) NormFloat32() float32 { return float32(generator.source.NormFloat64()) }

// Uint64 returns a uniform uint64.
func (generator *RNG) Uint64() uint64 { return generator.source.Uint64() }

// IntN returns a uniform integer in [0, n).
func (generator *RNG) IntN(n int) int { return generator.source.IntN(n) }

// FillNormal fills target with independent N(0, standardDeviation) samples.
func FillNormal(target *Tensor, generator *RNG, standardDeviation float32) {
	for index := range target.Data {
		target.Data[index] = generator.NormFloat32() * standardDeviation
	}
}

// FillUniform fills target with independent uniform samples in [lower, upper).
func FillUniform(target *Tensor, generator *RNG, lower, upper float32) {
	span := upper - lower
	for index := range target.Data {
		target.Data[index] = lower + generator.Float32()*span
	}
}

// FillZeros fills target with zeros.
func FillZeros(target *Tensor) {
	for index := range target.Data {
		target.Data[index] = 0
	}
}

// SampleMultinomial draws a single index from the unnormalized logits using
// the Gumbel-max trick. logits must be rank-1. A temperature of zero or less
// selects the argmax deterministically.
func SampleMultinomial(generator *RNG, logits []float32, temperature float32) int {
	if temperature <= 0 {
		return KernelBackend().ArgMax(logits)
	}
	bestIndex := 0
	bestScore := float32(math.Inf(-1))
	for index, value := range logits {
		// Gumbel(0,1) sample: -log(-log(u)).
		uniform := generator.Float32()
		if uniform <= 0 {
			uniform = 1e-12
		}
		gumbel := -float32(math.Log(-math.Log(float64(uniform))))
		score := value/temperature + gumbel
		if score > bestScore {
			bestScore = score
			bestIndex = index
		}
	}
	return bestIndex
}
