package tensor

import (
	"math"
	"math/rand/v2"
)

// RNG is a deterministic random number generator used for weight
// initialization, sampling, and data shuffling. It wraps math/rand/v2's PCG
// generator, which is a high-quality, zero-dependency source.
type RNG struct {
	r *rand.Rand
}

// NewRNG returns a generator seeded deterministically from seed.
func NewRNG(seed uint64) *RNG {
	return &RNG{r: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))}
}

// Float32 returns a uniform float32 in [0, 1).
func (r *RNG) Float32() float32 { return r.r.Float32() }

// Float64 returns a uniform float64 in [0, 1).
func (r *RNG) Float64() float64 { return r.r.Float64() }

// NormFloat32 returns a standard normal float32 (mean 0, std 1).
func (r *RNG) NormFloat32() float32 { return float32(r.r.NormFloat64()) }

// Uint64 returns a uniform uint64.
func (r *RNG) Uint64() uint64 { return r.r.Uint64() }

// IntN returns a uniform integer in [0, n).
func (r *RNG) IntN(n int) int { return r.r.IntN(n) }

// FillNormal fills t with independent N(0, std) samples.
func FillNormal(t *Tensor, rng *RNG, std float32) {
	for i := range t.Data {
		t.Data[i] = rng.NormFloat32() * std
	}
}

// FillUniform fills t with independent uniform samples in [lo, hi).
func FillUniform(t *Tensor, rng *RNG, lo, hi float32) {
	span := hi - lo
	for i := range t.Data {
		t.Data[i] = lo + rng.Float32()*span
	}
}

// FillZeros fills t with zeros.
func FillZeros(t *Tensor) {
	for i := range t.Data {
		t.Data[i] = 0
	}
}

// SampleMultinomial draws a single index from the unnormalized logits using
// the Gumbel-max trick. logits must be a 1D tensor. temperature <= 0 selects
// the argmax deterministically.
func SampleMultinomial(rng *RNG, logits []float32, temperature float32) int {
	if temperature <= 0 {
		return sliceMaxIndex(logits)
	}
	best := 0
	bestScore := float32(math.Inf(-1))
	for i, v := range logits {
		// Gumbel(0,1) sample: -log(-log(u)).
		u := rng.Float32()
		if u <= 0 {
			u = 1e-12
		}
		g := -float32(math.Log(-math.Log(float64(u))))
		score := v/temperature + g
		if score > bestScore {
			bestScore = score
			best = i
		}
	}
	return best
}
