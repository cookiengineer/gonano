package simdbackend

import (
	"math"

	"simd"
)

// Vectorized exp approximation. The Go simd package exposes no transcendental
// functions, so the scalar fallback (math.Exp) dominates flash-attention time.
// exp(x) is computed as 2^(x/ln2) with a degree-8 polynomial for 2^f on
// f in (-1, 1) and a bit-constructed power-of-two scale. The relative error is
// below 1e-6, which is well within fp32 softmax tolerance.
//
// The clamp keeps the exponent inside the int32 exponent range so the
// power-of-two scale never overflows.
const expLog2e = 1.4426950408889634

// expCoefficients are the Taylor coefficients of 2^f = exp(f*ln2), order 0..8.
var expCoefficients = [9]float32{
	1.0,
	0.6931471805599453,
	0.2402265069591007,
	0.05550410866482158,
	0.009618129107628477,
	0.0013333558146428443,
	0.00015403530393381606,
	1.525273380405984e-05,
	1.3215486790144309e-06,
}

// expVector approximates exp for every lane of x. Broadcasts are done inside
// the function because package-level simd vectors trigger a compiler ICE.
func expVector(x simd.Float32s) simd.Float32s {
	x = x.Max(simd.BroadcastFloat32s(-87.0)).Min(simd.BroadcastFloat32s(88.0))
	t := x.Mul(simd.BroadcastFloat32s(float32(expLog2e)))
	n := t.ConvertToInt32()
	f := t.Sub(n.ConvertToFloat32())

	polynomial := simd.BroadcastFloat32s(expCoefficients[8])
	for index := 7; index >= 0; index-- {
		polynomial = polynomial.MulAdd(f, simd.BroadcastFloat32s(expCoefficients[index]))
	}

	scale := n.Add(simd.BroadcastInt32s(127)).ShiftAllLeft(23).ToBits().BitsToFloat32()
	return polynomial.Mul(scale)
}

// expCore computes destination[i] = exp(source[i]) with SIMD vectorization.
func expCore(destination, source []float32) {
	length := min(len(destination), len(source))
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		expVector(simd.LoadFloat32s(source[index:])).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = float32(math.Exp(float64(source[index])))
	}
}

// expShiftedCore computes destination[i] = exp(source[i] - shift). It is the
// softmax primitive: the caller has already computed the row maximum.
func expShiftedCore(destination, source []float32, shift float32) {
	length := min(len(destination), len(source))
	shiftVector := simd.BroadcastFloat32s(shift)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		values := simd.LoadFloat32s(source[index:]).Sub(shiftVector)
		expVector(values).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = float32(math.Exp(float64(source[index] - shift)))
	}
}

// sigmoidCore computes destination[i] = 1 / (1 + exp(-source[i])).
func sigmoidCore(destination, source []float32) {
	length := min(len(destination), len(source))
	one := simd.BroadcastFloat32s(1)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		exponent := expVector(simd.LoadFloat32s(source[index:]).Neg())
		one.Div(one.Add(exponent)).Store(destination[index:])
	}
	for ; index < length; index++ {
		destination[index] = float32(1.0 / (1.0 + math.Exp(float64(-source[index]))))
	}
}
