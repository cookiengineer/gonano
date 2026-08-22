package tensor

import (
	"simd"
)

// maxFloat32Lanes is the largest possible float32 lane count: a 512-bit vector
// holds 16 float32 values. The simd package currently caps vector width at
// 512 bits, so this is an upper bound used to allocate fixed-size reduction
// scratch buffers without heap allocation.
const maxFloat32Lanes = 16

// float32Lanes is the number of float32 lanes per SIMD vector on this CPU.
var float32Lanes = simd.VectorBitSize() / 32

// horizontalSum reduces a SIMD vector to a scalar by summing all lanes.
func horizontalSum(v simd.Float32s) float32 {
	if float32Lanes <= maxFloat32Lanes {
		var buf [maxFloat32Lanes]float32
		v.Store(buf[:float32Lanes])
		return sumScalar(buf[:float32Lanes])
	}
	buf := make([]float32, float32Lanes)
	v.Store(buf)
	return sumScalar(buf)
}

// horizontalMax reduces a SIMD vector to a scalar by taking the maximum lane.
func horizontalMax(v simd.Float32s) float32 {
	if float32Lanes <= maxFloat32Lanes {
		var buf [maxFloat32Lanes]float32
		v.Store(buf[:float32Lanes])
		return maxScalar(buf[:float32Lanes])
	}
	buf := make([]float32, float32Lanes)
	v.Store(buf)
	return maxScalar(buf)
}

func sumScalar(s []float32) float32 {
	sum := s[0]
	for i := 1; i < len(s); i++ {
		sum += s[i]
	}
	return sum
}

func maxScalar(s []float32) float32 {
	m := s[0]
	for i := 1; i < len(s); i++ {
		if s[i] > m {
			m = s[i]
		}
	}
	return m
}
