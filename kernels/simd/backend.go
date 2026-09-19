// Package simd implements the kernels.Backend contract using the experimental
// Go simd standard-library package. It is the production backend for gonano:
// vectorized over float32 lanes and parallelized across CPU cores through the
// parallel package.
//
// The scalar backend in kernels/scalar is the reference used to verify every
// method here.
package simdbackend

import "simd"

// maximumFloat32Lanes is the largest possible float32 lane count: a 512-bit
// vector holds 16 float32 values. The simd package caps vector width at 512
// bits, so this is an upper bound used to size fixed reduction scratch buffers
// without heap allocation.
const maximumFloat32Lanes = 16

// float32LaneCount is the number of float32 lanes per SIMD vector on this CPU.
var float32LaneCount = simd.VectorBitSize() / 32

// Backend is the SIMD implementation of kernels.Backend.
type Backend struct{}

// New returns a new SIMD backend.
func New() *Backend { return &Backend{} }

// horizontalSum reduces a SIMD vector to a scalar by summing every lane.
func horizontalSum(vector simd.Float32s) float32 {
	if float32LaneCount <= maximumFloat32Lanes {
		var buffer [maximumFloat32Lanes]float32
		vector.Store(buffer[:float32LaneCount])
		return sumScalar(buffer[:float32LaneCount])
	}
	buffer := make([]float32, float32LaneCount)
	vector.Store(buffer)
	return sumScalar(buffer)
}

// horizontalMax reduces a SIMD vector to a scalar by taking the largest lane.
func horizontalMax(vector simd.Float32s) float32 {
	if float32LaneCount <= maximumFloat32Lanes {
		var buffer [maximumFloat32Lanes]float32
		vector.Store(buffer[:float32LaneCount])
		return maxScalar(buffer[:float32LaneCount])
	}
	buffer := make([]float32, float32LaneCount)
	vector.Store(buffer)
	return maxScalar(buffer)
}

// sumScalar returns the sum of a small slice, used for SIMD tails.
func sumScalar(values []float32) float32 {
	var total float32
	for _, value := range values {
		total += value
	}
	return total
}

// maxScalar returns the largest element of a small slice, used for SIMD tails.
func maxScalar(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	maximum := values[0]
	for _, value := range values[1:] {
		if value > maximum {
			maximum = value
		}
	}
	return maximum
}
