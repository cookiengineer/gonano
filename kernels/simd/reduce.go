package simdbackend

import "simd"

// Sum returns the sum of every element using SIMD accumulation with a single
// horizontal fold at the end.
func (backend *Backend) Sum(values []float32) float32 {
	return sumCore(values)
}

// sumCore holds the vectorized body of Sum, for the same compiler reason as
// addCore in elementwise.go.
func sumCore(values []float32) float32 {
	length := len(values)
	accumulator := simd.BroadcastFloat32s(0)
	index := 0
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		accumulator = accumulator.Add(simd.LoadFloat32s(values[index:]))
	}
	total := horizontalSum(accumulator)
	for ; index < length; index++ {
		total += values[index]
	}
	return total
}

// Max returns the largest element, or zero for an empty slice. Slices shorter
// than one vector fall back to a scalar scan.
func (backend *Backend) Max(values []float32) float32 {
	return maxCore(values)
}

// maxCore holds the vectorized body of Max, for the same compiler reason as
// addCore in elementwise.go.
func maxCore(values []float32) float32 {
	length := len(values)
	if length == 0 {
		return 0
	}
	if length < float32LaneCount {
		return maxScalar(values)
	}
	accumulator := simd.LoadFloat32s(values)
	index := float32LaneCount
	for ; index+float32LaneCount <= length; index += float32LaneCount {
		accumulator = accumulator.Max(simd.LoadFloat32s(values[index:]))
	}
	maximum := horizontalMax(accumulator)
	for ; index < length; index++ {
		if values[index] > maximum {
			maximum = values[index]
		}
	}
	return maximum
}

// ArgMax returns the index of the largest element, resolving ties to the
// smallest index. An empty slice returns zero.
func (backend *Backend) ArgMax(values []float32) int {
	bestIndex := 0
	for index := 1; index < len(values); index++ {
		if values[index] > values[bestIndex] {
			bestIndex = index
		}
	}
	return bestIndex
}
