package scalar

// MatMul computes destination = left @ right, where left is
// [rowCount, innerCount] and right is [innerCount, columnCount]. The
// accumulation is performed in float64 so this backend can serve as an
// accurate reference for the SIMD backend.
func (backend *Backend) MatMul(destination, left, right []float32, rowCount, columnCount, innerCount int) {
	for row := 0; row < rowCount; row++ {
		for column := 0; column < columnCount; column++ {
			var accumulator float64
			for inner := 0; inner < innerCount; inner++ {
				accumulator += float64(left[row*innerCount+inner]) * float64(right[inner*columnCount+column])
			}
			destination[row*columnCount+column] = float32(accumulator)
		}
	}
}

// MatMulTransposed computes destination = left @ right^T, where left is
// [rowCount, innerCount] and right is [columnCount, innerCount]. The
// accumulation is performed in float64.
func (backend *Backend) MatMulTransposed(destination, left, right []float32, rowCount, columnCount, innerCount int) {
	for row := 0; row < rowCount; row++ {
		for column := 0; column < columnCount; column++ {
			var accumulator float64
			for inner := 0; inner < innerCount; inner++ {
				accumulator += float64(left[row*innerCount+inner]) * float64(right[column*innerCount+inner])
			}
			destination[row*columnCount+column] = float32(accumulator)
		}
	}
}

// DotProduct returns the sum of left[i]*right[i] over the shorter input.
func (backend *Backend) DotProduct(left, right []float32) float32 {
	length := min(len(left), len(right))
	var accumulator float64
	for index := 0; index < length; index++ {
		accumulator += float64(left[index]) * float64(right[index])
	}
	return float32(accumulator)
}
