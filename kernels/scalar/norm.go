package scalar

import "math"

// SoftmaxLastDim computes a numerically stable softmax over the columns of
// every row. source and destination are [rowCount, columnCount].
func (backend *Backend) SoftmaxLastDim(destination, source []float32, rowCount, columnCount int) {
	for row := 0; row < rowCount; row++ {
		base := row * columnCount
		rowSource := source[base : base+columnCount]
		rowDestination := destination[base : base+columnCount]

		maximum := backend.Max(rowSource)
		var total float64
		for column, value := range rowSource {
			exponent := math.Exp(float64(value - maximum))
			rowDestination[column] = float32(exponent)
			total += exponent
		}
		backend.Scale(rowDestination, rowDestination, float32(1.0/total))
	}
}

// RMSNormLastDim normalizes every row to unit root-mean-square without affine
// parameters. source and destination are [rowCount, columnCount].
func (backend *Backend) RMSNormLastDim(destination, source []float32, rowCount, columnCount int, epsilon float32) {
	for row := 0; row < rowCount; row++ {
		base := row * columnCount
		rowSource := source[base : base+columnCount]
		rowDestination := destination[base : base+columnCount]

		var sumSquares float64
		for _, value := range rowSource {
			sumSquares += float64(value) * float64(value)
		}
		meanSquare := float32(sumSquares / float64(columnCount))
		inverseRootMeanSquare := float32(1.0 / math.Sqrt(float64(meanSquare)+float64(epsilon)))
		backend.Scale(rowDestination, rowSource, inverseRootMeanSquare)
	}
}
