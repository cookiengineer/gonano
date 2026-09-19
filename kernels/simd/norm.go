package simdbackend

import (
	"math"

	"github.com/cookiengineer/gonano/internal/parallel"
)

// SoftmaxLastDim computes a numerically stable softmax over the columns of
// every row. source and destination are [rowCount, columnCount]. Rows are
// independent and processed in parallel.
func (backend *Backend) SoftmaxLastDim(destination, source []float32, rowCount, columnCount int) {
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
			base := row * columnCount
			rowSource := source[base : base+columnCount]
			rowDestination := destination[base : base+columnCount]

			maximum := backend.Max(rowSource)
			expShiftedCore(rowDestination, rowSource, maximum)
			var total float64
			for _, exponent := range rowDestination {
				total += float64(exponent)
			}
			backend.Scale(rowDestination, rowDestination, float32(1.0/total))
		}
	})
}

// RMSNormLastDim normalizes every row to unit root-mean-square without affine
// parameters. source and destination are [rowCount, columnCount]. Rows are
// independent and processed in parallel.
func (backend *Backend) RMSNormLastDim(destination, source []float32, rowCount, columnCount int, epsilon float32) {
	parallel.Default().Chunks(0, rowCount, func(rowStart, rowEnd int) {
		for row := rowStart; row < rowEnd; row++ {
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
	})
}
