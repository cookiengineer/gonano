package simdbackend

import (
	"math"

	"github.com/cookiengineer/gonano/kernels"
)

// maskedAttentionScore replaces disallowed attention positions before softmax.
// It is a large negative finite value so that exp() underflows to zero.
const maskedAttentionScore = -1e30

// Flash attention tiling. Query rows and key columns are processed in blocks so
// the score matrix is never materialized; the running softmax statistics
// (maximum and sum) keep the result exact.
const (
	queryBlockSize = 16
	keyBlockSize   = 64
)

// AttentionForward computes Output = softmax(mask(Query @ Key^T)) @ Value for a
// single (batch, head) with a streaming (flash) softmax. Only a
// [queryBlockSize, keyBlockSize] score tile and a [queryBlockSize, headDim]
// accumulator are live, so peak memory is independent of sequence length.
func (backend *Backend) AttentionForward(parameters kernels.AttentionForwardParameters, result kernels.AttentionForwardResult) {
	queryLength := parameters.QueryLength
	keyLength := parameters.KeyLength
	headDim := parameters.HeadDim

	scoreTile := make([]float32, queryBlockSize*keyBlockSize)
	for queryStart := 0; queryStart < queryLength; queryStart += queryBlockSize {
		queryEnd := min(queryStart+queryBlockSize, queryLength)
		blockRows := queryEnd - queryStart

		runningMaximum := make([]float32, blockRows)
		runningSum := make([]float32, blockRows)
		accumulator := make([]float32, blockRows*headDim)
		for row := 0; row < blockRows; row++ {
			runningMaximum[row] = float32(math.Inf(-1))
		}

		for keyStart := 0; keyStart < keyLength; keyStart += keyBlockSize {
			keyEnd := min(keyStart+keyBlockSize, keyLength)
			blockColumns := keyEnd - keyStart
			computeMaskedScoreTile(scoreTile, parameters, queryStart, keyStart, blockRows, blockColumns)

			for row := 0; row < blockRows; row++ {
				rowScores := scoreTile[row*keyBlockSize : row*keyBlockSize+blockColumns]
				blockMaximum := maxCore(rowScores)
				previousMaximum := runningMaximum[row]
				newMaximum := previousMaximum
				if blockMaximum > newMaximum {
					newMaximum = blockMaximum
				}

				// Rescale the running statistics and accumulator to the new
				// maximum. exp(-Inf) is zero, so the first block starts clean.
				rescale := float32(math.Exp(float64(previousMaximum - newMaximum)))
				runningSum[row] *= rescale
				accumulatorRow := accumulator[row*headDim : (row+1)*headDim]
				scaleCore(accumulatorRow, accumulatorRow, rescale)

				var blockSum float64
				for column := 0; column < blockColumns; column++ {
					probability := float32(math.Exp(float64(rowScores[column] - newMaximum)))
					rowScores[column] = probability
					blockSum += float64(probability)
				}
				runningSum[row] += float32(blockSum)

				for column := 0; column < blockColumns; column++ {
					probability := rowScores[column]
					if probability == 0 {
						continue
					}
					valueRow := parameters.Value[(keyStart+column)*headDim : (keyStart+column+1)*headDim]
					addScaledCore(accumulatorRow, accumulatorRow, valueRow, probability)
				}
				runningMaximum[row] = newMaximum
			}
		}

		for row := 0; row < blockRows; row++ {
			accumulatorRow := accumulator[row*headDim : (row+1)*headDim]
			scaleCore(accumulatorRow, accumulatorRow, float32(1.0/runningSum[row]))
			copy(result.Output[(queryStart+row)*headDim:], accumulatorRow)
			if result.LogSumExp != nil {
				result.LogSumExp[queryStart+row] = runningMaximum[row] + float32(math.Log(float64(runningSum[row])))
			}
		}
	}
}

// computeMaskedScoreTile fills the [blockRows, keyBlockSize] score tile with
// Query @ Key^T values, replacing disallowed positions with the mask constant.
func computeMaskedScoreTile(scoreTile []float32, parameters kernels.AttentionForwardParameters, queryStart, keyStart, blockRows, blockColumns int) {
	queryLength := parameters.QueryLength
	keyLength := parameters.KeyLength
	headDim := parameters.HeadDim
	for row := 0; row < blockRows; row++ {
		queryIndex := queryStart + row
		if queryIndex >= queryLength {
			break
		}
		queryPosition := parameters.PositionOffset + queryIndex
		queryRow := parameters.Query[queryIndex*headDim : (queryIndex+1)*headDim]
		for column := 0; column < blockColumns; column++ {
			keyIndex := keyStart + column
			if keyIndex >= keyLength {
				break
			}
			score := dotProduct(queryRow, parameters.Key[keyIndex*headDim:(keyIndex+1)*headDim], headDim)
			if keyIndex > queryPosition || (parameters.Window >= 0 && queryPosition-keyIndex > parameters.Window) {
				score = maskedAttentionScore
			}
			scoreTile[row*keyBlockSize+column] = score
		}
	}
}

// AttentionBackward computes the query/key/value gradients from the saved
// forward statistics using a streaming recomputation of the attention
// probabilities. It processes one query row at a time, so no [queryLength,
// keyLength] matrix is ever allocated. Gradients are accumulated into the
// result slices, which is required for grouped-query attention where several
// query heads share one key/value head.
func (backend *Backend) AttentionBackward(parameters kernels.AttentionBackwardParameters, result kernels.AttentionBackwardResult) {
	queryLength := parameters.QueryLength
	keyLength := parameters.KeyLength
	headDim := parameters.HeadDim

	scores := make([]float32, keyLength)
	probabilities := make([]float32, keyLength)
	outputGradientProbability := make([]float32, keyLength)
	scoreGradient := make([]float32, keyLength)
	queryGradientRow := make([]float32, headDim)

	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		queryPosition := parameters.PositionOffset + queryIndex
		queryRow := parameters.Query[queryIndex*headDim : (queryIndex+1)*headDim]
		outputGradientRow := parameters.OutputGradient[queryIndex*headDim : (queryIndex+1)*headDim]

		for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
			score := dotProduct(queryRow, parameters.Key[keyIndex*headDim:(keyIndex+1)*headDim], headDim)
			if keyIndex > queryPosition || (parameters.Window >= 0 && queryPosition-keyIndex > parameters.Window) {
				score = maskedAttentionScore
			}
			scores[keyIndex] = score
		}

		logSumExp := parameters.LogSumExp[queryIndex]
		var rowDotProduct float64
		for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
			probability := float32(0)
			if scores[keyIndex] > maskedAttentionScore {
				probability = float32(math.Exp(float64(scores[keyIndex] - logSumExp)))
			}
			probabilities[keyIndex] = probability
			gradientProbability := dotProduct(outputGradientRow, parameters.Value[keyIndex*headDim:(keyIndex+1)*headDim], headDim)
			outputGradientProbability[keyIndex] = gradientProbability
			rowDotProduct += float64(gradientProbability) * float64(probability)
		}

		for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
			probability := float64(probabilities[keyIndex])
			scoreGradient[keyIndex] = float32(probability * (float64(outputGradientProbability[keyIndex]) - rowDotProduct))
		}

		clear(queryGradientRow)
		for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
			probability := probabilities[keyIndex]
			gradientScore := scoreGradient[keyIndex]
			if probability == 0 && gradientScore == 0 {
				continue
			}
			keyRow := parameters.Key[keyIndex*headDim : (keyIndex+1)*headDim]

			// dQuery += dS * Key; dKey += dS * Query; dValue += P * dOutput.
			addScaledCore(queryGradientRow, queryGradientRow, keyRow, gradientScore)
			keyGradientRow := result.KeyGradient[keyIndex*headDim : (keyIndex+1)*headDim]
			addScaledCore(keyGradientRow, keyGradientRow, queryRow, gradientScore)
			if probability != 0 {
				valueGradientRow := result.ValueGradient[keyIndex*headDim : (keyIndex+1)*headDim]
				addScaledCore(valueGradientRow, valueGradientRow, outputGradientRow, probability)
			}
		}

		queryGradientSlice := result.QueryGradient[queryIndex*headDim : (queryIndex+1)*headDim]
		for dimension := 0; dimension < headDim; dimension++ {
			queryGradientSlice[dimension] += queryGradientRow[dimension]
		}
	}
}
