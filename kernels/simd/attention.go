package simdbackend

import (
	"math"

	"simd"

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
	// scoreTileGemmMinRows is the query-block size at which transposing the key
	// tile (needed by the rank-1-update score GEMM) is amortized. Below it the
	// direct dot-product path is faster.
	scoreTileGemmMinRows = 8
)

// AttentionForward computes Output = softmax(mask(Query @ Key^T)) @ Value for a
// single (batch, head) with a streaming (flash) softmax. Only a
// [queryBlockSize, keyBlockSize] score tile and a [queryBlockSize, headDim]
// accumulator are live, so peak memory is independent of sequence length.
func (backend *Backend) AttentionForward(parameters kernels.AttentionForwardParameters, result kernels.AttentionForwardResult) {
	queryLength := parameters.QueryLength
	headDim := parameters.HeadDim

	maximum := make([]float32, queryLength)
	for row := range maximum {
		maximum[row] = float32(math.Inf(-1))
	}
	total := make([]float32, queryLength)
	accumulator := make([]float32, queryLength*headDim)

	attentionForwardRange(parameters.Query, parameters.Key, parameters.Value, queryLength, headDim,
		parameters.PositionOffset, parameters.Window, 0, parameters.KeyLength, maximum, total, accumulator)

	for row := 0; row < queryLength; row++ {
		accumulatorRow := accumulator[row*headDim : (row+1)*headDim]
		scaleCore(accumulatorRow, accumulatorRow, float32(1.0/total[row]))
		copy(result.Output[row*headDim:], accumulatorRow)
		if result.LogSumExp != nil {
			result.LogSumExp[row] = maximum[row] + float32(math.Log(float64(total[row])))
		}
	}
}

// AttentionForwardSplit computes the unnormalized output and softmax
// statistics for the key shard [KeyStart, KeyEnd) of a (batch, head). The
// result slices must be pre-sized to [QueryLength, HeadDim] (accumulator) and
// [QueryLength] (maximum, sum); they are overwritten.
func (backend *Backend) AttentionForwardSplit(parameters kernels.AttentionSplitParameters, result kernels.AttentionSplitResult) {
	queryLength := parameters.QueryLength
	headDim := parameters.HeadDim
	for row := 0; row < queryLength; row++ {
		result.Maximum[row] = float32(math.Inf(-1))
	}
	clear(result.Sum)
	clear(result.Accumulator)
	attentionForwardRange(parameters.Query, parameters.Key, parameters.Value, queryLength, headDim,
		parameters.PositionOffset, parameters.Window, parameters.KeyStart, parameters.KeyEnd,
		result.Maximum, result.Sum, result.Accumulator)
}

// AttentionCombine merges per-shard statistics into the normalized output and
// optional log-sum-exp, using the standard flash-attention reduction.
func (backend *Backend) AttentionCombine(parameters kernels.AttentionCombineParameters, result kernels.AttentionForwardResult) {
	queryLength := parameters.QueryLength
	headDim := parameters.HeadDim
	for row := 0; row < queryLength; row++ {
		globalMaximum := float32(math.Inf(-1))
		for _, partial := range parameters.Partials {
			if partial.Maximum[row] > globalMaximum {
				globalMaximum = partial.Maximum[row]
			}
		}
		var total float64
		outputRow := result.Output[row*headDim : (row+1)*headDim]
		clear(outputRow)
		for _, partial := range parameters.Partials {
			scale := float32(0)
			if partial.Maximum[row] > float32(math.Inf(-1)) {
				scale = float32(math.Exp(float64(partial.Maximum[row] - globalMaximum)))
			}
			total += float64(partial.Sum[row]) * float64(scale)
			accumulatorRow := partial.Accumulator[row*headDim : (row+1)*headDim]
			for dimension := 0; dimension < headDim; dimension++ {
				outputRow[dimension] += accumulatorRow[dimension] * scale
			}
		}
		if total > 0 {
			inverse := float32(1.0 / total)
			for dimension := 0; dimension < headDim; dimension++ {
				outputRow[dimension] *= inverse
			}
		}
		if result.LogSumExp != nil {
			result.LogSumExp[row] = globalMaximum + float32(math.Log(total))
		}
	}
}

// attentionForwardRange accumulates the flash-attention statistics for the key
// range [keyStartBound, keyEndBound) into the caller-provided maximum, sum, and
// accumulator. maximum must be initialized to -Inf and sum/accumulator zeroed.
func attentionForwardRange(query, key, value []float32, queryLength, headDim, positionOffset, window, keyStartBound, keyEndBound int, maximum, sum, accumulator []float32) {
	scoreTile := make([]float32, queryBlockSize*keyBlockSize)
	// The key-transpose scratch is only needed by the rank-1 GEMM path, which
	// requires a full query block; single-token decode uses the dot-product
	// path and must not pay the allocation.
	var keyTranspose []float32
	if queryLength >= scoreTileGemmMinRows {
		keyTranspose = make([]float32, headDim*keyBlockSize)
	}
	for queryStart := 0; queryStart < queryLength; queryStart += queryBlockSize {
		queryEnd := min(queryStart+queryBlockSize, queryLength)
		blockRows := queryEnd - queryStart

		for keyStart := keyStartBound; keyStart < keyEndBound; keyStart += keyBlockSize {
			keyEnd := min(keyStart+keyBlockSize, keyEndBound)
			blockColumns := keyEnd - keyStart
			computeMaskedScoreTile(scoreTile, keyTranspose, query, key, queryLength, headDim, positionOffset, window, queryStart, keyStart, blockRows, blockColumns)

			for row := 0; row < blockRows; row++ {
				globalRow := queryStart + row
				rowScores := scoreTile[row*keyBlockSize : row*keyBlockSize+blockColumns]
				blockMaximum := maxCore(rowScores)
				previousMaximum := maximum[globalRow]
				newMaximum := previousMaximum
				if blockMaximum > newMaximum {
					newMaximum = blockMaximum
				}

				// Rescale the running statistics and accumulator to the new
				// maximum. exp(-Inf) is zero, so the first block starts clean.
				rescale := float32(math.Exp(float64(previousMaximum - newMaximum)))
				sum[globalRow] *= rescale
				accumulatorRow := accumulator[globalRow*headDim : (globalRow+1)*headDim]
				scaleCore(accumulatorRow, accumulatorRow, rescale)

				expShiftedCore(rowScores[:blockColumns], rowScores[:blockColumns], newMaximum)
				var blockSum float64
				for _, probability := range rowScores[:blockColumns] {
					blockSum += float64(probability)
				}
				sum[globalRow] += float32(blockSum)

				for column := 0; column < blockColumns; column++ {
					probability := rowScores[column]
					if probability == 0 {
						continue
					}
					valueRow := value[(keyStart+column)*headDim : (keyStart+column+1)*headDim]
					addScaledCore(accumulatorRow, accumulatorRow, valueRow, probability)
				}
				maximum[globalRow] = newMaximum
			}
		}
	}
}

// computeMaskedScoreTile fills the [blockRows, blockColumns] score tile with
// Query @ Key^T values, replacing disallowed positions with the mask constant.
//
// For enough query rows the score tile is computed as a rank-1-update GEMM: the
// key block is transposed to [headDim, blockColumns] so each step broadcasts
// one query element and updates every key column with a vectorized
// multiply-add, avoiding the per-score horizontal reduction. The transpose is
// only worthwhile when it is amortized across several query rows, so for the
// single-row decode case the direct dot-product path is used instead.
// keyTranspose is caller-provided scratch of length headDim*keyBlockSize.
func computeMaskedScoreTile(scoreTile, keyTranspose, query, key []float32, queryLength, headDim, positionOffset, window, queryStart, keyStart, blockRows, blockColumns int) {
	if blockRows < scoreTileGemmMinRows {
		for row := 0; row < blockRows; row++ {
			queryIndex := queryStart + row
			if queryIndex >= queryLength {
				break
			}
			queryPosition := positionOffset + queryIndex
			queryRow := query[queryIndex*headDim : (queryIndex+1)*headDim]
			scoreRow := scoreTile[row*keyBlockSize : row*keyBlockSize+blockColumns]
			for column := 0; column < blockColumns; column++ {
				keyIndex := keyStart + column
				score := dotProduct(queryRow, key[keyIndex*headDim:(keyIndex+1)*headDim], headDim)
				if keyIndex > queryPosition || (window >= 0 && queryPosition-keyIndex > window) {
					score = maskedAttentionScore
				}
				scoreRow[column] = score
			}
		}
		return
	}

	for column := 0; column < blockColumns; column++ {
		keyRow := key[(keyStart+column)*headDim : (keyStart+column)*headDim+headDim]
		for inner := 0; inner < headDim; inner++ {
			keyTranspose[inner*blockColumns+column] = keyRow[inner]
		}
	}

	for row := 0; row < blockRows; row++ {
		queryIndex := queryStart + row
		if queryIndex >= queryLength {
			break
		}
		queryRow := query[queryIndex*headDim : (queryIndex+1)*headDim]
		scoreRow := scoreTile[row*keyBlockSize : row*keyBlockSize+blockColumns]
		clear(scoreRow)
		for inner := 0; inner < headDim; inner++ {
			elementVector := simd.BroadcastFloat32s(queryRow[inner])
			keyRow := keyTranspose[inner*blockColumns : (inner+1)*blockColumns]
			column := 0
			for ; column+float32LaneCount <= blockColumns; column += float32LaneCount {
				accumulator := simd.LoadFloat32s(scoreRow[column:])
				keyVector := simd.LoadFloat32s(keyRow[column:])
				keyVector.MulAdd(elementVector, accumulator).Store(scoreRow[column:])
			}
			for ; column < blockColumns; column++ {
				scoreRow[column] += queryRow[inner] * keyRow[column]
			}
		}

		queryPosition := positionOffset + queryIndex
		for column := 0; column < blockColumns; column++ {
			keyIndex := keyStart + column
			if keyIndex > queryPosition || (window >= 0 && queryPosition-keyIndex > window) {
				scoreRow[column] = maskedAttentionScore
			}
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
