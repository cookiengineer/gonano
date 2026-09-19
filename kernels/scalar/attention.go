package scalar

import (
	"math"

	"github.com/cookiengineer/gonano/kernels"
)

// maskedAttentionScore replaces disallowed attention positions before softmax.
// It is a large negative finite value so that exp() underflows to zero.
const maskedAttentionScore = -1e30

// AttentionForward computes Output = softmax(mask(Query @ Key^T)) @ Value for a
// single (batch, head). This implementation materializes the score matrix and
// serves as the accurate reference for the SIMD flash-attention kernel.
func (backend *Backend) AttentionForward(parameters kernels.AttentionForwardParameters, result kernels.AttentionForwardResult) {
	queryLength := parameters.QueryLength
	keyLength := parameters.KeyLength
	headDim := parameters.HeadDim

	scores := make([]float32, queryLength*keyLength)
	probabilities := make([]float32, queryLength*keyLength)
	backend.MatMulTransposed(scores, parameters.Query, parameters.Key, queryLength, keyLength, headDim)
	applyAttentionMask(scores, queryLength, keyLength, parameters.PositionOffset, parameters.Window)

	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		rowScores := scores[queryIndex*keyLength : (queryIndex+1)*keyLength]
		rowProbabilities := probabilities[queryIndex*keyLength : (queryIndex+1)*keyLength]

		maximum := backend.Max(rowScores)
		var total float64
		for keyIndex, score := range rowScores {
			exponent := math.Exp(float64(score - maximum))
			rowProbabilities[keyIndex] = float32(exponent)
			total += exponent
		}
		if result.LogSumExp != nil {
			result.LogSumExp[queryIndex] = maximum + float32(math.Log(total))
		}
		inverseTotal := float32(1.0 / total)
		for dim := 0; dim < headDim; dim++ {
			var accumulator float64
			for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
				accumulator += float64(rowProbabilities[keyIndex]) * float64(parameters.Value[keyIndex*headDim+dim])
			}
			result.Output[queryIndex*headDim+dim] = float32(accumulator) * inverseTotal
		}
	}
}

// AttentionBackward computes the query/key/value gradients from the saved
// forward statistics. Gradients are accumulated into the result slices.
func (backend *Backend) AttentionBackward(parameters kernels.AttentionBackwardParameters, result kernels.AttentionBackwardResult) {
	queryLength := parameters.QueryLength
	keyLength := parameters.KeyLength
	headDim := parameters.HeadDim

	scores := make([]float32, queryLength*keyLength)
	probabilities := make([]float32, queryLength*keyLength)
	backend.MatMulTransposed(scores, parameters.Query, parameters.Key, queryLength, keyLength, headDim)
	applyAttentionMask(scores, queryLength, keyLength, parameters.PositionOffset, parameters.Window)

	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		logSumExp := parameters.LogSumExp[queryIndex]
		for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
			score := scores[queryIndex*keyLength+keyIndex]
			if score <= maskedAttentionScore {
				probabilities[queryIndex*keyLength+keyIndex] = 0
				continue
			}
			probabilities[queryIndex*keyLength+keyIndex] = float32(math.Exp(float64(score - logSumExp)))
		}
	}

	// Gradient of the value: ValueGradient += probabilities^T @ OutputGradient.
	valueGradient := make([]float32, keyLength*headDim)
	transposedProbabilities := make([]float32, keyLength*queryLength)
	transposeMatrix(transposedProbabilities, probabilities, queryLength, keyLength)
	backend.MatMul(valueGradient, transposedProbabilities, parameters.OutputGradient, keyLength, headDim, queryLength)
	accumulate(result.ValueGradient, valueGradient)

	// Gradient of the scores: dS = P * (dP - rowSum(dP*P)).
	outputGradientProbability := make([]float32, queryLength*keyLength)
	backend.MatMulTransposed(outputGradientProbability, parameters.OutputGradient, parameters.Value, queryLength, keyLength, headDim)
	scoreGradient := make([]float32, queryLength*keyLength)
	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		base := queryIndex * keyLength
		var rowDotProduct float64
		for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
			rowDotProduct += float64(outputGradientProbability[base+keyIndex]) * float64(probabilities[base+keyIndex])
		}
		for keyIndex := 0; keyIndex < keyLength; keyIndex++ {
			probability := float64(probabilities[base+keyIndex])
			scoreGradient[base+keyIndex] = float32(probability * (float64(outputGradientProbability[base+keyIndex]) - rowDotProduct))
		}
	}

	// QueryGradient += scoreGradient @ Key.
	queryGradient := make([]float32, queryLength*headDim)
	backend.MatMul(queryGradient, scoreGradient, parameters.Key, queryLength, headDim, keyLength)
	accumulate(result.QueryGradient, queryGradient)

	// KeyGradient += scoreGradient^T @ Query.
	keyGradient := make([]float32, keyLength*headDim)
	transposedScoreGradient := make([]float32, keyLength*queryLength)
	transposeMatrix(transposedScoreGradient, scoreGradient, queryLength, keyLength)
	backend.MatMul(keyGradient, transposedScoreGradient, parameters.Query, keyLength, headDim, queryLength)
	accumulate(result.KeyGradient, keyGradient)
}

// applyAttentionMask marks disallowed positions with maskedAttentionScore.
func applyAttentionMask(scores []float32, queryLength, keyLength, positionOffset, window int) {
	for queryIndex := 0; queryIndex < queryLength; queryIndex++ {
		queryPosition := positionOffset + queryIndex
		rowScores := scores[queryIndex*keyLength : (queryIndex+1)*keyLength]
		for keyIndex := range rowScores {
			if keyIndex > queryPosition || (window >= 0 && queryPosition-keyIndex > window) {
				rowScores[keyIndex] = maskedAttentionScore
			}
		}
	}
}

// transposeMatrix writes the transpose of source [rows, columns] into
// destination [columns, rows].
func transposeMatrix(destination, source []float32, rows, columns int) {
	for row := 0; row < rows; row++ {
		for column := 0; column < columns; column++ {
			destination[column*rows+row] = source[row*columns+column]
		}
	}
}

// accumulate adds source into destination element by element.
func accumulate(destination, source []float32) {
	for index, value := range source {
		destination[index] += value
	}
}
