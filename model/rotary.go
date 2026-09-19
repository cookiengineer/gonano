package model

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// rotaryBase is the RoPE frequency base used by nanochat.
const rotaryBase = 100000

// precomputeRotary returns cosine and sine tables of shape
// [sequenceLength, headDimension/2] for the given head dimension. Positions
// stride over half the head dimension, as in the standard RoPE formulation.
func precomputeRotary(sequenceLength, headDimension int) (cosine, sine *tensors.Tensor) {
	halfDimension := headDimension / 2
	cosine = tensors.New(sequenceLength, halfDimension)
	sine = tensors.New(sequenceLength, halfDimension)
	for position := 0; position < sequenceLength; position++ {
		for frequencyIndex := 0; frequencyIndex < halfDimension; frequencyIndex++ {
			// inv_freq[j] = base^(-2j/headDim), then freq = position * inv_freq.
			inverseFrequency := math.Pow(rotaryBase, -2.0*float64(frequencyIndex)/float64(headDimension))
			frequency := float64(position) * inverseFrequency
			cosine.Set2(position, frequencyIndex, float32(math.Cos(frequency)))
			sine.Set2(position, frequencyIndex, float32(math.Sin(frequency)))
		}
	}
	return cosine, sine
}

// ApplyRotary rotates the last dimension of activations (shape [B,T,H,D]) using
// RoPE. cosine/sine have shape [seqLen, D/2]; positionOffset offsets the
// position of the first token (nonzero during KV-cache decode). The result is a
// new tensors.
//
// For each token at position p = positionOffset+sequenceIndex, the first and
// second halves of the last dimension are rotated pairwise:
// y1 = x1*cos + x2*sin, y2 = -x1*sin + x2*cos.
func ApplyRotary(activations, cosine, sine *tensors.Tensor, positionOffset int) *tensors.Tensor {
	batchSize, sequenceLength, headCount, headDimension := activations.Shape[0], activations.Shape[1], activations.Shape[2], activations.Shape[3]
	halfDimension := headDimension / 2
	output := tensors.New(batchSize, sequenceLength, headCount, headDimension)
	inputData, outputData := activations.Data, output.Data
	cosineData, sineData := cosine.Data, sine.Data
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for sequenceIndex := 0; sequenceIndex < sequenceLength; sequenceIndex++ {
			position := positionOffset + sequenceIndex
			cosineWindow := cosineData[position*halfDimension : (position+1)*halfDimension]
			sineWindow := sineData[position*halfDimension : (position+1)*halfDimension]
			for headIndex := 0; headIndex < headCount; headIndex++ {
				base := ((batchIndex*sequenceLength+sequenceIndex)*headCount + headIndex) * headDimension
				for dimensionIndex := 0; dimensionIndex < halfDimension; dimensionIndex++ {
					firstHalfValue := inputData[base+dimensionIndex]
					secondHalfValue := inputData[base+halfDimension+dimensionIndex]
					outputData[base+dimensionIndex] = firstHalfValue*cosineWindow[dimensionIndex] + secondHalfValue*sineWindow[dimensionIndex]
					outputData[base+halfDimension+dimensionIndex] = -firstHalfValue*sineWindow[dimensionIndex] + secondHalfValue*cosineWindow[dimensionIndex]
				}
			}
		}
	}
	return output
}
