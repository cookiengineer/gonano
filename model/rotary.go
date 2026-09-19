package model

import (
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// rotaryBase is the RoPE frequency base used by nanochat.
const rotaryBase = 100000

// precomputeRotary returns cosine and sine tables of shape
// [sequenceLength, rotaryDimension/2] for the given rotary dimension. Positions
// stride over half the rotary dimension, as in the standard RoPE formulation.
// rotaryDimension may be smaller than the head dimension (partial RoPE).
func precomputeRotary(sequenceLength, rotaryDimension int) (cosine, sine *tensors.Tensor) {
	halfDimension := rotaryDimension / 2
	cosine = tensors.New(sequenceLength, halfDimension)
	sine = tensors.New(sequenceLength, halfDimension)
	for position := 0; position < sequenceLength; position++ {
		for frequencyIndex := 0; frequencyIndex < halfDimension; frequencyIndex++ {
			// inv_freq[j] = base^(-2j/rotaryDim), then freq = position * inv_freq.
			inverseFrequency := math.Pow(rotaryBase, -2.0*float64(frequencyIndex)/float64(rotaryDimension))
			frequency := float64(position) * inverseFrequency
			cosine.Set2(position, frequencyIndex, float32(math.Cos(frequency)))
			sine.Set2(position, frequencyIndex, float32(math.Sin(frequency)))
		}
	}
	return cosine, sine
}

// ApplyRotary rotates the trailing dimensions of activations (shape [B,T,H,D])
// using RoPE. cosine/sine have shape [seqLen, R/2], where R is the rotary
// dimension; the last R channels of each head are rotated and the leading
// D-R channels are copied unchanged. A table covering the full head dimension
// (R == D) reproduces the historical full-rotation behavior. positionOffset
// offsets the position of the first token (nonzero during KV-cache decode).
// The result is a new tensors.
//
// For each token at position p = positionOffset+sequenceIndex, the first and
// second halves of the rotated sub-block are rotated pairwise:
// y1 = x1*cos + x2*sin, y2 = -x1*sin + x2*cos.
func ApplyRotary(activations, cosine, sine *tensors.Tensor, positionOffset int) *tensors.Tensor {
	batchSize, sequenceLength, headCount, headDimension := activations.Shape[0], activations.Shape[1], activations.Shape[2], activations.Shape[3]
	rotaryDimension := cosine.Shape[1] * 2
	if rotaryDimension > headDimension {
		panic("model: rotary table wider than head dimension")
	}
	halfDimension := rotaryDimension / 2
	offset := headDimension - rotaryDimension

	output := tensors.New(batchSize, sequenceLength, headCount, headDimension)
	inputData, outputData := activations.Data, output.Data
	// Channels outside the rotary slice are not modified.
	copy(outputData, inputData)

	cosineData, sineData := cosine.Data, sine.Data
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for sequenceIndex := 0; sequenceIndex < sequenceLength; sequenceIndex++ {
			position := positionOffset + sequenceIndex
			cosineWindow := cosineData[position*halfDimension : (position+1)*halfDimension]
			sineWindow := sineData[position*halfDimension : (position+1)*halfDimension]
			for headIndex := 0; headIndex < headCount; headIndex++ {
				base := ((batchIndex*sequenceLength+sequenceIndex)*headCount + headIndex) * headDimension
				for dimensionIndex := 0; dimensionIndex < halfDimension; dimensionIndex++ {
					firstHalfValue := inputData[base+offset+dimensionIndex]
					secondHalfValue := inputData[base+offset+halfDimension+dimensionIndex]
					outputData[base+offset+dimensionIndex] = firstHalfValue*cosineWindow[dimensionIndex] + secondHalfValue*sineWindow[dimensionIndex]
					outputData[base+offset+halfDimension+dimensionIndex] = -firstHalfValue*sineWindow[dimensionIndex] + secondHalfValue*cosineWindow[dimensionIndex]
				}
			}
		}
	}
	return output
}
