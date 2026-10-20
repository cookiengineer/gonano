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
	return precomputeRotaryScaled(sequenceLength, rotaryDimension, rotaryBase, RopeScalingConfig{})
}

// RopeScalingConfig parameterizes optional YaRN rotary scaling.
type RopeScalingConfig struct {
	Mode                string
	Factor              float64
	OriginalMaxPosition int
	BetaFast            float64
	BetaSlow            float64
	MScale              bool
}

// precomputeRotaryScaled builds RoPE tables for a base frequency and optional
// YaRN scaling.
func precomputeRotaryScaled(sequenceLength, rotaryDimension int, theta float64, scaling RopeScalingConfig) (cosine, sine *tensors.Tensor) {
	halfDimension := rotaryDimension / 2
	cosine = tensors.New(sequenceLength, halfDimension)
	sine = tensors.New(sequenceLength, halfDimension)
	if theta <= 0 {
		theta = rotaryBase
	}
	attentionFactor := float64(1)
	yarn := scaling.Mode == "yarn" && scaling.Factor > 1
	if yarn {
		if scaling.OriginalMaxPosition <= 0 {
			scaling.OriginalMaxPosition = sequenceLength
		}
		if scaling.BetaFast <= 0 {
			scaling.BetaFast = 32
		}
		if scaling.BetaSlow <= 0 {
			scaling.BetaSlow = 1
		}
		if scaling.MScale {
			attentionFactor = 0.1*math.Log(scaling.Factor) + 1
		}
	}
	for position := 0; position < sequenceLength; position++ {
		for frequencyIndex := 0; frequencyIndex < halfDimension; frequencyIndex++ {
			inverseFrequency := math.Pow(theta, -2.0*float64(frequencyIndex)/float64(rotaryDimension))
			if yarn {
				inverseFrequency = yarnInverseFrequency(frequencyIndex, rotaryDimension, theta, scaling)
			}
			frequency := float64(position) * inverseFrequency
			cosine.Set2(position, frequencyIndex, float32(attentionFactor*math.Cos(frequency)))
			sine.Set2(position, frequencyIndex, float32(attentionFactor*math.Sin(frequency)))
		}
	}
	return cosine, sine
}

// yarnInverseFrequency applies the YaRN interpolation/extrapolation blend.
func yarnInverseFrequency(index, dimension int, theta float64, scaling RopeScalingConfig) float64 {
	base := math.Pow(theta, -2.0*float64(index)/float64(dimension))
	if scaling.Factor <= 1 {
		return base
	}
	correctionDim := func(rotations float64) float64 {
		return (float64(dimension) * math.Log(float64(scaling.OriginalMaxPosition)/(rotations*2*math.Pi))) / (2 * math.Log(theta))
	}
	low := math.Floor(correctionDim(scaling.BetaFast))
	high := math.Ceil(correctionDim(scaling.BetaSlow))
	if high <= low {
		high = low + 1
	}
	ramp := (float64(index) - low) / (high - low)
	if ramp < 0 {
		ramp = 0
	}
	if ramp > 1 {
		ramp = 1
	}
	ramp = 1 - ramp
	extrapolation := base
	interpolation := base / scaling.Factor
	return interpolation*(1-ramp) + extrapolation*ramp
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


// ApplyRotaryPositions is ApplyRotary with one logical position per batch row.
// It is used for batched decode, where independent sequences sit at different
// context positions.
func ApplyRotaryPositions(activations, cosine, sine *tensors.Tensor, positions []int32) *tensors.Tensor {
	batchSize, sequenceLength, headCount, headDimension := activations.Shape[0], activations.Shape[1], activations.Shape[2], activations.Shape[3]
	rotaryDimension := cosine.Shape[1] * 2
	if rotaryDimension > headDimension {
		panic("model: rotary table wider than head dimension")
	}
	if len(positions) < batchSize {
		panic("model: rotary positions shorter than batch size")
	}
	halfDimension := rotaryDimension / 2
	offset := headDimension - rotaryDimension

	output := tensors.New(batchSize, sequenceLength, headCount, headDimension)
	inputData, outputData := activations.Data, output.Data
	copy(outputData, inputData)

	cosineData, sineData := cosine.Data, sine.Data
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for sequenceIndex := 0; sequenceIndex < sequenceLength; sequenceIndex++ {
			position := int(positions[batchIndex]) + sequenceIndex
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

// applyRotaryForCache applies rotary embeddings using per-row cache positions
// when a cache is present, and a uniform offset otherwise.
func applyRotaryForCache(activations, cosine, sine *tensors.Tensor, positionOffset int, cache *KVBuffer) *tensors.Tensor {
	if cache != nil {
		return ApplyRotaryPositions(activations, cosine, sine, cache.Positions(activations.Shape[0]))
	}
	return ApplyRotary(activations, cosine, sine, positionOffset)
}
