package model

import "github.com/cookiengineer/gonano/tensors"

// smearGateChannels is the number of leading embedding channels the smear
// gate reads (nanochat hardcodes 24).
const smearGateChannels = 24

// sliceChannels returns the first `channels` channels of activations (shape
// [B,T,C]) for the given row range [rowStart, rowEnd), as a contiguous
// [B, rows, channels] tensors. It is used to feed the smear and value-embedding
// gates, which read only a small prefix of each token's embedding.
func sliceChannels(activations *tensors.Tensor, rowStart, rowEnd, channels int) *tensors.Tensor {
	batchSize, sequenceLength, embeddingDimension := activations.Shape[0], activations.Shape[1], activations.Shape[2]
	rowCount := rowEnd - rowStart
	output := tensors.New(batchSize, rowCount, channels)
	inputData, outputData := activations.Data, output.Data
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for rowIndex := rowStart; rowIndex < rowEnd; rowIndex++ {
			source := inputData[(batchIndex*sequenceLength+rowIndex)*embeddingDimension : (batchIndex*sequenceLength+rowIndex)*embeddingDimension+channels]
			copy(outputData[(batchIndex*rowCount+(rowIndex-rowStart))*channels:], source)
		}
	}
	return output
}

// smearAdd mixes the previous token's embedding into positions 1..T-1 in
// place: activations[:, 1:, :] += gate[:, :, 0] * activations[:, :-1, :],
// where gate has shape [B, T-1, 1] (one scalar per position).
func smearAdd(activations, gate *tensors.Tensor) {
	batchSize, sequenceLength, embeddingDimension := activations.Shape[0], activations.Shape[1], activations.Shape[2]
	activationData, gateData := activations.Data, gate.Data
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for rowIndex := 1; rowIndex < sequenceLength; rowIndex++ {
			gateValue := gateData[batchIndex*(sequenceLength-1)+(rowIndex-1)]
			base := (batchIndex*sequenceLength + rowIndex) * embeddingDimension
			previousBase := (batchIndex*sequenceLength + (rowIndex - 1)) * embeddingDimension
			for channelIndex := 0; channelIndex < embeddingDimension; channelIndex++ {
				activationData[base+channelIndex] += gateValue * activationData[previousBase+channelIndex]
			}
		}
	}
}

// hasValueEmbedding reports whether a layer at layerIndex has a value
// embedding. Alternating layers have one, and the last layer always does.
func hasValueEmbedding(layerIndex, numLayers int) bool {
	return layerIndex%2 == (numLayers-1)%2
}
