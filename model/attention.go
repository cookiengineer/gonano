package model

import (
	"github.com/cookiengineer/gonano/internal/parallel"
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// normEps is the epsilon used in RMSNorm, matching torch's RMSNorm default.
const normEps = 1e-6

// qkScale is nanochat's "sharper attention" scaling, split across Q and K.
const qkScale = 1.2

// veGateChannels is the number of leading embedding channels the
// value-embedding gate reads.
const veGateChannels = 12

// veGateScale multiplies the sigmoid gate so it ranges over (0, 3).
const veGateScale = 3.0

// normalizeLastDim applies stateless RMSNorm over the last dimension of an
// arbitrary-rank tensor.
func normalizeLastDim(input *tensors.Tensor) *tensors.Tensor {
	lastDimension := input.Shape[len(input.Shape)-1]
	rowCount := input.Numel() / lastDimension
	return tensors.RMSNormLastDim(input.Reshape(rowCount, lastDimension), normEps).Reshape(input.Shape...)
}

// toBatchHeadLayout transposes [batch, sequence, head, dim] to
// [batch, head, sequence, dim], making per-(batch, head) slices contiguous for
// the attention kernels.
func toBatchHeadLayout(input *tensors.Tensor) *tensors.Tensor {
	batchSize, sequenceLength, headCount, headDimension := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	output := tensors.New(batchSize, headCount, sequenceLength, headDimension)
	source, destination := input.Data, output.Data
	parallel.Default().For(0, batchSize*headCount, func(index int) {
		batchIndex := index / headCount
		headIndex := index % headCount
		for position := 0; position < sequenceLength; position++ {
			sourceOffset := ((batchIndex*sequenceLength+position)*headCount + headIndex) * headDimension
			copy(destination[((batchIndex*headCount+headIndex)*sequenceLength+position)*headDimension:], source[sourceOffset:sourceOffset+headDimension])
		}
	})
	return output
}

// toBatchSequenceLayout transposes [batch, head, sequence, dim] back to
// [batch, sequence, head, dim].
func toBatchSequenceLayout(input *tensors.Tensor) *tensors.Tensor {
	batchSize, headCount, sequenceLength, headDimension := input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]
	output := tensors.New(batchSize, sequenceLength, headCount, headDimension)
	source, destination := input.Data, output.Data
	parallel.Default().For(0, batchSize*headCount, func(index int) {
		batchIndex := index / headCount
		headIndex := index % headCount
		for position := 0; position < sequenceLength; position++ {
			sourceOffset := ((batchIndex*headCount+headIndex)*sequenceLength + position) * headDimension
			copy(destination[((batchIndex*sequenceLength+position)*headCount+headIndex)*headDimension:], source[sourceOffset:sourceOffset+headDimension])
		}
	})
	return output
}

// headSlice returns the contiguous [sequenceLength, headDimension] block for
// the head identified by index inside a [batch, head, sequence, dim] buffer.
func headSlice(data []float32, index, sequenceLength, headDimension int) []float32 {
	start := index * sequenceLength * headDimension
	return data[start : start+sequenceLength*headDimension]
}

// CausalSelfAttention is a multi-head (grouped-query) causal attention layer
// with QK-norm, rotary embeddings, and an optional value-embedding gate.
type CausalSelfAttention struct {
	queryHeadCount     int
	keyValueHeadCount  int
	embeddingDimension int
	headDimension      int

	queryProjection  *layers.Linear
	keyProjection    *layers.Linear
	valueProjection  *layers.Linear
	outputProjection *layers.Linear

	valueEmbeddingGate *layers.Linear // nil on layers without value embeddings
}

// NewCausalSelfAttention builds an attention layer. hasValueEmbedding selects
// whether the ResFormer value-embedding gate is present on this layer.
func NewCausalSelfAttention(configuration Config, hasValueEmbedding bool) *CausalSelfAttention {
	headDimension := configuration.HeadDim()
	attention := &CausalSelfAttention{
		queryHeadCount:     configuration.NumHead,
		keyValueHeadCount:  configuration.NumKVHead,
		embeddingDimension: configuration.EmbedDim,
		headDimension:      headDimension,
		queryProjection:    layers.NewLinear(configuration.EmbedDim, configuration.NumHead*headDimension),
		keyProjection:      layers.NewLinear(configuration.EmbedDim, configuration.NumKVHead*headDimension),
		valueProjection:    layers.NewLinear(configuration.EmbedDim, configuration.NumKVHead*headDimension),
		outputProjection:   layers.NewLinear(configuration.EmbedDim, configuration.EmbedDim),
	}
	if hasValueEmbedding {
		attention.valueEmbeddingGate = layers.NewLinear(veGateChannels, configuration.NumKVHead)
	}
	return attention
}

// Forward computes attention for input of shape [batch, sequence, embedding].
// valueEmbedding is non-nil on ResFormer layers. cosine/sine are the rotary
// tables; positionOffset is the position of the first query; window is the
// (left, right) sliding window; cache, when non-nil, stores and reads KV. The
// result has shape [batch, sequence, embedding].
func (attention *CausalSelfAttention) Forward(input, valueEmbedding, cosine, sine *tensors.Tensor, positionOffset int, window [2]int, cache *KVBuffer, layer int) *tensors.Tensor {
	batchSize, sequenceLength := input.Shape[0], input.Shape[1]
	query := attention.queryProjection.Forward(input).Reshape(batchSize, sequenceLength, attention.queryHeadCount, attention.headDimension)
	key := attention.keyProjection.Forward(input).Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)
	value := attention.valueProjection.Forward(input).Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)

	if valueEmbedding != nil {
		valueEmbeddingHeads := valueEmbedding.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)
		gate := attention.valueEmbeddingGate.Forward(sliceChannels(input, 0, sequenceLength, veGateChannels)) // [B,T,Hkv]
		gate = tensors.Scale(tensors.Sigmoid(gate), veGateScale)
		value = addGateTimesValueEmbedding(value, gate, valueEmbeddingHeads)
	}

	query = ApplyRotary(query, cosine, sine, positionOffset)
	key = ApplyRotary(key, cosine, sine, positionOffset)

	query = tensors.Scale(normalizeLastDim(query), qkScale)
	key = tensors.Scale(normalizeLastDim(key), qkScale)

	// Transpose to [batch, head, sequence, dim] so per-head slices are
	// contiguous.
	queryHeadMajor := toBatchHeadLayout(query)
	keyHeadMajor := toBatchHeadLayout(key)
	valueHeadMajor := toBatchHeadLayout(value)

	headRatio := attention.queryHeadCount / attention.keyValueHeadCount
	outputHeadMajor := tensors.New(batchSize, attention.queryHeadCount, sequenceLength, attention.headDimension)
	headDimension := attention.headDimension

	parallel.Default().For(0, batchSize*attention.queryHeadCount, func(index int) {
		batchIndex := index / attention.queryHeadCount
		queryHead := index % attention.queryHeadCount
		keyValueHead := queryHead / headRatio

		queryHeadData := headSlice(queryHeadMajor.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)
		outputHeadData := headSlice(outputHeadMajor.Data, batchIndex*attention.queryHeadCount+queryHead, sequenceLength, headDimension)

		var keyFull, valueFull []float32
		if cache != nil {
			// Write k/v for this (batch, kv-head) into the cache and read the
			// full prefix as the available context.
			keyNew := headSlice(keyHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
			valueNew := headSlice(valueHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
			keyFull, valueFull = cache.writeKeyValue(layer, batchIndex, keyValueHead, keyNew, valueNew)
		} else {
			keyFull = headSlice(keyHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
			valueFull = headSlice(valueHeadMajor.Data, batchIndex*attention.keyValueHeadCount+keyValueHead, sequenceLength, headDimension)
		}
		keyLength := len(keyFull) / headDimension
		tensors.AttentionForward(queryHeadData, keyFull, valueFull, outputHeadData, nil, sequenceLength, keyLength, headDimension, positionOffset, window[0])
	})

	output := toBatchSequenceLayout(outputHeadMajor).Reshape(batchSize, sequenceLength, attention.queryHeadCount*attention.headDimension)
	return attention.outputProjection.Forward(output)
}

// addGateTimesValueEmbedding computes value + gate * valueEmbedding, where gate
// (shape [batch, sequence, head]) scales each row of valueEmbedding (shape
// [batch, sequence, head, dim]) before adding to value.
func addGateTimesValueEmbedding(value, gate, valueEmbedding *tensors.Tensor) *tensors.Tensor {
	batchSize, sequenceLength, headCount, headDimension := value.Shape[0], value.Shape[1], value.Shape[2], value.Shape[3]
	output := value.Clone()
	rowCount := batchSize * sequenceLength * headCount
	outputData, valueEmbeddingData, gateData := output.Data, valueEmbedding.Data, gate.Data
	for row := 0; row < rowCount; row++ {
		scale := gateData[row]
		base := row * headDimension
		for dimension := 0; dimension < headDimension; dimension++ {
			outputData[base+dimension] += scale * valueEmbeddingData[base+dimension]
		}
	}
	return output
}
