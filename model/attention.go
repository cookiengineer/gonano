package model

import (
	"github.com/cookiengineer/gonano/internal/parallel"
	"github.com/cookiengineer/gonano/kernels"
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
	parallel.KernelPool().For(0, batchSize*headCount, func(index int) {
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
	parallel.KernelPool().For(0, batchSize*headCount, func(index int) {
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

	// compressor is non-nil when HCA-style dense KV compression is enabled;
	// it merges every compressionRatio key/value rows into one entry.
	compressor       *ChannelCompressor
	compressionRatio int
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
	if ratio := configuration.Compression(); ratio > 1 {
		attention.compressionRatio = ratio
		attention.compressor = NewChannelCompressor(configuration.EmbedDim, headDimension, ratio)
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

	if attention.compressor != nil {
		if cache == nil {
			panic("model: compressed attention requires a KV cache during inference")
		}
		return attention.forwardCompressed(input, query, key, value, cache, layer, positionOffset)
	}

	// Transpose to [batch, head, sequence, dim] so per-head slices are
	// contiguous.
	queryHeadMajor := toBatchHeadLayout(query)
	keyHeadMajor := toBatchHeadLayout(key)
	valueHeadMajor := toBatchHeadLayout(value)

	headRatio := attention.queryHeadCount / attention.keyValueHeadCount
	outputHeadMajor := tensors.New(batchSize, attention.queryHeadCount, sequenceLength, attention.headDimension)
	headDimension := attention.headDimension
	kvHeadCount := attention.keyValueHeadCount

	// Collect the full KV prefix once per (batch, kv-head). Writing once here
	// (rather than once per query head) also avoids concurrent duplicate writes
	// to the same cache slot when grouped-query attention shares a KV head.
	keyValueFull := make([]keyValueSlice, batchSize*kvHeadCount)
	for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
		for keyValueHead := 0; keyValueHead < kvHeadCount; keyValueHead++ {
			slot := batchIndex*kvHeadCount + keyValueHead
			keyNew := headSlice(keyHeadMajor.Data, slot, sequenceLength, headDimension)
			valueNew := headSlice(valueHeadMajor.Data, slot, sequenceLength, headDimension)
			if cache != nil {
				// Write k/v into the cache and read the full prefix back.
				keyFull, valueFull := cache.writeKeyValue(layer, batchIndex, keyValueHead, keyNew, valueNew)
				keyValueFull[slot] = keyValueSlice{key: keyFull, value: valueFull}
			} else {
				keyValueFull[slot] = keyValueSlice{key: keyNew, value: valueNew}
			}
		}
	}

	keyLength := sequenceLength
	if cache != nil {
		keyLength = cache.Position() + sequenceLength
	}
	splits := attentionSplitCount(keyLength, batchSize*attention.queryHeadCount, parallel.Default().Workers())

	parallel.KernelPool().For(0, batchSize*attention.queryHeadCount, func(index int) {
		batchIndex := index / attention.queryHeadCount
		queryHead := index % attention.queryHeadCount
		keyValueHead := queryHead / headRatio

		queryHeadData := headSlice(queryHeadMajor.Data, index, sequenceLength, headDimension)
		outputHeadData := headSlice(outputHeadMajor.Data, index, sequenceLength, headDimension)
		full := keyValueFull[batchIndex*kvHeadCount+keyValueHead]
		attentionForwardCombined(queryHeadData, full.key, full.value, outputHeadData, nil, sequenceLength, keyLength, headDimension, positionOffset, window[0], splits)
	})

	output := toBatchSequenceLayout(outputHeadMajor).Reshape(batchSize, sequenceLength, attention.queryHeadCount*attention.headDimension)
	return attention.outputProjection.Forward(output)
}

// keyValueSlice is a (key, value) pair of matching [length, headDim] buffers.
type keyValueSlice struct {
	key   []float32
	value []float32
}

// attentionSplitK tuning. Split-K only helps when the key dimension is long
// and the (batch, head) task count cannot fill the cores.
const (
	attentionSplitKeyThreshold = 512
	attentionSplitMinKeys      = 128
	attentionMaxSplits         = 16
)

// attentionSplitCount returns how many key shards to use for a (batch, head)
// attention pass. It is 1 when the key dimension is short, when the existing
// (batch, head) parallelism already fills the cores, or when shards would hold
// too few keys.
func attentionSplitCount(keyLength, baseTasks, workers int) int {
	if keyLength < attentionSplitKeyThreshold || workers < 2 || baseTasks < 1 {
		return 1
	}
	splits := workers / baseTasks
	if splits < 2 {
		return 1
	}
	if byLength := keyLength / attentionSplitMinKeys; byLength < splits {
		splits = byLength
	}
	if splits > attentionMaxSplits {
		splits = attentionMaxSplits
	}
	if splits < 2 {
		return 1
	}
	return splits
}

// attentionForwardCombined computes one (batch, head) attention pass. With
// splits > 1 it partitions the key dimension across workers, computes each
// shard's unnormalized statistics, and merges them. logSumExp may be nil.
func attentionForwardCombined(query, key, value, output, logSumExp []float32, queryLength, keyLength, headDim, positionOffset, window, splits int) {
	if splits <= 1 {
		tensors.AttentionForward(query, key, value, output, logSumExp, queryLength, keyLength, headDim, positionOffset, window)
		return
	}
	partials := make([]kernels.AttentionSplitResult, splits)
	keysPerSplit := (keyLength + splits - 1) / splits
	parallel.KernelPool().For(0, splits, func(split int) {
		keyStart := split * keysPerSplit
		keyEnd := min(keyStart+keysPerSplit, keyLength)
		partial := kernels.AttentionSplitResult{
			Accumulator: make([]float32, queryLength*headDim),
			Maximum:     make([]float32, queryLength),
			Sum:         make([]float32, queryLength),
		}
		tensors.AttentionForwardSplit(query, key, value, partial.Accumulator, partial.Maximum, partial.Sum,
			queryLength, keyLength, headDim, positionOffset, window, keyStart, keyEnd)
		partials[split] = partial
	})
	tensors.AttentionCombine(partials, output, logSumExp, queryLength, headDim)
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
