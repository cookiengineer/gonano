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

	// queryDown, when non-nil, is the low-rank down-projection feeding the
	// query up-projection (MLA-style query compression). kvDown, when non-nil,
	// is the shared latent from which key and value are up-projected.
	queryDown *layers.Linear
	kvDown    *layers.Linear

	valueEmbeddingGate *layers.Linear // nil on layers without value embeddings

	// compressor is non-nil when HCA-style dense KV compression is enabled;
	// it merges every compressionRatio key/value rows into one entry.
	compressor       *ChannelCompressor
	compressionRatio int
	// indexer is non-nil when CSA-style sparse selection is enabled on top of
	// compression; sparseTopK is the number of blocks each query attends to.
	indexer           *SparseIndexer
	sparseTopK        int
	indexerLossWeight float32
	indexerPool       int
	indexerCandidates int
	// swaWindow is the local sliding-window width merged with the compressed
	// global branch on compressed layers (DeepSeek-V4.1 §2.2). Zero disables
	// the local branch.
	swaWindow int

	// reuseMode/producer implement cross-layer compressed KV/index reuse. A
	// full layer produces compressed KV (and selection); reindex/reuse layers
	// consume the producing layer's state. producer is the layer index whose
	// compressed state this layer uses (== its own index on full layers).
	reuseMode ReuseMode
	producer  int

	// ced marks a decoder layer of the causal encoder-decoder split. On such a
	// layer the global compressed keys/values are projected from the encoder's
	// final hidden state (held in the group's compressionShare) instead of the
	// layer's own hidden state; the local sliding-window branch still reads the
	// layer's own hidden state.
	ced bool

	// mla, when non-nil, replaces the standard query/key/value projections with
	// the absorbed Multi-head Latent Attention parameters.
	mla *MLAttention
}

// usesCompression reports whether the layer takes the compressed-attention
// path. This is true for full, reindex, and reuse layers whenever a compression
// ratio is configured.
func (attention *CausalSelfAttention) usesCompression() bool {
	return attention.compressionRatio > 1
}

// projectQuery computes the query from the attention input, applying the
// low-rank query bottleneck when configured (MLA-style query compression).
func (attention *CausalSelfAttention) projectQuery(input *tensors.Tensor) *tensors.Tensor {
	if attention.queryDown != nil {
		return attention.queryProjection.Forward(attention.queryDown.Forward(input))
	}
	return attention.queryProjection.Forward(input)
}

// projectKeyValue computes the key and value from the attention input. When a
// low-rank KV latent is configured both are up-projected from one shared
// latent, otherwise they are projected directly from the input.
func (attention *CausalSelfAttention) projectKeyValue(input *tensors.Tensor) (key, value *tensors.Tensor) {
	if attention.kvDown != nil {
		latent := attention.kvDown.Forward(input)
		return attention.keyProjection.Forward(latent), attention.valueProjection.Forward(latent)
	}
	return attention.keyProjection.Forward(input), attention.valueProjection.Forward(input)
}

// cedGlobalKeyValue projects a decoder layer's encoder-hidden state into its
// global keys and values using the layer's own projections, applying RoPE and
// the QK norm to the key. It returns head-major key/value tensors plus the
// post-rope, pre-norm key (needed by the backward pass). The value-embedding
// gate does not apply to the global branch because it reads the layer's own
// hidden state.
func (attention *CausalSelfAttention) cedGlobalKeyValue(encoderHidden, cosine, sine *tensors.Tensor, positionOffset int) (keyHeadMajor, valueHeadMajor, keyRotary *tensors.Tensor) {
	batchSize, sequenceLength := encoderHidden.Shape[0], encoderHidden.Shape[1]
	key, value := attention.projectKeyValue(encoderHidden)
	key = key.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)
	value = value.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)
	key = ApplyRotary(key, cosine, sine, positionOffset)
	keyRotary = key
	key = tensors.Scale(normalizeLastDim(key), qkScale)
	return toBatchHeadLayout(key), toBatchHeadLayout(value), keyRotary
}

// NewCausalSelfAttention builds an attention layer. hasValueEmbedding selects
// whether the ResFormer value-embedding gate is present on this layer. layer is
// the block index, used to resolve the cross-layer reuse role.
func NewCausalSelfAttention(configuration Config, hasValueEmbedding bool, layer int) *CausalSelfAttention {
	headDimension := configuration.HeadDim()
	attention := &CausalSelfAttention{
		queryHeadCount:     configuration.NumHead,
		keyValueHeadCount:  configuration.NumKVHead,
		embeddingDimension: configuration.EmbedDim,
		headDimension:      headDimension,
		outputProjection:   layers.NewLinear(configuration.EmbedDim, configuration.EmbedDim),
		reuseMode:          configuration.ReuseModeAt(layer),
		producer:           configuration.ReuseProducer(layer),
		ced:                configuration.IsDecoderLayer(layer),
	}
	if hasValueEmbedding {
		attention.valueEmbeddingGate = layers.NewLinear(veGateChannels, configuration.NumKVHead)
	}
	if configuration.MLAEnabled() {
		attention.mla = newMLAttention(configuration)
		return attention
	}
	if rank := configuration.QueryRank(); rank > 0 {
		attention.queryDown = layers.NewLinear(configuration.EmbedDim, rank)
		attention.queryProjection = layers.NewLinear(rank, configuration.NumHead*headDimension)
	} else {
		attention.queryProjection = layers.NewLinear(configuration.EmbedDim, configuration.NumHead*headDimension)
	}
	if rank := configuration.KVRank(); rank > 0 {
		attention.kvDown = layers.NewLinear(configuration.EmbedDim, rank)
		attention.keyProjection = layers.NewLinear(rank, configuration.NumKVHead*headDimension)
		attention.valueProjection = layers.NewLinear(rank, configuration.NumKVHead*headDimension)
	} else {
		attention.keyProjection = layers.NewLinear(configuration.EmbedDim, configuration.NumKVHead*headDimension)
		attention.valueProjection = layers.NewLinear(configuration.EmbedDim, configuration.NumKVHead*headDimension)
	}
	if ratio := configuration.Compression(); ratio > 1 {
		attention.compressionRatio = ratio
		attention.sparseTopK = configuration.SparseTopK
		attention.swaWindow = configuration.SWAWindowSize()
		// Only full layers produce compressed KV. Reindex layers borrow the
		// compressed KV but keep their own indexer for fresh selection; reuse
		// layers borrow both and own neither.
		if attention.reuseMode == ReuseFull {
			attention.compressor = NewChannelCompressor(configuration.EmbedDim, headDimension, ratio)
		}
		if configuration.SparseTopK > 0 && attention.reuseMode != ReuseReuse {
			dim, heads := configuration.indexerDefaults()
			attention.indexerLossWeight = configuration.IndexerWeight()
			attention.indexerPool = configuration.IndexerPool
			attention.indexerCandidates = configuration.IndexerCandidateBudget()
			attention.indexer = NewSparseIndexer(configuration.EmbedDim, headDimension, dim, heads)
		}
	}
	return attention
}

// Forward computes attention for input of shape [batch, sequence, embedding].
// valueEmbedding is non-nil on ResFormer layers. cosine/sine are the rotary
// tables; positionOffset is the position of the first query; window is the
// (left, right) sliding window; cache, when non-nil, stores and reads KV. The
// result has shape [batch, sequence, embedding].
func (attention *CausalSelfAttention) Forward(input, valueEmbedding, cosine, sine *tensors.Tensor, positionOffset int, window [2]int, cache *KVBuffer, layer int, share *compressionShare) *tensors.Tensor {
	if attention.mla != nil {
		return attention.mlaForwardInference(input, cosine, sine, positionOffset, window, cache, layer)
	}
	batchSize, sequenceLength := input.Shape[0], input.Shape[1]
	query := attention.projectQuery(input).Reshape(batchSize, sequenceLength, attention.queryHeadCount, attention.headDimension)
	keyProjected, valueProjected := attention.projectKeyValue(input)
	key := keyProjected.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)
	value := valueProjected.Reshape(batchSize, sequenceLength, attention.keyValueHeadCount, attention.headDimension)

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

	if attention.usesCompression() {
		if cache == nil {
			panic("model: compressed attention requires a KV cache during inference")
		}
		// CED decoder layers project their global keys/values from the encoder's
		// final hidden state. The local branch keeps using the layer's own
		// keys/values.
		var encoderHidden, cedKeySequence, cedValueSequence *tensors.Tensor
		if attention.ced && share != nil && share.encoderHidden != nil {
			encoderHidden = share.encoderHidden
			keyHead, valueHead, _ := attention.cedGlobalKeyValue(encoderHidden, cosine, sine, positionOffset)
			cedKeySequence = toBatchSequenceLayout(keyHead)
			cedValueSequence = toBatchSequenceLayout(valueHead)
		}
		return attention.forwardCompressed(input, query, key, value, encoderHidden, cedKeySequence, cedValueSequence, cache, layer, positionOffset, share)
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
