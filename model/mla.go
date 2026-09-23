package model

import (
	"github.com/cookiengineer/gonano/model/layers"
	"github.com/cookiengineer/gonano/tensors"
)

// MLAttention holds the absorbed Multi-head Latent Attention parameters
// (DeepSeek-V4.1 §2.3/§4.2.1). Content keys and values are up-projected from a
// shared latent c_kv; a decoupled rotary key is stored alongside it. At
// inference the content up-projections are absorbed into the query and output
// so attention reads the cached latent directly, which is what makes the KV
// cache hold a single latent row per token instead of a key and a value per
// head.
type MLAttention struct {
	// QueryDown projects the hidden state to the query latent.
	QueryDown *layers.Linear // [latentDim, EmbedDim]
	// QueryUp expands the query latent to the per-head content query.
	QueryUp *layers.Linear // [NumHead*contentDim, latentDim]
	// QueryRope projects the decoupled rotary query.
	QueryRope *layers.Linear // [NumHead*ropeDim, EmbedDim]
	// KVDown projects the hidden state to the shared key/value latent.
	KVDown *layers.Linear // [latentDim, EmbedDim]
	// KeyUp expands the latent to the per-head content key.
	KeyUp *layers.Linear // [NumKVHead*contentDim, latentDim]
	// ValueUp expands the latent to the per-head value.
	ValueUp *layers.Linear // [NumKVHead*valueDim, latentDim]
	// KeyRope projects the decoupled rotary key.
	KeyRope *layers.Linear // [NumKVHead*ropeDim, EmbedDim]

	queryHeadCount int
	kvHeadCount    int
	contentDim     int
	valueDim       int
	ropeDim        int
	latentDim      int
}

// newMLAttention builds the MLA parameters for a config. The caller guarantees
// MLAEnabled.
func newMLAttention(config Config) *MLAttention {
	headDimension := config.HeadDim()
	ropeDim := config.MLARotaryDimension()
	return &MLAttention{
		QueryDown:      layers.NewLinear(config.EmbedDim, config.MLARank()),
		QueryUp:        layers.NewLinear(config.MLARank(), config.NumHead*(headDimension-ropeDim)),
		QueryRope:      layers.NewLinear(config.EmbedDim, config.NumHead*ropeDim),
		KVDown:         layers.NewLinear(config.EmbedDim, config.MLARank()),
		KeyUp:          layers.NewLinear(config.MLARank(), config.NumKVHead*(headDimension-ropeDim)),
		ValueUp:        layers.NewLinear(config.MLARank(), config.NumKVHead*headDimension),
		KeyRope:        layers.NewLinear(config.EmbedDim, config.NumKVHead*ropeDim),
		queryHeadCount: config.NumHead,
		kvHeadCount:    config.NumKVHead,
		contentDim:     headDimension - ropeDim,
		valueDim:       headDimension,
		ropeDim:        ropeDim,
		latentDim:      config.MLARank(),
	}
}

// parameters returns the MLA weight tensors in a deterministic order.
func (mla *MLAttention) parameters() []*tensors.Tensor {
	return []*tensors.Tensor{
		mla.QueryDown.Weight,
		mla.QueryUp.Weight,
		mla.QueryRope.Weight,
		mla.KVDown.Weight,
		mla.KeyUp.Weight,
		mla.ValueUp.Weight,
		mla.KeyRope.Weight,
	}
}

// numParameters returns the total number of MLA weights.
func (mla *MLAttention) numParameters() int {
	total := 0
	for _, parameter := range mla.parameters() {
		total += parameter.Numel()
	}
	return total
}

// mlaParamsForConfig returns the per-layer MLA matrix-parameter count without
// building a model, for the scaling-law and FLOP formulas.
func mlaParamsForConfig(config Config) int {
	embeddingDimension := config.EmbedDim
	headDimension := config.HeadDim()
	ropeDim := config.MLARotaryDimension()
	contentDim := headDimension - ropeDim
	latent := config.MLARank()
	queryHeads := config.NumHead
	kvHeads := config.NumKVHead
	return latent*embeddingDimension +
		queryHeads*contentDim*latent +
		queryHeads*ropeDim*embeddingDimension +
		latent*embeddingDimension +
		kvHeads*contentDim*latent +
		kvHeads*headDimension*latent +
		kvHeads*ropeDim*embeddingDimension
}

// mlaProjectHeads builds the MLA query, key, and value head tensors from the
// hidden state, saving the shared query and key/value latents for the backward
// pass. The query and key lay out the non-rotary content first and the
// decoupled rotary part last, so ApplyRotary rotates exactly the rotary slice.
func (attention *CausalSelfAttention) mlaProjectHeads(input *tensors.Tensor) (query, key, value, queryLatent, kvLatent *tensors.Tensor) {
	mla := attention.mla
	batchSize, sequenceLength := input.Shape[0], input.Shape[1]

	queryLatent = mla.QueryDown.Forward(input)
	queryContent := mla.QueryUp.Forward(queryLatent).Reshape(batchSize, sequenceLength, mla.queryHeadCount, mla.contentDim)
	queryRope := mla.QueryRope.Forward(input).Reshape(batchSize, sequenceLength, mla.queryHeadCount, mla.ropeDim)
	query = concatHeads(queryContent, queryRope, mla.queryHeadCount, mla.contentDim, mla.ropeDim)

	kvLatent = mla.KVDown.Forward(input)
	keyContent := mla.KeyUp.Forward(kvLatent).Reshape(batchSize, sequenceLength, mla.kvHeadCount, mla.contentDim)
	keyRope := mla.KeyRope.Forward(input).Reshape(batchSize, sequenceLength, mla.kvHeadCount, mla.ropeDim)
	key = concatHeads(keyContent, keyRope, mla.kvHeadCount, mla.contentDim, mla.ropeDim)

	value = mla.ValueUp.Forward(kvLatent).Reshape(batchSize, sequenceLength, mla.kvHeadCount, mla.valueDim)
	return query, key, value, queryLatent, kvLatent
}

// mlaProjectBackward backpropagates the query, key, and value gradients through
// the MLA up-projections into the hidden state.
func (attention *CausalSelfAttention) mlaProjectBackward(context *attentionContext, gradientQuery, gradientKey, gradientValue, input *tensors.Tensor) *tensors.Tensor {
	mla := attention.mla
	batchSize, sequenceLength := input.Shape[0], input.Shape[1]

	queryContent := tensors.New(batchSize, sequenceLength, mla.queryHeadCount, mla.contentDim)
	queryRope := tensors.New(batchSize, sequenceLength, mla.queryHeadCount, mla.ropeDim)
	splitHeads(gradientQuery, queryContent, queryRope, mla.queryHeadCount, mla.contentDim, mla.ropeDim)
	gradientQueryLatent := mla.QueryUp.Backward(context.mlaQueryLatent, queryContent.Reshape(batchSize, sequenceLength, mla.queryHeadCount*mla.contentDim))
	gradientInput := mla.QueryDown.Backward(input, gradientQueryLatent)
	gradientInput = tensors.Add(gradientInput, mla.QueryRope.Backward(input, queryRope.Reshape(batchSize, sequenceLength, mla.queryHeadCount*mla.ropeDim)))

	keyContent := tensors.New(batchSize, sequenceLength, mla.kvHeadCount, mla.contentDim)
	keyRope := tensors.New(batchSize, sequenceLength, mla.kvHeadCount, mla.ropeDim)
	splitHeads(gradientKey, keyContent, keyRope, mla.kvHeadCount, mla.contentDim, mla.ropeDim)
	gradientKVLatent := mla.KeyUp.Backward(context.mlaKVLatent, keyContent.Reshape(batchSize, sequenceLength, mla.kvHeadCount*mla.contentDim))
	valueGradient := gradientValue.Reshape(batchSize, sequenceLength, mla.kvHeadCount*mla.valueDim)
	gradientKVLatent = tensors.Add(gradientKVLatent, mla.ValueUp.Backward(context.mlaKVLatent, valueGradient))
	gradientInput = tensors.Add(gradientInput, mla.KVDown.Backward(input, gradientKVLatent))
	gradientInput = tensors.Add(gradientInput, mla.KeyRope.Backward(input, keyRope.Reshape(batchSize, sequenceLength, mla.kvHeadCount*mla.ropeDim)))
	return gradientInput
}

// concatHeads concatenates a content block and a rope block per head:
// [B,T,H,contentDim] and [B,T,H,ropeDim] become [B,T,H,contentDim+ropeDim].
func concatHeads(content, rope *tensors.Tensor, headCount, contentDim, ropeDim int) *tensors.Tensor {
	batchSize, sequenceLength := content.Shape[0], content.Shape[1]
	headDimension := contentDim + ropeDim
	output := tensors.New(batchSize, sequenceLength, headCount, headDimension)
	rows := batchSize * sequenceLength * headCount
	for row := 0; row < rows; row++ {
		copy(output.Data[row*headDimension:row*headDimension+contentDim], content.Data[row*contentDim:(row+1)*contentDim])
		copy(output.Data[row*headDimension+contentDim:(row+1)*headDimension], rope.Data[row*ropeDim:(row+1)*ropeDim])
	}
	return output
}

// splitHeads is the inverse of concatHeads.
func splitHeads(input, content, rope *tensors.Tensor, headCount, contentDim, ropeDim int) {
	headDimension := contentDim + ropeDim
	rows := batchSizeTimesHeads(input, headDimension)
	for row := 0; row < rows; row++ {
		copy(content.Data[row*contentDim:(row+1)*contentDim], input.Data[row*headDimension:row*headDimension+contentDim])
		copy(rope.Data[row*ropeDim:(row+1)*ropeDim], input.Data[row*headDimension+contentDim:(row+1)*headDimension])
	}
}

// batchSizeTimesHeads returns the number of [head] rows in a [B,T,H,D] tensor.
func batchSizeTimesHeads(input *tensors.Tensor, headDimension int) int {
	return input.Numel() / headDimension
}

// mlaForwardInference computes absorbed MLA attention for one layer. It stores
// only the shared latent and the decoupled rotary key in the cache, folds the
// content key up-projection into the query, and concatenates the folded content
// query with the rotary query (and the latent with the rotary key) so the
// attention score is the per-position sum of the content and rotary scores. The
// value is the latent padded so the kernel's key and value widths match; the
// value up-projection is applied to the attention-weighted latent sum, so no
// per-cached-position up-projection is ever materialized.
func (attention *CausalSelfAttention) mlaForwardInference(input, cosine, sine *tensors.Tensor, positionOffset int, window [2]int, cache *KVBuffer, layer int) *tensors.Tensor {
	mla := attention.mla
	batchSize, sequenceLength := input.Shape[0], input.Shape[1]
	contentDim := mla.contentDim
	ropeDim := mla.ropeDim
	latentDim := mla.latentDim
	queryHeads := mla.queryHeadCount
	kvHeads := mla.kvHeadCount
	valueDim := mla.valueDim
	headRatio := queryHeads / kvHeads
	headDim := latentDim + ropeDim
	if !cache.MLAEnabled() {
		panic("model: MLA attention requires an MLA-enabled KV cache")
	}

	queryLatent := mla.QueryDown.Forward(input)
	queryContent := mla.QueryUp.Forward(queryLatent).Reshape(batchSize, sequenceLength, queryHeads, contentDim)
	queryRope := ApplyRotary(mla.QueryRope.Forward(input).Reshape(batchSize, sequenceLength, queryHeads, ropeDim), cosine, sine, positionOffset)

	kvLatent := mla.KVDown.Forward(input)
	keyRope := ApplyRotary(mla.KeyRope.Forward(input).Reshape(batchSize, sequenceLength, kvHeads, ropeDim), cosine, sine, positionOffset)

	for batch := 0; batch < batchSize; batch++ {
		for row := 0; row < sequenceLength; row++ {
			latentBase := (batch*sequenceLength + row) * latentDim
			cache.AppendMLALatentAt(layer, batch, row, kvLatent.Data[latentBase:latentBase+latentDim])
			ropeBase := (batch*sequenceLength + row) * kvHeads * ropeDim
			cache.AppendMLARopeKeyAt(layer, batch, row, keyRope.Data[ropeBase:ropeBase+kvHeads*ropeDim])
		}
	}

	count := cache.Position() + sequenceLength
	output := tensors.New(batchSize, sequenceLength, queryHeads*valueDim)
	keyUp := mla.KeyUp.Weight.Data
	valueUp := mla.ValueUp.Weight.Data

	query := make([]float32, sequenceLength*headDim)
	key := make([]float32, count*headDim)
	value := make([]float32, count*headDim)
	attentionOut := make([]float32, sequenceLength*headDim)
	logSumExp := make([]float32, sequenceLength)

	for batch := 0; batch < batchSize; batch++ {
		latentCache := cache.MLALatent(layer, batch, count)
		ropeCache := cache.MLARopeKey(layer, batch, count)
		for queryHead := 0; queryHead < queryHeads; queryHead++ {
			kvHead := queryHead / headRatio

			// Fold the content key up-projection into the query, then build
			// Q = [q_abs | q_rope].
			for token := 0; token < sequenceLength; token++ {
				contentBase := ((batch*sequenceLength+token)*queryHeads + queryHead) * contentDim
				queryBase := token * headDim
				for rank := 0; rank < latentDim; rank++ {
					var sum float32
					for channel := 0; channel < contentDim; channel++ {
						sum += queryContent.Data[contentBase+channel] * keyUp[(kvHead*contentDim+channel)*latentDim+rank]
					}
					query[queryBase+rank] = sum
				}
				ropeBase := ((batch*sequenceLength+token)*queryHeads + queryHead) * ropeDim
				copy(query[queryBase+latentDim:queryBase+headDim], queryRope.Data[ropeBase:ropeBase+ropeDim])
			}

			// K = [c_kv | k_rope], V = [c_kv | 0].
			for token := 0; token < count; token++ {
				base := token * headDim
				copy(key[base:base+latentDim], latentCache[token*latentDim:(token+1)*latentDim])
				ropeBase := token*kvHeads*ropeDim + kvHead*ropeDim
				copy(key[base+latentDim:base+headDim], ropeCache[ropeBase:ropeBase+ropeDim])
				copy(value[base:base+latentDim], latentCache[token*latentDim:(token+1)*latentDim])
				for index := base + latentDim; index < base+headDim; index++ {
					value[index] = 0
				}
			}

			tensors.AttentionForward(query, key, value, attentionOut, logSumExp,
				sequenceLength, count, headDim, positionOffset, window[0])

			for token := 0; token < sequenceLength; token++ {
				outputBase := ((batch*sequenceLength+token)*queryHeads + queryHead) * valueDim
				latentOutBase := token * headDim
				for dimension := 0; dimension < valueDim; dimension++ {
					var sum float32
					for rank := 0; rank < latentDim; rank++ {
						sum += attentionOut[latentOutBase+rank] * valueUp[(kvHead*valueDim+dimension)*latentDim+rank]
					}
					output.Data[outputBase+dimension] = sum
				}
			}
		}
	}
	sequenceFlat := output.Reshape(batchSize, sequenceLength, queryHeads*valueDim)
	return attention.outputProjection.Forward(sequenceFlat)
}
