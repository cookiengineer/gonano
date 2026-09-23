package model

import (
	"github.com/cookiengineer/gonano/tensors"
)

// DrafterLayers is the number of transformer blocks in the DSpark drafter
// (DeepSeek-V4.1 §2.4.3).
const DrafterLayers = 3

// DrafterConfig derives the drafter configuration from a backbone config. The
// drafter shares the backbone's tokenizer, so it keeps the vocabulary and head
// geometry, but it is a small standalone transformer with its own embedding and
// language-model head and a bounded context (the paper's sliding window). The
// context is capped so drafting stays cheap.
func DrafterConfig(backbone Config) Config {
	sequenceLen := backbone.SequenceLen
	if sequenceLen > 128 {
		sequenceLen = 128
	}
	config := Config{
		SequenceLen:   sequenceLen,
		VocabSize:     backbone.VocabSize,
		NumLayer:      DrafterLayers,
		NumHead:       backbone.NumHead,
		NumKVHead:     backbone.NumKVHead,
		EmbedDim:      backbone.EmbedDim,
		WindowPattern: "L",
		RotaryDims:    backbone.RotaryDims,
	}
	config.Validate()
	return config
}

// NewDrafter builds a randomly initialized drafter for a backbone. The caller
// initializes it with InitWeights or trains it by distilling the backbone
// (DeepSeek-V4.1 §2.4.3 trains the drafter separately with the backbone frozen).
func NewDrafter(backbone *Transformer) *Transformer {
	return NewTransformer(DrafterConfig(backbone.Config))
}

// DraftTokens greedily proposes up to count tokens following the context, using
// the model as a draft model. It only sees the last SequenceLen tokens, so the
// per-round drafting cost is bounded independently of the context length. It
// returns fewer than count tokens only when the context is empty.
func (model *Transformer) DraftTokens(context []int, count int) []int {
	if count <= 0 {
		return nil
	}
	window := model.Config.SequenceLen
	if len(context) > window {
		context = context[len(context)-window:]
	}
	if len(context) == 0 {
		return nil
	}

	cache := NewKVBuffer(1, len(context)+count, model.Config.NumLayer, model.Config.NumKVHead, model.Config.HeadDim())
	indexes := tensors.NewInt32sWithData([]int{1, len(context)}, toInt32(context))
	logits := model.Forward(indexes, cache)

	drafts := make([]int, 0, count)
	for len(drafts) < count {
		token := argmaxLastRow(logits, model.Config.VocabSize)
		drafts = append(drafts, token)
		if len(drafts) == count {
			break
		}
		next := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(token)})
		logits = model.Forward(next, cache)
	}
	return drafts
}

// SetPreviousEmbeddingForToken stores the pre-smear normalized embedding of a
// token as the cache's previous embedding, so a later multi-token forward
// against the partially filled cache smears its first position correctly.
// Speculative verification calls it after rewinding to the last accepted token.
func (model *Transformer) SetPreviousEmbeddingForToken(cache *KVBuffer, token int) {
	if cache == nil {
		return
	}
	indexes := tensors.NewInt32sWithData([]int{1, 1}, []int32{int32(token)})
	cache.SetPrevEmbedding(normalizeLastDim(model.tokenEmbedding.Forward(indexes)))
}

// argmaxLastRow returns the argmax of the last row's first vocabSize logits.
func argmaxLastRow(logits *tensors.Tensor, vocabSize int) int {
	rowCount := logits.Numel() / vocabSize
	base := (rowCount - 1) * vocabSize
	best := 0
	bestValue := logits.Data[base]
	for index := 1; index < vocabSize; index++ {
		if logits.Data[base+index] > bestValue {
			bestValue = logits.Data[base+index]
			best = index
		}
	}
	return best
}

// toInt32 converts token ids to the int32 representation used by the tensors.
func toInt32(values []int) []int32 {
	converted := make([]int32, len(values))
	for index, value := range values {
		converted[index] = int32(value)
	}
	return converted
}
