package data

import (
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// DocProvider yields the next batch of raw documents (strings) together with
// the dataset position. It must be infinite (cycle forever).
type DocProvider func() ([]string, State)

// PretrainLoader is the BOS-aligned best-fit dataloader used for pretraining.
//
// Every row starts with the BOS token, documents are packed with a best-fit
// algorithm to minimize cropping, and when nothing fits the shortest document
// is cropped to fill the row exactly. Utilization is 100% (no padding).
type PretrainLoader struct {
	tok        *tokenizer.Tokenizer
	B, T       int
	bufferSize int
	provider   DocProvider

	docBuffer [][]int32
	state     State
	scratch   []int32
	// segmentScratch holds the per-token document index used for sample-level
	// attention masking (DeepSeek-V4.1 §4.2.2).
	segmentScratch []int32
}

// NewPretrainLoader builds a pretraining dataloader. provider yields batches
// of raw documents (already split into documents; tokenization is done here).
func NewPretrainLoader(tok *tokenizer.Tokenizer, batchSize, sequenceLength int, provider DocProvider, bufferSize int) *PretrainLoader {
	if bufferSize <= 0 {
		bufferSize = 1000
	}
	return &PretrainLoader{
		tok:            tok,
		B:              batchSize,
		T:              sequenceLength,
		bufferSize:     bufferSize,
		provider:       provider,
		scratch:        make([]int32, sequenceLength+1),
		segmentScratch: make([]int32, sequenceLength+1),
	}
}

// refill pulls more documents until the buffer has at least bufferSize entries.
func (loader *PretrainLoader) refill() {
	bos := loader.tok.BOSTokenID()
	for len(loader.docBuffer) < loader.bufferSize {
		docs, state := loader.provider()
		loader.state = state
		for _, document := range docs {
			ids := loader.tok.Encode(document)
			row := make([]int32, 0, len(ids)+1)
			row = append(row, int32(bos))
			row = append(row, toInt32(ids)...)
			loader.docBuffer = append(loader.docBuffer, row)
		}
	}
}

func toInt32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for index, value := range ids {
		out[index] = int32(value)
	}
	return out
}

// Next produces the next training batch: inputs of shape [B,T] and targets of
// shape [B,T] (inputs shifted by one). It also returns the dataset position
// for checkpointing.
func (loader *PretrainLoader) Next() (*tensors.Int32s, *tensors.Int32s, State) {
	inputs, targets, _, state := loader.next(false)
	return inputs, targets, state
}

// NextSegments is Next plus a per-token segment id of shape [B,T]. A new
// segment begins at every packed document (each starts with BOS), so callers
// can apply sample-level attention masking and keep tokens from different
// documents from attending to each other (DeepSeek-V4.1 §4.2.2).
func (loader *PretrainLoader) NextSegments() (*tensors.Int32s, *tensors.Int32s, *tensors.Int32s, State) {
	return loader.next(true)
}

func (loader *PretrainLoader) next(withSegments bool) (*tensors.Int32s, *tensors.Int32s, *tensors.Int32s, State) {
	rowCapacity := loader.T + 1
	inputs := tensors.NewInt32s(loader.B, loader.T)
	targets := tensors.NewInt32s(loader.B, loader.T)
	var segments *tensors.Int32s
	if withSegments {
		segments = tensors.NewInt32s(loader.B, loader.T)
	}

	for row := 0; row < loader.B; row++ {
		pos := 0
		segment := int32(0)
		for pos < rowCapacity {
			for len(loader.docBuffer) < loader.bufferSize {
				loader.refill()
			}
			remaining := rowCapacity - pos

			bestIdx, bestLen := -1, 0
			for index, document := range loader.docBuffer {
				if len(document) <= remaining && len(document) > bestLen {
					bestIdx, bestLen = index, len(document)
				}
			}

			if bestIdx >= 0 {
				document := loader.docBuffer[bestIdx]
				loader.docBuffer = append(loader.docBuffer[:bestIdx], loader.docBuffer[bestIdx+1:]...)
				copy(loader.scratch[pos:], document)
				if withSegments {
					for fill := pos; fill < pos+len(document); fill++ {
						loader.segmentScratch[fill] = segment
					}
				}
				pos += len(document)
			} else {
				shortest := 0
				for index := range loader.docBuffer {
					if len(loader.docBuffer[index]) < len(loader.docBuffer[shortest]) {
						shortest = index
					}
				}
				document := loader.docBuffer[shortest]
				loader.docBuffer = append(loader.docBuffer[:shortest], loader.docBuffer[shortest+1:]...)
				copy(loader.scratch[pos:], document[:remaining])
				if withSegments {
					for fill := pos; fill < pos+remaining; fill++ {
						loader.segmentScratch[fill] = segment
					}
				}
				pos += remaining
			}
			segment++
		}
		copy(inputs.Data[row*loader.T:], loader.scratch[:loader.T])
		copy(targets.Data[row*loader.T:], loader.scratch[1:loader.T+1])
		if withSegments {
			copy(segments.Data[row*loader.T:], loader.segmentScratch[:loader.T])
		}
	}
	return inputs, targets, segments, loader.state
}

// ConvProvider yields the next batch of conversations for SFT.
type ConvProvider func() ([]*tokenizer.Conversation, bool)

type convWithMask struct {
	ids  []int32
	mask []int32
}

// SFTLoader is the BOS-aligned best-fit packing dataloader used for supervised
// fine-tuning. Unlike pretraining, it never crops conversations: when no
// conversation fits, the remainder of the row is padded (with BOS) and the
// padded targets are masked with -1 (the cross-entropy ignore index).
type SFTLoader struct {
	tok        *tokenizer.Tokenizer
	B, T       int
	bufferSize int
	provider   ConvProvider

	convBuffer  []convWithMask
	done        bool
	scratch     []int32
	maskScratch []int32
	// segmentScratch holds the per-token conversation index used for
	// sample-level attention masking (DeepSeek-V4.1 §4.2.2).
	segmentScratch []int32
}

// NewSFTLoader builds an SFT dataloader over the given conversation provider.
func NewSFTLoader(tok *tokenizer.Tokenizer, batchSize, sequenceLength int, provider ConvProvider, bufferSize int) *SFTLoader {
	if bufferSize <= 0 {
		bufferSize = 100
	}
	return &SFTLoader{
		tok:            tok,
		B:              batchSize,
		T:              sequenceLength,
		bufferSize:     bufferSize,
		provider:       provider,
		scratch:        make([]int32, sequenceLength+1),
		maskScratch:    make([]int32, sequenceLength+1),
		segmentScratch: make([]int32, sequenceLength+1),
	}
}

// refill pulls more conversations until the buffer has enough entries.
func (loader *SFTLoader) refill() {
	for len(loader.convBuffer) < loader.bufferSize {
		convs, ok := loader.provider()
		if !ok {
			loader.done = true
			return
		}
		for _, conversation := range convs {
			ids, mask := loader.tok.RenderConversation(conversation, 1<<30)
			loader.convBuffer = append(loader.convBuffer, convWithMask{ids: toInt32(ids), mask: toInt32(mask)})
		}
	}
}

// Next produces the next SFT batch of shape [B,T]. Targets have -1 at masked
// (non-assistant) and padded positions. ok is false when the provider is
// exhausted and the buffer is empty.
func (loader *SFTLoader) Next() (*tensors.Int32s, *tensors.Int32s, bool) {
	inputs, targets, _, ok := loader.next(false)
	return inputs, targets, ok
}

// NextSegments is Next plus a per-token segment id of shape [B,T]. Each packed
// conversation is its own segment and the padded tail is a final segment, so
// callers can keep tokens from different conversations from attending to each
// other (DeepSeek-V4.1 §4.2.2).
func (loader *SFTLoader) NextSegments() (*tensors.Int32s, *tensors.Int32s, *tensors.Int32s, bool) {
	return loader.next(true)
}

func (loader *SFTLoader) next(withSegments bool) (*tensors.Int32s, *tensors.Int32s, *tensors.Int32s, bool) {
	if loader.done && len(loader.convBuffer) == 0 {
		return nil, nil, nil, false
	}
	rowCapacity := loader.T + 1
	bos := int32(loader.tok.BOSTokenID())
	inputs := tensors.NewInt32s(loader.B, loader.T)
	targets := tensors.NewInt32s(loader.B, loader.T)
	var segments *tensors.Int32s
	if withSegments {
		segments = tensors.NewInt32s(loader.B, loader.T)
	}

	for row := 0; row < loader.B; row++ {
		loader.refill()
		if len(loader.convBuffer) == 0 {
			loader.done = true
			break
		}
		pos := 0
		segment := int32(0)

		for pos < rowCapacity {
			if len(loader.convBuffer) == 0 {
				loader.refill()
				if len(loader.convBuffer) == 0 {

					break
				}
			}
			remaining := rowCapacity - pos

			bestIdx, bestLen := -1, 0
			for index, conversation := range loader.convBuffer {
				if len(conversation.ids) <= remaining && len(conversation.ids) > bestLen {
					bestIdx, bestLen = index, len(conversation.ids)
				}
			}

			if bestIdx >= 0 {
				conversation := loader.convBuffer[bestIdx]
				loader.convBuffer = append(loader.convBuffer[:bestIdx], loader.convBuffer[bestIdx+1:]...)
				copy(loader.scratch[pos:], conversation.ids)
				copy(loader.maskScratch[pos:], conversation.mask)
				if withSegments {
					for fill := pos; fill < pos+len(conversation.ids); fill++ {
						loader.segmentScratch[fill] = segment
					}
				}
				pos += len(conversation.ids)
			} else {

				break
			}
			segment++
		}
		// Pad the remainder with BOS and mask 0. Padding is its own segment so
		// real tokens never attend to it.
		for index := pos; index < rowCapacity; index++ {
			loader.scratch[index] = bos
			loader.maskScratch[index] = 0
			if withSegments {
				loader.segmentScratch[index] = segment
			}
		}

		copy(inputs.Data[row*loader.T:], loader.scratch[:loader.T])
		for index := 0; index < loader.T; index++ {
			if loader.maskScratch[index+1] == 1 {
				targets.Data[row*loader.T+index] = loader.scratch[index+1]
			} else {
				targets.Data[row*loader.T+index] = -1
			}
		}
		if withSegments {
			copy(segments.Data[row*loader.T:], loader.segmentScratch[:loader.T])
		}
	}
	return inputs, targets, segments, true
}
