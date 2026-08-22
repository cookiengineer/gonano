package data

import (
	"github.com/cookiengineer/gonano/tensor"
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
}

// NewPretrainLoader builds a pretraining dataloader. provider yields batches
// of raw documents (already split into documents; tokenization is done here).
func NewPretrainLoader(tok *tokenizer.Tokenizer, B, T int, provider DocProvider, bufferSize int) *PretrainLoader {
	if bufferSize <= 0 {
		bufferSize = 1000
	}
	return &PretrainLoader{
		tok:        tok,
		B:          B,
		T:          T,
		bufferSize: bufferSize,
		provider:   provider,
		scratch:    make([]int32, T+1),
	}
}

// refill pulls more documents until the buffer has at least bufferSize entries.
func (l *PretrainLoader) refill() {
	bos := l.tok.BOSTokenID()
	for len(l.docBuffer) < l.bufferSize {
		docs, state := l.provider()
		l.state = state
		for _, d := range docs {
			ids := l.tok.Encode(d)
			row := make([]int32, 0, len(ids)+1)
			row = append(row, int32(bos))
			row = append(row, toInt32(ids)...)
			l.docBuffer = append(l.docBuffer, row)
		}
	}
}

func toInt32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}

// Next produces the next training batch: inputs of shape [B,T] and targets of
// shape [B,T] (inputs shifted by one). It also returns the dataset position
// for checkpointing.
func (l *PretrainLoader) Next() (*tensor.Int32s, *tensor.Int32s, State) {
	rowCapacity := l.T + 1
	inputs := tensor.NewInt32s(l.B, l.T)
	targets := tensor.NewInt32s(l.B, l.T)

	for row := 0; row < l.B; row++ {
		pos := 0
		for pos < rowCapacity {
			for len(l.docBuffer) < l.bufferSize {
				l.refill()
			}
			remaining := rowCapacity - pos

			bestIdx, bestLen := -1, 0
			for i, doc := range l.docBuffer {
				if len(doc) <= remaining && len(doc) > bestLen {
					bestIdx, bestLen = i, len(doc)
				}
			}

			if bestIdx >= 0 {
				doc := l.docBuffer[bestIdx]
				l.docBuffer = append(l.docBuffer[:bestIdx], l.docBuffer[bestIdx+1:]...)
				copy(l.scratch[pos:], doc)
				pos += len(doc)
			} else {
				shortest := 0
				for i := range l.docBuffer {
					if len(l.docBuffer[i]) < len(l.docBuffer[shortest]) {
						shortest = i
					}
				}
				doc := l.docBuffer[shortest]
				l.docBuffer = append(l.docBuffer[:shortest], l.docBuffer[shortest+1:]...)
				copy(l.scratch[pos:], doc[:remaining])
				pos += remaining
			}
		}
		copy(inputs.Data[row*l.T:], l.scratch[:l.T])
		copy(targets.Data[row*l.T:], l.scratch[1:l.T+1])
	}
	return inputs, targets, l.state
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

	convBuffer []convWithMask
	done       bool
	scratch    []int32
	maskScratch []int32
}

// NewSFTLoader builds an SFT dataloader over the given conversation provider.
func NewSFTLoader(tok *tokenizer.Tokenizer, B, T int, provider ConvProvider, bufferSize int) *SFTLoader {
	if bufferSize <= 0 {
		bufferSize = 100
	}
	return &SFTLoader{
		tok:         tok,
		B:           B,
		T:           T,
		bufferSize:  bufferSize,
		provider:    provider,
		scratch:     make([]int32, T+1),
		maskScratch: make([]int32, T+1),
	}
}

// refill pulls more conversations until the buffer has enough entries.
func (l *SFTLoader) refill() {
	for len(l.convBuffer) < l.bufferSize {
		convs, ok := l.provider()
		if !ok {
			l.done = true
			return
		}
		for _, c := range convs {
			ids, mask := l.tok.RenderConversation(c, 1<<30)
			l.convBuffer = append(l.convBuffer, convWithMask{ids: toInt32(ids), mask: toInt32(mask)})
		}
	}
}

// Next produces the next SFT batch of shape [B,T]. Targets have -1 at masked
// (non-assistant) and padded positions. ok is false when the provider is
// exhausted and the buffer is empty.
func (l *SFTLoader) Next() (*tensor.Int32s, *tensor.Int32s, bool) {
	if l.done && len(l.convBuffer) == 0 {
		return nil, nil, false
	}
	rowCapacity := l.T + 1
	bos := int32(l.tok.BOSTokenID())
	inputs := tensor.NewInt32s(l.B, l.T)
	targets := tensor.NewInt32s(l.B, l.T)

	for row := 0; row < l.B; row++ {
		l.refill()
		if len(l.convBuffer) == 0 {
			l.done = true
			break
		}
		pos := 0

		for pos < rowCapacity {
			if len(l.convBuffer) == 0 {
				l.refill()
				if len(l.convBuffer) == 0 {

					break
				}
			}
			remaining := rowCapacity - pos

			bestIdx, bestLen := -1, 0
			for i, conv := range l.convBuffer {
				if len(conv.ids) <= remaining && len(conv.ids) > bestLen {
					bestIdx, bestLen = i, len(conv.ids)
				}
			}

			if bestIdx >= 0 {
				conv := l.convBuffer[bestIdx]
				l.convBuffer = append(l.convBuffer[:bestIdx], l.convBuffer[bestIdx+1:]...)
				copy(l.scratch[pos:], conv.ids)
				copy(l.maskScratch[pos:], conv.mask)
				pos += len(conv.ids)
			} else {

				break
			}
		}
		// Pad the remainder with BOS and mask 0.
		for i := pos; i < rowCapacity; i++ {
			l.scratch[i] = bos
			l.maskScratch[i] = 0
		}

		copy(inputs.Data[row*l.T:], l.scratch[:l.T])
		for i := 0; i < l.T; i++ {
			if l.maskScratch[i+1] == 1 {
				targets.Data[row*l.T+i] = l.scratch[i+1]
			} else {
				targets.Data[row*l.T+i] = -1
			}
		}
	}
	return inputs, targets, true
}
