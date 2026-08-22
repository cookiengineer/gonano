// Package eval implements the evaluation metrics for gonano: bits-per-byte
// (BPB), the DCLM CORE benchmark, and ChatCORE (categorical + generative
// task evaluation).
package eval

import (
	"math"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
)

// BatchProvider yields the next validation batch (inputs, targets).
type BatchProvider func() (*tensor.Int32s, *tensor.Int32s)

// BitsPerByte evaluates the model on `steps` batches and returns the
// vocabulary-size-invariant bits-per-byte metric. tokenBytes maps each token
// id to its byte length (0 for special tokens, which are excluded).
func BitsPerByte(m *model.Transformer, batches BatchProvider, steps int, tokenBytes []int32) float32 {
	var totalNats float64
	var totalBytes int64
	for i := 0; i < steps; i++ {
		x, y := batches()
		logits := m.Forward(x, nil) // [B,T,vocab]
		flat := logits.Reshape(x.Numel(), m.Config.VocabSize)
		yflat := y.Reshape(x.Numel())
		losses, _ := tensor.CrossEntropyPerPosition(flat, yflat, -1)
		for j, target := range yflat.Data {
			if target < 0 {
				continue
			}
			nb := tokenBytes[target]
			if nb == 0 {
				continue
			}
			totalNats += float64(losses[j])
			totalBytes += int64(nb)
		}
	}
	if totalBytes == 0 {
		return float32(math.Inf(1))
	}
	return float32(totalNats / (math.Log(2) * float64(totalBytes)))
}
