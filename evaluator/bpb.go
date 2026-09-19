// Package evaluator implements the evaluation metrics for gonano: bits-per-byte
// (BPB), the DCLM CORE benchmark, and ChatCORE (categorical + generative
// task evaluation).
package evaluator

import (
	"math"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// BatchProvider yields the next validation batch (inputs, targets).
type BatchProvider func() (*tensors.Int32s, *tensors.Int32s)

// BitsPerByte evaluates the model on `steps` batches and returns the
// vocabulary-size-invariant bits-per-byte metric. tokenBytes maps each token
// id to its byte length (0 for special tokens, which are excluded).
func BitsPerByte(transformer *model.Transformer, batches BatchProvider, steps int, tokenBytes []int32) float32 {
	var totalNats float64
	var totalBytes int64
	for index := 0; index < steps; index++ {
		inputs, targets := batches()
		logits := transformer.Forward(inputs, nil) // [B,T,vocab]
		flat := logits.Reshape(inputs.Numel(), transformer.Config.VocabSize)
		targetsFlat := targets.Reshape(inputs.Numel())
		losses, _ := tensors.CrossEntropyPerPosition(flat, targetsFlat, -1)
		for position, target := range targetsFlat.Data {
			if target < 0 {
				continue
			}
			numBytes := tokenBytes[target]
			if numBytes == 0 {
				continue
			}
			totalNats += float64(losses[position])
			totalBytes += int64(numBytes)
		}
	}
	if totalBytes == 0 {
		return float32(math.Inf(1))
	}
	return float32(totalNats / (math.Log(2) * float64(totalBytes)))
}
