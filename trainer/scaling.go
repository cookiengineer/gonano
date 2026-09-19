// Package trainer implements the pretraining orchestration: compute-optimal
// hyperparameter derivation from scaling laws, learning-rate schedules, and the
// gradient-accumulation training loop. It is usable as a library (a Trainer)
// and is driven by the cmd/base_train command.
package trainer

import (
	"math"

	"github.com/cookiengineer/gonano/model"
)

// Hyperparams holds the derived training hyperparameters for a given model
// depth (the single "complexity dial", as in nanochat).
type Hyperparams struct {
	TotalBatchSize int
	NumIterations  int
	TotalTokens    int64
	BatchLRScale   float32
	WeightDecay    float32
	FlopsPerToken  float64
}

// DeriveHyperparams computes the compute-optimal training horizon, batch size,
// learning-rate scaling, and weight decay from a single dial (depth), using
// nanochat's scaling laws:
//
//   - target tokens = ratio * (transformer matrices + lm_head params)
//   - B_opt ∝ D^0.383 (Power Lines, arxiv 2505.13738)
//   - lr ∝ sqrt(B / B_ref)
//   - weight decay follows the T_epoch framework (arxiv 2405.13698)
func DeriveHyperparams(depth, vocabSize, targetParamDataRatio int, baseWeightDecay float32) Hyperparams {
	config := model.ConfigForDepth(depth, vocabSize, 64, 128, 2048, "SSSL")
	scalingParams := model.ScalingParamsForConfig(config)
	targetTokens := int64(targetParamDataRatio) * scalingParams

	referenceConfig := model.ConfigForDepth(12, vocabSize, 64, 128, 2048, "SSSL")
	referenceTokens := int64(targetParamDataRatio) * model.ScalingParamsForConfig(referenceConfig)
	referenceBatchSize := float64(1 << 19) // 524288 tokens

	tokenRatio := float64(targetTokens) / float64(referenceTokens)
	predictedBatchSize := referenceBatchSize * math.Pow(tokenRatio, 0.383)
	totalBatchSize := int(math.Pow(2, math.Round(math.Log2(predictedBatchSize))))
	if totalBatchSize < 1 {
		totalBatchSize = 1
	}

	batchLRScale := float32(math.Sqrt(float64(totalBatchSize) / referenceBatchSize))
	weightDecay := baseWeightDecay * float32(math.Sqrt(float64(totalBatchSize)/referenceBatchSize)) * float32(float64(referenceTokens)/float64(targetTokens))

	numIterations := int(targetTokens / int64(totalBatchSize))
	if numIterations < 1 {
		numIterations = 1
	}

	return Hyperparams{
		TotalBatchSize: totalBatchSize,
		NumIterations:  numIterations,
		TotalTokens:    int64(totalBatchSize) * int64(numIterations),
		BatchLRScale:   batchLRScale,
		WeightDecay:    weightDecay,
		FlopsPerToken:  model.EstimateFlopsPerTokenForConfig(config),
	}
}
