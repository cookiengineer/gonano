package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// TestSWAMergeMatchesConcatenatedAttention verifies that merging two attention
// branches through the shared-softmax log-sum-exp equals a single attention over
// the concatenation of their keys, both forward and backward. This is the
// numerical contract behind the compressed-global + sliding-window merge.
func TestSWAMergeMatchesConcatenatedAttention(t *testing.T) {
	queryLength, headDim := 5, 8
	keysA, keysB := 6, 4
	rng := tensors.NewRNG(7)
	random := func(count int) []float32 {
		values := make([]float32, count)
		for index := range values {
			values[index] = rng.NormFloat32()
		}
		return values
	}

	query := random(queryLength * headDim)
	keyA, valueA := random(keysA*headDim), random(keysA*headDim)
	keyB, valueB := random(keysB*headDim), random(keysB*headDim)
	// PositionOffset past every key disables the causal/window mask so the two
	// branches and the concatenation attend to exactly the same positions.
	offset := queryLength + keysA + keysB

	branchForward := func(key, value []float32, keyLength int) ([]float32, []float32) {
		output := make([]float32, queryLength*headDim)
		logSumExp := make([]float32, queryLength)
		tensors.AttentionForward(query, key, value, output, logSumExp, queryLength, keyLength, headDim, offset, -1)
		return output, logSumExp
	}
	outputA, logSumExpA := branchForward(keyA, valueA, keysA)
	outputB, logSumExpB := branchForward(keyB, valueB, keysB)

	mergedOutput := tensors.New(1, 1, queryLength, headDim)
	mergedLogSumExp := tensors.New(1, 1, queryLength)
	mergeAttentionBranches(mergedOutput, mergedLogSumExp,
		tensors.NewWithData([]int{1, 1, queryLength, headDim}, outputA),
		tensors.NewWithData([]int{1, 1, queryLength}, logSumExpA),
		tensors.NewWithData([]int{1, 1, queryLength, headDim}, outputB),
		tensors.NewWithData([]int{1, 1, queryLength}, logSumExpB),
		headDim)

	key := append(append([]float32(nil), keyA...), keyB...)
	value := append(append([]float32(nil), valueA...), valueB...)
	referenceOutput := make([]float32, queryLength*headDim)
	referenceLogSumExp := make([]float32, queryLength)
	tensors.AttentionForward(query, key, value, referenceOutput, referenceLogSumExp, queryLength, keysA+keysB, headDim, offset, -1)

	assertClose := func(name string, got, want float32) {
		scale := float32(math.Abs(float64(want))) + 1e-4
		if diff := float32(math.Abs(float64(got - want))); diff > 1e-3*scale {
			t.Fatalf("%s mismatch: got %v want %v", name, got, want)
		}
	}
	for index := range referenceOutput {
		assertClose("merged output", mergedOutput.Data[index], referenceOutput[index])
	}
	for index := range referenceLogSumExp {
		assertClose("merged logsumexp", mergedLogSumExp.Data[index], referenceLogSumExp[index])
	}

	// Backward: each branch uses the merged log-sum-exp and the merged row
	// correction dO . O, and their combined gradients must match the single
	// attention over the concatenated keys.
	outputGradient := random(queryLength * headDim)
	correction := make([]float32, queryLength)
	for row := 0; row < queryLength; row++ {
		var dot float64
		for dimension := 0; dimension < headDim; dimension++ {
			dot += float64(outputGradient[row*headDim+dimension]) * float64(mergedOutput.Data[row*headDim+dimension])
		}
		correction[row] = float32(dot)
	}

	gradientQueryA := make([]float32, queryLength*headDim)
	gradientKeyA := make([]float32, keysA*headDim)
	gradientValueA := make([]float32, keysA*headDim)
	tensors.AttentionBackwardCorrected(query, keyA, valueA, outputA, outputGradient, mergedLogSumExp.Data, correction,
		gradientQueryA, gradientKeyA, gradientValueA, queryLength, keysA, headDim, offset, -1)

	gradientQueryB := make([]float32, queryLength*headDim)
	gradientKeyB := make([]float32, keysB*headDim)
	gradientValueB := make([]float32, keysB*headDim)
	tensors.AttentionBackwardCorrected(query, keyB, valueB, outputB, outputGradient, mergedLogSumExp.Data, correction,
		gradientQueryB, gradientKeyB, gradientValueB, queryLength, keysB, headDim, offset, -1)

	referenceQuery := make([]float32, queryLength*headDim)
	referenceKey := make([]float32, (keysA+keysB)*headDim)
	referenceValue := make([]float32, (keysA+keysB)*headDim)
	tensors.AttentionBackward(query, key, value, referenceOutput, outputGradient, referenceLogSumExp,
		referenceQuery, referenceKey, referenceValue, queryLength, keysA+keysB, headDim, offset, -1)

	for index := range referenceQuery {
		assertClose("query gradient", gradientQueryA[index]+gradientQueryB[index], referenceQuery[index])
	}
	for index := range gradientKeyA {
		assertClose("key A gradient", gradientKeyA[index], referenceKey[index])
	}
	for index := range gradientKeyB {
		assertClose("key B gradient", gradientKeyB[index], referenceKey[keysA*headDim+index])
	}
	for index := range gradientValueA {
		assertClose("value A gradient", gradientValueA[index], referenceValue[index])
	}
	for index := range gradientValueB {
		assertClose("value B gradient", gradientValueB[index], referenceValue[keysA*headDim+index])
	}
}

// tinySWAModel is a compressed model with the local sliding-window branch
// enabled on every layer.
func tinySWAModel() *Transformer {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 3,
	}
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += perturb.NormFloat32() * 0.1
		}
	}
	return model
}

// TestBackpropDirectionalGradientCheckSWA verifies the merged global/local
// attention backward through a full compressed model.
func TestBackpropDirectionalGradientCheckSWA(t *testing.T) {
	runDirectionalGradientCheck(t, tinySWAModel())
}

func TestSWACompressedTrainStepFinite(t *testing.T) {
	for _, sparseTopK := range []int{0, 2} {
		config := Config{
			SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
			EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2, SWAWindow: 3,
			SparseTopK: sparseTopK, IndexerDim: 4,
		}
		transformer := NewTransformer(config)
		transformer.InitWeights(tensors.NewRNG(42))
		indexes, targets := tinyData()

		transformer.ZeroGrad()
		logits, context := transformer.TrainForward(indexes)
		for _, value := range logits.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("sparseTopK=%d: non-finite logit %v", sparseTopK, value)
			}
		}
		flattened := logits.Reshape(indexes.Numel(), config.VocabSize)
		_, valid := tensors.CrossEntropyPerPosition(flattened, targets.Reshape(indexes.Numel()), -1)
		gradLogits := tensors.CrossEntropyGrad(flattened, targets.Reshape(indexes.Numel()), -1, 1/float32(valid))
		transformer.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], config.VocabSize))
		for _, parameter := range transformer.Parameters() {
			for _, value := range parameter.Grad {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("sparseTopK=%d: non-finite gradient", sparseTopK)
				}
			}
		}
	}
}

// TestSWAFlopsAccounting checks that the local branch adds cost and that
// compression reduces the global attention cost relative to uncompressed
// attention.
func TestSWAFlopsAccounting(t *testing.T) {
	base := testConfig()
	base.SequenceLen = 128
	base.CompressionRatio = 4
	withoutSWA := NewTransformer(base)
	withSWAConfig := base
	withSWAConfig.SWAWindow = 8
	withSWA := NewTransformer(withSWAConfig)
	if withSWA.EstimateDecodeFlops(64) <= withoutSWA.EstimateDecodeFlops(64) {
		t.Fatal("SWA must add decode FLOPs")
	}
	if withSWA.EstimatePrefillFlops(64) <= withoutSWA.EstimatePrefillFlops(64) {
		t.Fatal("SWA must add prefill FLOPs")
	}

	uncompressedConfig := base
	uncompressedConfig.CompressionRatio = 0
	uncompressed := NewTransformer(uncompressedConfig)
	if withoutSWA.EstimateDecodeFlops(64) >= uncompressed.EstimateDecodeFlops(64) {
		t.Fatal("compressed global attention should reduce decode FLOPs")
	}
}

// TestSWABranchIsActive verifies that enabling the window changes the
// compressed attention output for the same weights, confirming the local branch
// participates in the forward pass.
func TestSWABranchIsActive(t *testing.T) {
	base := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", CompressionRatio: 2,
	}
	withSWAConfig := base
	withSWAConfig.SWAWindow = 3
	withoutSWA := NewTransformer(base)
	withoutSWA.InitWeights(tensors.NewRNG(42))
	withSWA := NewTransformer(withSWAConfig)
	withSWA.InitWeights(tensors.NewRNG(42))
	// The attention output projection is zero-initialized, so perturb it for
	// both models identically to make the attention reach the logits.
	for _, transformer := range []*Transformer{withoutSWA, withSWA} {
		perturb := tensors.NewRNG(321)
		for _, block := range transformer.blocks {
			for elementIndex := range block.attention.outputProjection.Weight.Data {
				block.attention.outputProjection.Weight.Data[elementIndex] += perturb.NormFloat32() * 0.1
			}
		}
	}

	indexes := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 5, 2, 8})
	firstLogits, _ := withoutSWA.TrainForward(indexes)
	secondLogits, _ := withSWA.TrainForward(indexes)

	identical := true
	for index := range firstLogits.Data {
		if firstLogits.Data[index] != secondLogits.Data[index] {
			identical = false
			break
		}
	}
	if identical {
		t.Fatal("enabling the sliding-window branch did not change the output")
	}
}
