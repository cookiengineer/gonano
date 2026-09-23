package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
)

// TestHeadWiseViewsShareStorage verifies that head-wise views alias both the
// parameter data and its gradient buffer of the parent weight.
func TestHeadWiseViewsShareStorage(t *testing.T) {
	weight := tensors.New(6, 4) // 3 heads of headDim 2
	for index := range weight.Data {
		weight.Data[index] = float32(index)
	}
	views := headWiseViews(weight, 3, 2)
	if len(views) != 3 {
		t.Fatalf("view count = %d, want 3", len(views))
	}
	for head, view := range views {
		if view.Shape[0] != 2 || view.Shape[1] != 4 {
			t.Fatalf("view %d shape = %v, want [2 4]", head, view.Shape)
		}
		start := head * 2 * 4
		view.Data[0] += 100
		if weight.Data[start] != float32(start)+100 {
			t.Fatalf("view %d data is not shared with the parent", head)
		}
	}

	// Gradients: EnsureGrad was called by headWiseViews, so overwrite in place
	// and then create fresh views to observe the alias.
	for index := range weight.Grad {
		weight.Grad[index] = float32(index)
	}
	views = headWiseViews(weight, 3, 2)
	views[1].Grad[0] += 5
	if weight.Grad[1*2*4] != float32(1*2*4)+5 {
		t.Fatalf("view gradient is not shared with the parent")
	}
}

// TestSetupOptimizerHeadWiseSplitsQueryKey verifies that head-wise mode replaces
// the whole query/key weights with one view per head.
func TestSetupOptimizerHeadWiseSplitsQueryKey(t *testing.T) {
	config := testConfig()
	config.NumLayer = 1

	plainModel := NewTransformer(config)
	plainModel.InitWeights(tensors.NewRNG(1))
	plainQuery := plainModel.blocks[0].attention.queryProjection.Weight
	plainKey := plainModel.blocks[0].attention.keyProjection.Weight
	plainGroups := plainModel.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	if !groupsContainPointer(plainGroups, plainQuery) || !groupsContainPointer(plainGroups, plainKey) {
		t.Fatal("plain groups should contain the whole query and key weights")
	}

	config.HeadWiseMuon = true
	headModel := NewTransformer(config)
	headModel.InitWeights(tensors.NewRNG(1))
	headQuery := headModel.blocks[0].attention.queryProjection.Weight
	headKey := headModel.blocks[0].attention.keyProjection.Weight
	headGroups := headModel.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	if groupsContainPointer(headGroups, headQuery) || groupsContainPointer(headGroups, headKey) {
		t.Fatal("head-wise groups must not contain the whole query or key weights")
	}
	// Q is split into NumHead views and K into NumKVHead views: two extra
	// parameters per layer compared with plain mode.
	plainCount := muonParameterCount(plainGroups)
	headCount := muonParameterCount(headGroups)
	if headCount != plainCount+config.NumHead+config.NumKVHead-2 {
		t.Fatalf("head-wise Muon parameter count = %d, want %d", headCount, plainCount+config.NumHead+config.NumKVHead-2)
	}
}

// TestHeadWiseMuonTrainStepFinite runs a few steps with head-wise Muon and
// checks that the optimizer actually updates the query and key weights.
func TestHeadWiseMuonTrainStepFinite(t *testing.T) {
	config := testConfig()
	config.HeadWiseMuon = true
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	optimizerImpl := optimizer.NewMuonAdamW(groups)

	indexes, targets := tinyData()
	queryBefore := append([]float32(nil), transformer.blocks[0].attention.queryProjection.Weight.Data...)
	keyBefore := append([]float32(nil), transformer.blocks[0].attention.keyProjection.Weight.Data...)

	for step := 0; step < 3; step++ {
		transformer.ZeroGrad()
		logits, context := transformer.TrainForward(indexes)
		flattened := logits.Reshape(indexes.Numel(), config.VocabSize)
		_, valid := tensors.CrossEntropyPerPosition(flattened, targets.Reshape(indexes.Numel()), -1)
		gradLogits := tensors.CrossEntropyGrad(flattened, targets.Reshape(indexes.Numel()), -1, 1/float32(valid))
		transformer.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], config.VocabSize))
		optimizerImpl.Step()
	}

	for _, parameter := range transformer.Parameters() {
		for _, value := range parameter.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("non-finite parameter after head-wise Muon step")
			}
		}
	}
	if !sliceChanged(transformer.blocks[0].attention.queryProjection.Weight.Data, queryBefore) {
		t.Fatal("query weight was not updated")
	}
	if !sliceChanged(transformer.blocks[0].attention.keyProjection.Weight.Data, keyBefore) {
		t.Fatal("key weight was not updated")
	}
}

// TestSetupOptimizerSinkhornRoutesEmbeddings verifies that enabling the
// Sinkhorn-balanced update routes the embedding table, lm_head, and value
// embeddings to it while leaving scalars on AdamW.
func TestSetupOptimizerSinkhornRoutesEmbeddings(t *testing.T) {
	config := testConfig()
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(1))
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, true)

	if kind := kindFor(groups, transformer.tokenEmbedding.Weight); kind != optimizer.KindSinkhorn {
		t.Fatalf("token embedding kind = %v, want Sinkhorn", kind)
	}
	if kind := kindFor(groups, transformer.lmHead.Weight); kind != optimizer.KindSinkhorn {
		t.Fatalf("lm_head kind = %v, want Sinkhorn", kind)
	}
	for _, embedding := range transformer.valueEmbeds {
		if kind := kindFor(groups, embedding.Weight); kind != optimizer.KindSinkhorn {
			t.Fatalf("value embedding kind = %v, want Sinkhorn", kind)
		}
	}
	if kind := kindFor(groups, transformer.residLambdas); kind != optimizer.KindAdamW {
		t.Fatalf("resid lambdas kind = %v, want AdamW", kind)
	}

	// The default keeps the historical AdamW routing.
	plain := NewTransformer(config)
	plain.InitWeights(tensors.NewRNG(1))
	plainGroups := plain.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, false)
	if kind := kindFor(plainGroups, plain.tokenEmbedding.Weight); kind != optimizer.KindAdamW {
		t.Fatalf("default token embedding kind = %v, want AdamW", kind)
	}
}

// TestSinkhornTrainStepFinite runs a few steps with the Sinkhorn-balanced
// update and checks that the embedding table is updated to finite values.
func TestSinkhornTrainStepFinite(t *testing.T) {
	config := testConfig()
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, true)
	optimizerImpl := optimizer.NewMuonAdamW(groups)

	indexes, targets := tinyData()
	embeddingBefore := append([]float32(nil), transformer.tokenEmbedding.Weight.Data...)

	for step := 0; step < 3; step++ {
		transformer.ZeroGrad()
		logits, context := transformer.TrainForward(indexes)
		flattened := logits.Reshape(indexes.Numel(), config.VocabSize)
		_, valid := tensors.CrossEntropyPerPosition(flattened, targets.Reshape(indexes.Numel()), -1)
		gradLogits := tensors.CrossEntropyGrad(flattened, targets.Reshape(indexes.Numel()), -1, 1/float32(valid))
		transformer.TrainBackward(context, gradLogits.Reshape(indexes.Shape[0], indexes.Shape[1], config.VocabSize))
		optimizerImpl.Step()
	}

	for _, parameter := range transformer.Parameters() {
		for _, value := range parameter.Data {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("non-finite parameter after Sinkhorn step")
			}
		}
	}
	if !sliceChanged(transformer.tokenEmbedding.Weight.Data, embeddingBefore) {
		t.Fatal("embedding table was not updated")
	}
}

func kindFor(groups []optimizer.ParamGroup, target *tensors.Tensor) optimizer.Kind {
	for _, group := range groups {
		for _, parameter := range group.Params {
			if parameter == target {
				return group.Kind
			}
		}
	}
	return optimizer.Kind(-1)
}

func groupsContainPointer(groups []optimizer.ParamGroup, target *tensors.Tensor) bool {
	for _, group := range groups {
		for _, parameter := range group.Params {
			if parameter == target {
				return true
			}
		}
	}
	return false
}

func muonParameterCount(groups []optimizer.ParamGroup) int {
	count := 0
	for _, group := range groups {
		if group.Kind == optimizer.KindMuon {
			count += len(group.Params)
		}
	}
	return count
}

func sliceChanged(current, before []float32) bool {
	if len(current) != len(before) {
		return true
	}
	for index := range current {
		if current[index] != before[index] {
			return true
		}
	}
	return false
}
