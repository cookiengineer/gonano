package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/kernels/scalar"
	"github.com/cookiengineer/gonano/tensors"
)

// moeTestConfig is a tiny MoE config: 4 routed experts, top-2, with fine-grained
// expert and shared widths.
func moeTestConfig() Config {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
		NumExperts: 4, NumExpertsPerToken: 2,
		ExpertHiddenDim: 16, SharedExpertHiddenDim: 16,
	}
	config.ApplyMoEDefaults()
	return config
}

// moeTestModel builds a perturbed (non-zero projection) MoE model so gradient
// checks see signal through every layer.
func moeTestModel() *Transformer {
	model := NewTransformer(moeTestConfig())
	model.InitWeights(tensors.NewRNG(42))
	perturb := tensors.NewRNG(123)
	for _, parameter := range model.Parameters() {
		for index := range parameter.Data {
			parameter.Data[index] += perturb.NormFloat32() * 0.1
		}
	}
	return model
}

func TestMoEConfigDefaults(t *testing.T) {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 4, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L", NumExperts: 8,
	}
	config.ApplyMoEDefaults()
	if config.NumExpertsPerToken != 2 {
		t.Fatalf("NumExpertsPerToken = %d, want 2", config.NumExpertsPerToken)
	}
	if config.ExpertHiddenDim != config.EmbedDim {
		t.Fatalf("ExpertHiddenDim = %d, want %d", config.ExpertHiddenDim, config.EmbedDim)
	}
	if config.SharedExpertHiddenDim != config.ExpertHiddenDim {
		t.Fatalf("SharedExpertHiddenDim = %d, want %d", config.SharedExpertHiddenDim, config.ExpertHiddenDim)
	}
	if config.MoEEffectiveClamp() != 10 {
		t.Fatalf("MoEEffectiveClamp = %v, want 10", config.MoEEffectiveClamp())
	}
	if config.MoEEffectiveScale() != 1 {
		t.Fatalf("MoEEffectiveScale = %v, want 1", config.MoEEffectiveScale())
	}
	if config.MoERouterBiasUpdate() != 0.001 {
		t.Fatalf("MoERouterBiasUpdate = %v, want 0.001", config.MoERouterBiasUpdate())
	}
	if !config.MoEEnabled() {
		t.Fatal("MoEEnabled should be true")
	}
	if MoEExpertsForDepth(2) != 8 || MoEExpertsForDepth(12) != 24 {
		t.Fatalf("MoEExpertsForDepth = %d, %d", MoEExpertsForDepth(2), MoEExpertsForDepth(12))
	}
}

func TestMoEConfigValidation(t *testing.T) {
	assertPanics := func(name string, mutate func(*Config)) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("%s: expected Validate to panic", name)
			}
		}()
		config := moeTestConfig()
		mutate(&config)
		config.Validate()
	}
	assertPanics("topk exceeds experts", func(config *Config) { config.NumExpertsPerToken = 99 })
	assertPanics("negative experts", func(config *Config) { config.NumExperts = -1 })
	assertPanics("experts without hyperparams", func(config *Config) {
		config.NumExperts = 0
		config.NumExpertsPerToken = 2
	})
	assertPanics("negative hidden", func(config *Config) { config.ExpertHiddenDim = -1 })

	// A valid MoE config must not panic.
	moeTestConfig().Validate()
}

func TestMoEForwardFiniteAndShapes(t *testing.T) {
	model := moeTestModel()
	indexes := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	logits := model.Forward(indexes, nil)
	wantShape := []int{1, 4, model.Config.VocabSize}
	if len(logits.Shape) != 3 || logits.Shape[0] != wantShape[0] || logits.Shape[1] != wantShape[1] || logits.Shape[2] != wantShape[2] {
		t.Fatalf("logits shape = %v, want %v", logits.Shape, wantShape)
	}
	for index, value := range logits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite logit at %d", index)
		}
	}
	for _, block := range model.blocks {
		if block.moe == nil || block.mlp == nil {
			t.Fatal("MoE block should have both a routed MoE and a shared expert")
		}
		if block.mlp.gateProjection == nil {
			t.Fatal("shared expert should use the SwiGLU gate projection")
		}
	}
}

func TestMoENamedParameters(t *testing.T) {
	model := moeTestModel()
	names := model.NamedParameters()
	expected := []string{
		"transformer.h.0.moe.router.weight",
		"transformer.h.0.moe.router_bias",
		"transformer.h.0.moe.experts.gate.weight",
		"transformer.h.0.moe.experts.up.weight",
		"transformer.h.0.moe.experts.down.weight",
		"transformer.h.1.moe.router.weight",
		"transformer.h.0.mlp.c_gate.weight",
	}
	for _, name := range expected {
		if names[name] == nil {
			t.Errorf("missing parameter %q", name)
		}
	}
	if names["transformer.h.0.moe.experts.gate.weight"].Shape[0] != model.Config.NumExperts {
		t.Fatalf("packed gate weight should have one slice per expert")
	}
}

func TestMoEActiveParameterAccounting(t *testing.T) {
	model := moeTestModel()
	if model.ActiveMatmulParams() >= model.MatmulParams() {
		t.Fatalf("active params %d must be fewer than total %d", model.ActiveMatmulParams(), model.MatmulParams())
	}
	if model.WeightReadBytes() != model.ActiveMatmulParams()*4 {
		t.Fatalf("WeightReadBytes should reflect active experts")
	}
	if model.TotalParams() <= model.MatmulParams() {
		t.Fatal("TotalParams should include embeddings/router bias")
	}
	dense := NewTransformer(testConfig())
	if dense.ActiveMatmulParams() != dense.MatmulParams() {
		t.Fatal("dense model active and total params must match")
	}
}

// TestMoERouterSelectsDeterministically forces a strong router preference and
// checks that every token routes to the same expert.
func TestMoERouterSelectsDeterministically(t *testing.T) {
	config := moeTestConfig()
	config.NumExpertsPerToken = 1
	config.ExpertHiddenDim = 16
	config.SharedExpertHiddenDim = 16
	config.Validate()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(1))
	for _, block := range model.blocks {
		// Expert 0 gets a large positive affinity, expert 1 a large negative
		// one, so the top-1 selection is unambiguous.
		block.moe.router.weight.Weight.Set(0)
		for expert := 0; expert < block.moe.NumExperts; expert++ {
			value := float32(-10)
			if expert == 0 {
				value = 10
			}
			for column := 0; column < config.EmbedDim; column++ {
				block.moe.router.weight.Weight.Set2(expert, column, value)
			}
		}
	}
	indexes := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	input := tensors.New(int(indexes.Numel()), config.EmbedDim)
	for index := range input.Data {
		input.Data[index] = float32(index%7) * 0.1
	}
	_, context := model.blocks[0].moe.forwardTraining(input, 0)
	if len(context.experts) != 1 || context.experts[0].id != 0 {
		t.Fatalf("expected all tokens routed to expert 0, got %d active experts", len(context.experts))
	}
}

// TestMoERouterBiasBalancing checks the auxiliary-loss-free update: an
// overloaded expert's correction bias decreases and an idle expert's increases.
func TestMoERouterBiasBalancing(t *testing.T) {
	config := moeTestConfig()
	config.NumExpertsPerToken = 1
	config.Validate()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(2))

	// Simulate an optimizer window in which expert 0 received all tokens.
	block := model.blocks[0]
	block.moe.loadCounts[0] = 12
	block.moe.loadTotal = 12

	before := block.moe.router.bias.Data[0]
	model.UpdateRouterBias()
	after := block.moe.router.bias.Data[0]
	if !(after < before) {
		t.Fatalf("overloaded expert bias should decrease: before=%v after=%v", before, after)
	}
	if block.moe.router.bias.Data[1] <= 0 {
		t.Fatalf("idle expert bias should increase above zero, got %v", block.moe.router.bias.Data[1])
	}
	if block.moe.loadTotal != 0 {
		t.Fatal("load counters should reset after the update")
	}
}

// TestMoEForwardMatchesTrainForward verifies inference and training use the
// same routing and expert math.
func TestMoEForwardMatchesTrainForward(t *testing.T) {
	model := moeTestModel()
	indexes, _ := tinyData()
	inference := model.Forward(indexes, nil)
	training, _ := model.TrainForward(indexes)
	if len(inference.Data) != len(training.Data) {
		t.Fatalf("length mismatch %d != %d", len(inference.Data), len(training.Data))
	}
	for index := range inference.Data {
		if math.Abs(float64(inference.Data[index]-training.Data[index])) > 1e-4 {
			t.Fatalf("logit %d differs: inference=%v training=%v", index, inference.Data[index], training.Data[index])
		}
	}
}

// TestMoEDecodeMatchesPrefill checks that a batched MoE forward matches
// token-by-token decoding against a KV cache.
func TestMoEDecodeMatchesPrefill(t *testing.T) {
	model := moeTestModel()
	sequence := []int32{1, 5, 2, 8, 3, 7, 4, 6}

	sequentialCache := NewKVBuffer(1, 16, model.Config.NumLayer, model.Config.NumKVHead, model.Config.HeadDim())
	var sequential []float32
	for index, token := range sequence {
		ids := tensors.NewInt32sWithData([]int{1, 1}, []int32{token})
		logits := model.Forward(ids, sequentialCache)
		if index == len(sequence)-1 {
			sequential = append([]float32(nil), logits.Data...)
		}
	}
	batchedCache := NewKVBuffer(1, 16, model.Config.NumLayer, model.Config.NumKVHead, model.Config.HeadDim())
	batched := model.Forward(tensors.NewInt32sWithData([]int{1, len(sequence)}, sequence), batchedCache)
	batchedLast := batched.Data[(len(sequence)-1)*model.Config.VocabSize:]
	assertLogitsClose(t, sequential, batchedLast)
}

// TestMoEGradientCheck validates the analytic MoE backprop (router gate,
// packed experts, and the SwiGLU shared expert) against finite differences.
// The router correction bias is excluded because expert selection is
// piecewise-constant in it; its update is covered by the balancing test.
func TestMoEGradientCheck(t *testing.T) {
	model := moeTestModel()
	indexes, targets := tinyData()
	analyticGrads(model, indexes, targets)

	parameters := make([]*tensors.Tensor, 0)
	for _, block := range model.blocks {
		if block.moe == nil {
			continue
		}
		parameters = append(parameters, block.mlp.parameters()...)
		parameters = append(parameters, block.moe.router.weight.Weight)
		parameters = append(parameters, block.moe.gateWeight, block.moe.upWeight, block.moe.downWeight)
	}

	rng := tensors.NewRNG(999)
	directions := make([][]float32, len(parameters))
	var analytic float64
	for parameterIndex, parameter := range parameters {
		directions[parameterIndex] = make([]float32, parameter.Numel())
		for elementIndex := range parameter.Data {
			value := rng.NormFloat32()
			directions[parameterIndex][elementIndex] = value
			analytic += float64(parameter.Grad[elementIndex]) * float64(value)
		}
	}

	epsilon := float32(1e-3)
	for parameterIndex, parameter := range parameters {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += epsilon * directions[parameterIndex][elementIndex]
		}
	}
	lossPlus := meanLoss(model, indexes, targets)
	for parameterIndex, parameter := range parameters {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] -= 2 * epsilon * directions[parameterIndex][elementIndex]
		}
	}
	lossMinus := meanLoss(model, indexes, targets)
	for parameterIndex, parameter := range parameters {
		for elementIndex := range parameter.Data {
			parameter.Data[elementIndex] += epsilon * directions[parameterIndex][elementIndex]
		}
	}
	numeric := float64(lossPlus-lossMinus) / (2 * float64(epsilon))

	scale := math.Abs(analytic) + 1e-6
	if math.Abs(numeric-analytic) > 5e-2*scale {
		t.Fatalf("MoE directional gradient mismatch: analytic=%v numeric=%v", analytic, numeric)
	}
}

// TestMoEPerElementGradientCheck checks a representative element of each MoE
// weight against a per-element finite difference, catching per-expert indexing
// bugs that an aggregate projection could mask.
func TestMoEPerElementGradientCheck(t *testing.T) {
	model := moeTestModel()
	indexes, targets := tinyData()
	analyticGrads(model, indexes, targets)

	block := model.blocks[0]
	checks := []struct {
		name         string
		parameter    *tensors.Tensor
		elementIndex int
	}{
		{"router", block.moe.router.weight.Weight, 5},
		{"expert0.gate", block.moe.gateWeight, 3},
		{"expert1.gate", block.moe.gateWeight, block.moe.Hidden*block.moe.Dim + 3},
		{"expert0.up", block.moe.upWeight, 7},
		{"expert2.down", block.moe.downWeight, 2*block.moe.Dim*block.moe.Hidden + 4},
		{"shared.c_gate", block.mlp.gateProjection.Weight, 6},
		{"shared.c_fc", block.mlp.inputProjection.Weight, 6},
		{"shared.c_proj", block.mlp.outputProjection.Weight, 6},
	}

	epsilon := float32(1e-3)
	for _, check := range checks {
		analytic := check.parameter.Grad[check.elementIndex]
		original := check.parameter.Data[check.elementIndex]
		check.parameter.Data[check.elementIndex] = original + epsilon
		lossPlus := meanLoss(model, indexes, targets)
		check.parameter.Data[check.elementIndex] = original - epsilon
		lossMinus := meanLoss(model, indexes, targets)
		check.parameter.Data[check.elementIndex] = original
		numeric := (lossPlus - lossMinus) / (2 * epsilon)

		scale := math.Abs(float64(analytic))
		if math.Abs(float64(numeric)) > scale {
			scale = math.Abs(float64(numeric))
		}
		scale = math.Max(scale, 1e-3)
		if math.Abs(float64(numeric-analytic)) > 0.5*scale {
			t.Errorf("%s[%d]: analytic=%v numeric=%v", check.name, check.elementIndex, analytic, numeric)
		}
	}
}

// TestMoEConfigAccounting checks the config-only scaling and FLOP formulas
// distinguish total parameters (all experts) from per-token active compute
// (top-k experts).
func TestMoEConfigAccounting(t *testing.T) {
	base := moeTestConfig()

	moreExperts := base
	moreExperts.NumExperts = 8
	moreExperts.NumExpertsPerToken = 2
	if ScalingParamsForConfig(moreExperts) <= ScalingParamsForConfig(base) {
		t.Fatal("more experts should increase total scaling parameters")
	}

	allActive := base
	allActive.NumExpertsPerToken = allActive.NumExperts
	if EstimateFlopsPerTokenForConfig(allActive) <= EstimateFlopsPerTokenForConfig(base) {
		t.Fatal("activating all experts must cost more than the top-2 subset")
	}
	if ScalingParamsForConfig(allActive) != ScalingParamsForConfig(base) {
		t.Fatal("changing only the top-k must not change total scaling parameters")
	}
}

// TestMoEComposesWithGQA verifies the MoE feed-forward composes with grouped
// query attention.
func TestMoEComposesWithGQA(t *testing.T) {
	config := moeTestConfig()
	config.NumKVHead = 1
	config.Validate()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(7))
	indexes := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	logits := model.Forward(indexes, nil)
	for index, value := range logits.Data {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("non-finite logit at %d", index)
		}
	}
	// Backprop must also be finite with grouped KV heads.
	analyticGrads(model, indexes, indexes)
	for _, block := range model.blocks {
		if block.moe.gateWeight.Grad == nil {
			t.Fatal("expert gradient should be allocated")
		}
	}
}

// TestMoEScalarBackendMatchesSIMD exercises the fused MoE kernels through the
// model on the portable scalar backend and compares against the SIMD backend.
func TestMoEScalarBackendMatchesSIMD(t *testing.T) {
	model := moeTestModel()
	indexes, targets := tinyData()
	simdLogits := model.Forward(indexes, nil)

	previous := tensors.KernelBackend()
	tensors.UseKernelBackend(scalar.New())
	t.Cleanup(func() { tensors.UseKernelBackend(previous) })

	scalarLogits := model.Forward(indexes, nil)
	for index := range simdLogits.Data {
		if math.Abs(float64(simdLogits.Data[index]-scalarLogits.Data[index])) > 1e-3 {
			t.Fatalf("logit %d differs between backends: simd=%v scalar=%v", index, simdLogits.Data[index], scalarLogits.Data[index])
		}
	}

	// Backprop through the scalar grouped kernels must stay finite.
	analyticGrads(model, indexes, targets)
	for _, block := range model.blocks {
		for _, weight := range []*tensors.Tensor{block.moe.gateWeight, block.moe.upWeight, block.moe.downWeight} {
			if weight.Grad == nil {
				t.Fatal("expert gradient should be allocated on the scalar backend")
			}
			for _, value := range weight.Grad {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatal("non-finite expert gradient on the scalar backend")
				}
			}
		}
	}
}

// TestMoEComposesWithCompression verifies the MoE feed-forward composes with
// dense KV compression (the non-differentiable sparse selection is excluded).
func TestMoEComposesWithCompression(t *testing.T) {
	config := moeTestConfig()
	config.CompressionRatio = 2
	config.Validate()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(11))
	indexes, targets := tinyData()

	cache := NewKVBuffer(1, 16, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableCompression(config.Compression(), config.EmbedDim, config.NumKVHead*config.HeadDim(), 4)
	logits := model.Forward(indexes, cache)
	if logits.Numel() != indexes.Numel()*config.VocabSize {
		t.Fatalf("logits numel = %d", logits.Numel())
	}
	analyticGrads(model, indexes, targets)
	for _, block := range model.blocks {
		if block.moe.gateWeight.Grad == nil || block.moe.downWeight.Grad == nil {
			t.Fatal("expert gradients should flow under compression")
		}
	}
}

// TestMoEComposesWithCED verifies the MoE feed-forward composes with the
// causal encoder-decoder split (which changes the attention path, not the FFN).
func TestMoEComposesWithCED(t *testing.T) {
	config := moeTestConfig()
	config.CompressionRatio = 2
	config.SWAWindow = 3
	config.CED = true
	config.Validate()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(12))
	indexes, targets := tinyData()

	cache := NewKVBuffer(1, 16, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableCEDCache(cache, config, 16)
	// PrefillCED returns logits only for the replayed window (the last
	// SWAWindow tokens), not the full prompt.
	logits := model.PrefillCED(indexes, cache)
	if logits.Shape[0] != indexes.Shape[0] || logits.Shape[1] != config.SWAWindowSize() || logits.Shape[2] != config.VocabSize {
		t.Fatalf("CED logits shape = %v, want [%d %d %d]", logits.Shape, indexes.Shape[0], config.SWAWindowSize(), config.VocabSize)
	}
	analyticGrads(model, indexes, targets)
}

// TestMoEComposesWithMLA verifies the MoE feed-forward composes with absorbed
// Multi-head Latent Attention.
func TestMoEComposesWithMLA(t *testing.T) {
	config := moeTestConfig()
	config.MLALatent = 8
	config.MLARotaryDims = 8
	config.Validate()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(13))
	indexes, targets := tinyData()

	cache := NewKVBuffer(1, 16, config.NumLayer, config.NumKVHead, config.HeadDim())
	cache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
	if logits := model.Forward(indexes, cache); logits.Numel() != indexes.Numel()*config.VocabSize {
		t.Fatalf("MLA logits numel = %d", logits.Numel())
	}
	// A cache-less forward allocates an MLA scratch cache internally.
	analyticGrads(model, indexes, targets)
}

// TestMoEBalanceWeightDefault checks the effective sequence-level balance weight
// semantics: zero selects the paper default, negative disables, positive
// overrides.
func TestMoEBalanceWeightDefault(t *testing.T) {
	config := moeTestConfig()
	if config.MoEBalanceLossWeight() != 1e-4 {
		t.Fatalf("default balance weight = %v, want 1e-4", config.MoEBalanceLossWeight())
	}
	config.MoEBalanceWeight = -1
	if config.MoEBalanceLossWeight() != 0 {
		t.Fatalf("negative balance weight should disable, got %v", config.MoEBalanceLossWeight())
	}
	config.MoEBalanceWeight = 0.5
	if config.MoEBalanceLossWeight() != 0.5 {
		t.Fatalf("explicit balance weight = %v, want 0.5", config.MoEBalanceLossWeight())
	}
}

// TestMoEBalanceLossUniformIsWeight checks the sequence-level balance loss on a
// purely uniform router: with equal gate probabilities the loss is exactly the
// balance weight regardless of the top-k selection (a closed-form invariant of
// the DeepSeek-V3 formulation).
func TestMoEBalanceLossUniformIsWeight(t *testing.T) {
	config := moeTestConfig()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(3))
	for _, block := range model.blocks {
		if block.moe == nil {
			continue
		}
		block.moe.router.weight.Weight.Set(0)
		block.moe.router.bias.Set(0)
		block.moe.BalanceWeight = 0.0003
	}
	sequenceLength := 8
	input := tensors.New(1, sequenceLength, config.EmbedDim)
	_, context := model.blocks[0].moe.forwardTraining(input, sequenceLength)
	if math.Abs(float64(context.balanceLoss)-0.0003) > 1e-6 {
		t.Fatalf("uniform balance loss = %v, want 0.0003", context.balanceLoss)
	}
	if len(context.balanceGradProbs) != context.rows*model.blocks[0].moe.NumExperts {
		t.Fatalf("balance gradient length = %d", len(context.balanceGradProbs))
	}
}

// TestMoEBalanceLossGradientFlowsToRouter verifies the balance loss contributes
// to the router weight gradient: disabling it must change the accumulated router
// gradient.
func TestMoEBalanceLossGradientFlowsToRouter(t *testing.T) {
	model := moeTestModel()
	indexes, targets := tinyData()
	analyticGrads(model, indexes, targets)
	withBalance := append([]float32(nil), model.blocks[0].moe.router.weight.Weight.Grad...)

	for _, block := range model.blocks {
		if block.moe != nil {
			block.moe.BalanceWeight = 0
		}
	}
	analyticGrads(model, indexes, targets)
	withoutBalance := model.blocks[0].moe.router.weight.Weight.Grad

	changed := false
	for index := range withBalance {
		if math.Abs(float64(withBalance[index]-withoutBalance[index])) > 1e-9 {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("balance loss should contribute to the router gradient")
	}
}

// TestMoEBalanceLossReference recomputes the sequence-level balance loss from
// the saved router probabilities and selection counts and compares it to the
// value stored in the training context on a non-uniform router.
func TestMoEBalanceLossReference(t *testing.T) {
	config := moeTestConfig()
	model := NewTransformer(config)
	model.InitWeights(tensors.NewRNG(5))
	block := model.blocks[0]
	block.moe.BalanceWeight = 1e-3
	sequenceLength := 4
	input := tensors.New(2, sequenceLength, config.EmbedDim)
	rng := tensors.NewRNG(77)
	for index := range input.Data {
		input.Data[index] = rng.NormFloat32()
	}
	_, context := block.moe.forwardTraining(input, sequenceLength)

	expertCount := block.moe.NumExperts
	topK := block.moe.TopK
	sequences := context.rows / sequenceLength
	tokenCount := float64(sequenceLength)

	counts := make([]float64, sequences*expertCount)
	for index := range context.experts {
		expert := &context.experts[index]
		for _, row := range expert.rows {
			counts[(row/sequenceLength)*expertCount+expert.id]++
		}
	}
	sums := make([]float64, sequences*expertCount)
	for row := 0; row < context.rows; row++ {
		sequenceBase := (row / sequenceLength) * expertCount
		for expert := 0; expert < expertCount; expert++ {
			sums[sequenceBase+expert] += float64(context.probs.Data[row*expertCount+expert])
		}
	}
	var want float64
	for sequence := 0; sequence < sequences; sequence++ {
		for expert := 0; expert < expertCount; expert++ {
			fraction := float64(expertCount) / (float64(topK) * tokenCount) * counts[sequence*expertCount+expert]
			probability := sums[sequence*expertCount+expert] / tokenCount
			want += fraction * probability
		}
	}
	want = float64(block.moe.BalanceWeight) * want / float64(sequences)
	if math.Abs(context.balanceLoss-want) > 1e-12 {
		t.Fatalf("balance loss = %v, want %v", context.balanceLoss, want)
	}
}
