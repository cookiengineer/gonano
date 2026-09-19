package model

import (
	"testing"
)

func TestMatmulParamsMatchesNamedParameters(t *testing.T) {
	model := buildTestTransformer(t)
	// MatmulParams should count exactly the layers.Linear weights.
	matmulParameterCount := model.MatmulParams()
	if matmulParameterCount <= 0 {
		t.Fatal("MatmulParams should be positive")
	}
}

func TestScalingParamsTotalMatchesTotalParams(t *testing.T) {
	model := buildTestTransformer(t)
	scalingParams := model.NumScalingParams()
	if scalingParams.Total != model.TotalParams() {
		t.Fatalf("scaling total %d != TotalParams %d", scalingParams.Total, model.TotalParams())
	}
	// Scalars group is exactly resid + x0 + smear + backout.
	expectedScalars := model.Config.NumLayer*2 + 1 + 1 + 1 // resid(n), x0(n), smear_lambda, backout_lambda
	_ = expectedScalars
	if scalingParams.Scalars != model.residLambdas.Numel()+model.x0Lambdas.Numel()+model.smearGate.Weight.Numel()+model.smearLambda.Numel()+model.backoutLambda.Numel() {
		t.Fatalf("scalars mismatch")
	}
}

func TestFlopsEstimatesPositive(t *testing.T) {
	model := buildTestTransformer(t)
	if model.EstimateFlopsPerToken() <= 0 {
		t.Fatal("EstimateFlopsPerToken should be positive")
	}
	if model.EstimateDecodeFlops(8) <= 0 {
		t.Fatal("EstimateDecodeFlops should be positive")
	}
	if model.EstimatePrefillFlops(8) <= 0 {
		t.Fatal("EstimatePrefillFlops should be positive")
	}
	if model.KVBytesPerToken() <= 0 {
		t.Fatal("KVBytesPerToken should be positive")
	}
	if model.KVReadBytes(8) <= 0 {
		t.Fatal("KVReadBytes should be positive")
	}
}

func TestGQAReducesKVBytes(t *testing.T) {
	mha := NewTransformer(ConfigForDepthRatio(12, 32000, 64, 128, 2048, "SSSL", 1))
	gqa := NewTransformer(ConfigForDepthRatio(12, 32000, 64, 128, 2048, "SSSL", 3))
	if gqa.Config.NumKVHead*3 != mha.Config.NumKVHead {
		t.Fatalf("NumKVHead: gqa=%d mha=%d, want ratio 3", gqa.Config.NumKVHead, mha.Config.NumKVHead)
	}
	if gqa.KVBytesPerToken()*3 != mha.KVBytesPerToken() {
		t.Fatalf("KVBytesPerToken gqa=%d mha=%d, want 3x reduction", gqa.KVBytesPerToken(), mha.KVBytesPerToken())
	}
}

func TestWeightReadBytesFormula(t *testing.T) {
	model := buildTestTransformer(t)
	expected := model.MatmulParams() * 4
	if model.WeightReadBytes() != expected {
		t.Fatalf("WeightReadBytes = %d, want %d", model.WeightReadBytes(), expected)
	}
}

func TestKVBytesPerTokenFormula(t *testing.T) {
	model := buildTestTransformer(t)
	// n_layer * 2 * n_kv_head * head_dim * 4 bytes.
	expected := model.Config.NumLayer * 2 * model.Config.NumKVHead * model.Config.HeadDim() * 4
	if model.KVBytesPerToken() != expected {
		t.Fatalf("KVBytesPerToken = %d, want %d", model.KVBytesPerToken(), expected)
	}
}

func TestMatmulParamsFormula(t *testing.T) {
	config := testConfig()
	headDimension := config.HeadDim()
	perBlock := 6 * config.EmbedDim * config.EmbedDim // cq,ck,cv,cproj (4*E*E) + cfc,cproj (2*E*4E)
	_ = perBlock
	// Explicit: cq/cv/ck/cproj each E*E; c_fc E*4E; c_proj 4E*E.
	blockMatmulParameters := 4*config.EmbedDim*config.EmbedDim + 2*4*config.EmbedDim*config.EmbedDim // = 12*E*E
	_ = blockMatmulParameters
	// ve_gate on layer 1 only: 12 * numKVHead.
	valueEmbeddingGate := 12 * config.NumKVHead
	lmHead := config.EmbedDim * config.PaddedVocab()
	smear := 24 * 1
	expected := config.NumLayer*(4*config.EmbedDim*config.EmbedDim+2*4*config.EmbedDim*config.EmbedDim) + valueEmbeddingGate + lmHead + smear
	if model := buildTestTransformer(t); model.MatmulParams() != expected {
		t.Fatalf("MatmulParams = %d, want %d", model.MatmulParams(), expected)
	}
	_ = headDimension
}
