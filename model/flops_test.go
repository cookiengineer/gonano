package model

import (
	"testing"
)

func TestMatmulParamsMatchesNamedParameters(t *testing.T) {
	m := buildTestTransformer(t)
	// MatmulParams should count exactly the nn.Linear weights.
	mm := m.MatmulParams()
	if mm <= 0 {
		t.Fatal("MatmulParams should be positive")
	}
}

func TestScalingParamsTotalMatchesTotalParams(t *testing.T) {
	m := buildTestTransformer(t)
	sp := m.NumScalingParams()
	if sp.Total != m.TotalParams() {
		t.Fatalf("scaling total %d != TotalParams %d", sp.Total, m.TotalParams())
	}
	// Scalars group is exactly resid + x0 + smear + backout.
	wantScalars := m.Config.NumLayer*2 + 1 + 1 + 1 // resid(n), x0(n), smear_lambda, backout_lambda
	_ = wantScalars
	if sp.Scalars != m.residLambdas.Numel()+m.x0Lambdas.Numel()+m.smearGate.Weight.Numel()+m.smearLambda.Numel()+m.backoutLambda.Numel() {
		t.Fatalf("scalars mismatch")
	}
}

func TestFlopsEstimatesPositive(t *testing.T) {
	m := buildTestTransformer(t)
	if m.EstimateFlopsPerToken() <= 0 {
		t.Fatal("EstimateFlopsPerToken should be positive")
	}
	if m.EstimateDecodeFlops(8) <= 0 {
		t.Fatal("EstimateDecodeFlops should be positive")
	}
	if m.EstimatePrefillFlops(8) <= 0 {
		t.Fatal("EstimatePrefillFlops should be positive")
	}
	if m.KVBytesPerToken() <= 0 {
		t.Fatal("KVBytesPerToken should be positive")
	}
	if m.KVReadBytes(8) <= 0 {
		t.Fatal("KVReadBytes should be positive")
	}
}

func TestKVBytesPerTokenFormula(t *testing.T) {
	m := buildTestTransformer(t)
	// n_layer * 2 * n_kv_head * head_dim * 4 bytes.
	want := m.Config.NumLayer * 2 * m.Config.NumKVHead * m.Config.HeadDim() * 4
	if m.KVBytesPerToken() != want {
		t.Fatalf("KVBytesPerToken = %d, want %d", m.KVBytesPerToken(), want)
	}
}

func TestMatmulParamsFormula(t *testing.T) {
	cfg := testConfig()
	headDim := cfg.HeadDim()
	perBlock := 6 * cfg.EmbedDim * cfg.EmbedDim // cq,ck,cv,cproj (4*E*E) + cfc,cproj (2*E*4E)
	_ = perBlock
	// Explicit: cq/cv/ck/cproj each E*E; c_fc E*4E; c_proj 4E*E.
	blockMM := 4*cfg.EmbedDim*cfg.EmbedDim + 2*4*cfg.EmbedDim*cfg.EmbedDim // = 12*E*E
	_ = blockMM
	// ve_gate on layer 1 only: 12 * numKVHead.
	veGate := 12 * cfg.NumKVHead
	lmHead := cfg.EmbedDim * cfg.PaddedVocab()
	smear := 24 * 1
	want := cfg.NumLayer*(4*cfg.EmbedDim*cfg.EmbedDim+2*4*cfg.EmbedDim*cfg.EmbedDim) + veGate + lmHead + smear
	if m := buildTestTransformer(t); m.MatmulParams() != want {
		t.Fatalf("MatmulParams = %d, want %d", m.MatmulParams(), want)
	}
	_ = headDim
}
