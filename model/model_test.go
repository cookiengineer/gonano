package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensor"
)

func testConfig() Config {
	return Config{
		SequenceLen:   16,
		VocabSize:     32,
		NumLayer:      2,
		NumHead:       2,
		NumKVHead:     2,
		EmbedDim:      32,
		WindowPattern: "L",
	}
}

func TestConfigForDepth(t *testing.T) {
	cfg := ConfigForDepth(12, 32000, 64, 128, 2048, "SSSL")
	if cfg.EmbedDim != 12*64 {
		t.Fatalf("EmbedDim = %d, want %d", cfg.EmbedDim, 12*64)
	}
	if cfg.NumHead != cfg.EmbedDim/128 {
		t.Fatalf("NumHead = %d, want %d", cfg.NumHead, cfg.EmbedDim/128)
	}
	if cfg.EmbedDim%cfg.NumHead != 0 {
		t.Fatal("EmbedDim must be divisible by NumHead")
	}
}

func TestWindowSizes(t *testing.T) {
	cfg := Config{
		SequenceLen:   2048,
		VocabSize:     32,
		NumLayer:      4,
		NumHead:       2,
		NumKVHead:     2,
		EmbedDim:      8,
		WindowPattern: "SSSL",
	}
	sizes := cfg.WindowSizes()
	// short = ceil(2048/4/128)*128 = ceil(4)*128 = 512.
	if sizes[0] != [2]int{512, 0} {
		t.Fatalf("layer 0 window = %v, want [512 0]", sizes[0])
	}
	if sizes[1] != [2]int{512, 0} {
		t.Fatalf("layer 1 window = %v", sizes[1])
	}
	if sizes[2] != [2]int{512, 0} {
		t.Fatalf("layer 2 window = %v", sizes[2])
	}
	// Last layer always full context.
	if sizes[3] != [2]int{2048, 0} {
		t.Fatalf("layer 3 window = %v, want [2048 0]", sizes[3])
	}
}

func TestPaddedVocab(t *testing.T) {
	cfg := testConfig() // vocab 32
	if cfg.PaddedVocab() != 64 {
		t.Fatalf("PaddedVocab = %d, want 64", cfg.PaddedVocab())
	}
	cfg.VocabSize = 65
	if cfg.PaddedVocab() != 128 {
		t.Fatalf("PaddedVocab = %d, want 128", cfg.PaddedVocab())
	}
}

func TestApplyRotaryQuarterTurn(t *testing.T) {
	// headDim = 2 (half = 1): position 1 rotates by 90 degrees.
	cos := tensor.New(2, 1)
	sin := tensor.New(2, 1)
	cos.Set2(0, 0, 1)
	cos.Set2(1, 0, 0)
	sin.Set2(0, 0, 0)
	sin.Set2(1, 0, 1)

	x := tensor.NewWithData([]int{1, 1, 1, 2}, []float32{3, 4})
	out := ApplyRotary(x, cos, sin, 1)
	// y1 = x1*cos + x2*sin = 3*0 + 4*1 = 4; y2 = -x1*sin + x2*cos = -3*1 + 4*0 = -3.
	if out.Data[0] != 4 || out.Data[1] != -3 {
		t.Fatalf("rotary = %v, want [4 -3]", out.Data)
	}
}

func TestApplyRotaryPreservesNorm(t *testing.T) {
	headDim := 8
	cos, sin := precomputeRotary(32, headDim)
	rng := tensor.NewRNG(3)
	x := tensor.New(2, 4, 2, headDim)
	tensor.FillNormal(x, rng, 1)

	out := ApplyRotary(x, cos, sin, 0)
	// Sum of squares must be preserved per (b,t,h) pair.
	for b := 0; b < 2; b++ {
		for tt := 0; tt < 4; tt++ {
			for h := 0; h < 2; h++ {
				var in, out2 float64
				base := ((b*4+tt)*2 + h) * headDim
				for j := 0; j < headDim; j++ {
					in += float64(x.Data[base+j]) * float64(x.Data[base+j])
					out2 += float64(out.Data[base+j]) * float64(out.Data[base+j])
				}
				if math.Abs(in-out2) > 1e-3 {
					t.Fatalf("rotation changed norm: %v -> %v", in, out2)
				}
			}
		}
	}
}

func buildTestTransformer(t *testing.T) *Transformer {
	t.Helper()
	m := NewTransformer(testConfig())
	m.InitWeights(tensor.NewRNG(42))
	return m
}

func TestForwardShapes(t *testing.T) {
	m := buildTestTransformer(t)
	idx := tensor.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	logits := m.Forward(idx, nil)
	if logits.Shape[0] != 1 || logits.Shape[1] != 4 || logits.Shape[2] != 32 {
		t.Fatalf("logits shape = %v, want [1 4 32]", logits.Shape)
	}
	for _, v := range logits.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logits contain non-finite value %v", v)
		}
	}
}

func TestForwardDeterministic(t *testing.T) {
	idx := tensor.NewInt32sWithData([]int{1, 6}, []int32{1, 2, 3, 4, 5, 6})
	a := buildTestTransformer(t)
	b := buildTestTransformer(t)
	la := a.Forward(idx, nil)
	lb := b.Forward(idx, nil)
	for i := range la.Data {
		if la.Data[i] != lb.Data[i] {
			t.Fatalf("nondeterministic forward at %d", i)
		}
	}
}

func TestForwardBatch(t *testing.T) {
	m := buildTestTransformer(t)
	idx := tensor.NewInt32sWithData([]int{3, 5}, []int32{
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
	})
	logits := m.Forward(idx, nil)
	if logits.Shape[0] != 3 || logits.Shape[1] != 5 || logits.Shape[2] != 32 {
		t.Fatalf("logits shape = %v", logits.Shape)
	}
}

func TestForwardDecodePath(t *testing.T) {
	m := buildTestTransformer(t)
	cache := NewKVBuffer(1, 16, m.NumLayers(), m.Config.NumKVHead, m.Config.HeadDim())

	prefill := tensor.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	p := m.Forward(prefill, cache)
	if p.Shape[0] != 1 || p.Shape[1] != 4 {
		t.Fatalf("prefill logits shape = %v", p.Shape)
	}
	if cache.Position() != 4 {
		t.Fatalf("cache position = %d, want 4", cache.Position())
	}

	step := tensor.NewInt32sWithData([]int{1, 1}, []int32{5})
	d := m.Forward(step, cache)
	if d.Shape[0] != 1 || d.Shape[1] != 1 {
		t.Fatalf("decode logits shape = %v", d.Shape)
	}
	if cache.Position() != 5 {
		t.Fatalf("cache position = %d, want 5", cache.Position())
	}
}

func TestInitWeightsStats(t *testing.T) {
	cfg := Config{
		SequenceLen: 16, VocabSize: 64, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 64, WindowPattern: "L",
	}
	m := NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(7))

	wteStd := stdDev(m.wte.Weight.Data)
	if math.Abs(float64(wteStd)-0.8) > 0.05 {
		t.Fatalf("wte std = %v, want ~0.8", wteStd)
	}
	lmStd := stdDev(m.lm.Weight.Data)
	if math.Abs(float64(lmStd)-0.001) > 0.0005 {
		t.Fatalf("lm_head std = %v, want ~0.001", lmStd)
	}
	// resid lambdas decrease from 1.15 to 1.05 across 2 layers.
	if m.residLambdas.Data[0] != 1.15 || m.residLambdas.Data[1] != 1.05 {
		t.Fatalf("resid lambdas = %v, want [1.15 1.05]", m.residLambdas.Data)
	}
	if m.backoutLambda.Data[0] != 0.2 {
		t.Fatalf("backout = %v, want 0.2", m.backoutLambda.Data[0])
	}
	if m.smearLambda.Data[0] != 0 {
		t.Fatalf("smear lambda = %v, want 0", m.smearLambda.Data[0])
	}
}

func stdDev(s []float32) float32 {
	var mean, m2 float64
	for i, v := range s {
		d := float64(v) - mean
		mean += d / float64(i+1)
		m2 += d * (float64(v) - mean)
	}
	return float32(math.Sqrt(m2 / float64(len(s))))
}
