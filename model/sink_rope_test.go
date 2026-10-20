package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestAttentionSinkGradientMatchesFiniteDifference(t *testing.T) {
	config := testConfig()
	config.AttentionSink = true
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	indexes, targets := tinyData()
	analyticGrads(transformer, indexes, targets)

	sink := transformer.blocks[0].attention.sink
	epsilon := float32(1e-3)
	for head := 0; head < config.NumHead; head++ {
		original := sink.Data[head]
		sink.Data[head] = original + epsilon
		plus := meanLoss(transformer, indexes, targets)
		sink.Data[head] = original - epsilon
		minus := meanLoss(transformer, indexes, targets)
		sink.Data[head] = original
		numeric := float64(plus-minus) / (2 * float64(epsilon))
		analytic := float64(sink.Grad[head])
		if math.Abs(numeric-analytic) > 1e-2*math.Max(1, math.Abs(numeric)) {
			t.Fatalf("head %d sink gradient = %v, numeric %v", head, analytic, numeric)
		}
	}
}

func TestYarnRotaryTablesDifferFromBase(t *testing.T) {
	baseCos, baseSin := precomputeRotaryScaled(64, 16, 100000, RopeScalingConfig{})
	yarnCos, yarnSin := precomputeRotaryScaled(64, 16, 100000, RopeScalingConfig{
		Mode: "yarn", Factor: 4, OriginalMaxPosition: 64, BetaFast: 32, BetaSlow: 1,
	})
	differ := false
	for index := range baseCos.Data {
		if baseCos.Data[index] != yarnCos.Data[index] || baseSin.Data[index] != yarnSin.Data[index] {
			differ = true
			break
		}
	}
	if !differ {
		t.Fatal("YaRN tables should differ from base RoPE")
	}
}

func TestPerLayerRopeTheta(t *testing.T) {
	config := Config{
		SequenceLen: 16, VocabSize: 16, NumLayer: 2, NumHead: 2, NumKVHead: 1,
		EmbedDim: 16, WindowPattern: "L", CompressionRatios: []int{0, 2},
		RopeTheta: 100000, CompressRopeTheta: 500000, SWAWindow: 4,
	}
	transformer := NewTransformer(config)
	firstCos, _ := transformer.rotaryForLayer(0)
	secondCos, _ := transformer.rotaryForLayer(1)
	if firstCos == secondCos {
		t.Fatal("layers with different rope theta should not share a table")
	}
	differ := false
	for index := range firstCos.Data {
		if firstCos.Data[index] != secondCos.Data[index] {
			differ = true
			break
		}
	}
	if !differ {
		t.Fatal("per-layer rope tables should differ")
	}
}
