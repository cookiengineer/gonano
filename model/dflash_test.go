package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func dflashTestBackbone() *Transformer {
	config := ConfigForDepth(4, 16, 16, 32, 64, "SSSL")
	config.ApplyPreset(PresetFlash)
	config.DFlashBlockSize = 3
	config.DFlashMarkovRank = 8
	config.DFlashTargetLayerIDs = []int{config.NumLayer - 1}
	config.Validate()
	backbone := NewTransformer(config)
	backbone.InitWeights(tensors.NewRNG(42))
	return backbone
}

func TestDFlashTrainStepFinite(t *testing.T) {
	backbone := dflashTestBackbone()
	indexes, targets := tinyData()
	_, context := backbone.TrainForward(indexes)
	hidden := TargetHiddenFromContext(context, backbone.Config.DFlashTargetLayerIDs)
	dflash := NewDFlash(backbone)
	dflash.InitWeights(tensors.NewRNG(7))

	loss := dflash.TrainStep(indexes, targets, hidden, 3)
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("non-finite DFlash loss %v", loss)
	}
	nonzero := false
	for _, value := range dflash.Projection.Weight.Grad {
		if value != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Fatal("DFlash projection received no gradient")
	}
	for _, parameter := range dflash.Parameters() {
		for _, value := range parameter.Grad {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("DFlash produced a non-finite gradient")
			}
		}
	}
}

func TestDFlashDraftBounded(t *testing.T) {
	backbone := dflashTestBackbone()
	dflash := NewDFlash(backbone)
	dflash.InitWeights(tensors.NewRNG(7))
	context := []int{1, 5, 2, 8, 3, 7}
	hidden := tensors.NewWithData([]int{1, 1, backbone.Config.EmbedDim}, make([]float32, backbone.Config.EmbedDim))
	tokens, confidences := dflash.Draft(context, hidden, 3)
	if len(tokens) != 3 || len(confidences) != 3 {
		t.Fatalf("draft length = %d confidences = %d, want 3/3", len(tokens), len(confidences))
	}
	for _, token := range tokens {
		if token < 0 || token >= backbone.Config.VocabSize {
			t.Fatalf("draft token %d out of range", token)
		}
	}
	for _, confidence := range confidences {
		if confidence < 0 || confidence > 1 {
			t.Fatalf("confidence %v out of [0,1]", confidence)
		}
	}
}

func TestDFlashTargetHiddenMatchesTraining(t *testing.T) {
	cases := [][]int{
		{1},
		{0, 1},
	}
	for _, targetIDs := range cases {
		config := testConfig()
		config.DFlashTargetLayerIDs = targetIDs
		backbone := NewTransformer(config)
		backbone.InitWeights(tensors.NewRNG(42))
		indexes, _ := tinyData()

		_, context := backbone.TrainForward(indexes)
		training := TargetHiddenFromContext(context, targetIDs)
		cache := NewKVBuffer(1, indexes.Shape[1], config.NumLayer, config.NumKVHead, config.HeadDim())
		_, inference := backbone.ForwardHiddenAtLayers(indexes, cache, targetIDs)
		if len(training.Data) != len(inference.Data) {
			t.Fatalf("targetIDs %v: hidden length %d != %d", targetIDs, len(training.Data), len(inference.Data))
		}
		for index := range training.Data {
			if math.Abs(float64(training.Data[index]-inference.Data[index])) > 1e-5 {
				t.Fatalf("targetIDs %v: hidden %d = %v, want %v", targetIDs, index, inference.Data[index], training.Data[index])
			}
		}
	}
}
