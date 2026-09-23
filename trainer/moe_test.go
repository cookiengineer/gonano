package trainer

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// TestTrainerMoEOverfitsTiny trains a tiny MoE model and checks that the loss
// decreases, the expert weights receive gradient updates, and the
// auxiliary-loss-free router bias moves off zero during the optimizer steps.
func TestTrainerMoEOverfitsTiny(tester *testing.T) {
	config := model.Config{
		SequenceLen: 8, VocabSize: 16, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
		NumExperts: 4, NumExpertsPerToken: 2, ExpertHiddenDim: 16, SharedExpertHiddenDim: 16,
	}
	config.ApplyMoEDefaults()
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(0))
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1, false)
	trainer := NewTrainer(transformer, groups, 1)
	trainer.WarmupSteps = 2
	trainer.WarmdownRatio = 0.0

	inputs := tensors.NewInt32sWithData([]int{1, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	targets := tensors.NewInt32sWithData([]int{1, 8}, []int32{2, 3, 4, 5, 6, 7, 8, 9})

	first := trainer.TrainStep(inputs, targets)
	trainer.StepOptimizer(0, 100)
	var last float32
	for step := 1; step < 100; step++ {
		last = trainer.TrainStep(inputs, targets)
		trainer.StepOptimizer(step, 100)
	}
	if math.IsNaN(float64(last)) || math.IsInf(float64(last), 0) {
		tester.Fatalf("loss became non-finite: %v", last)
	}
	if last >= first {
		tester.Fatalf("loss did not decrease: %v -> %v", first, last)
	}

	// Every expert should have received a weight update from the zero-init down
	// projection (at least one nonzero element).
	parameters := transformer.NamedParameters()
	down := parameters["transformer.h.0.moe.experts.down.weight"]
	updated := false
	for _, value := range down.Data {
		if value != 0 {
			updated = true
			break
		}
	}
	if !updated {
		tester.Fatal("expert weights were never updated")
	}

	// The auxiliary-loss-free bias must have moved from its zero initialization.
	biasMoved := false
	for _, value := range parameters["transformer.h.0.moe.router_bias"].Data {
		if value != 0 {
			biasMoved = true
			break
		}
	}
	if !biasMoved {
		tester.Fatal("router bias did not update during training")
	}
}

// TestDeriveHyperparamsForMoEConfig verifies the scaling path sees the full MoE
// parameter count (target tokens) while per-token FLOPs use the active experts.
func TestDeriveHyperparamsForMoEConfig(tester *testing.T) {
	dense := model.ConfigForDepth(8, 32000, 64, 128, 2048, "SSSL")
	moe := dense
	moe.NumExperts = model.MoEExpertsForDepth(8)
	moe.NumExpertsPerToken = 2
	moe.ApplyMoEDefaults()
	moe.Validate()

	denseHyper := DeriveHyperparamsForConfig(dense, 12, 0.28)
	moeHyper := DeriveHyperparamsForConfig(moe, 12, 0.28)
	if moeHyper.TotalTokens <= denseHyper.TotalTokens {
		tester.Fatalf("MoE target tokens %d should exceed dense %d", moeHyper.TotalTokens, denseHyper.TotalTokens)
	}
	if moeHyper.FlopsPerToken <= denseHyper.FlopsPerToken {
		tester.Fatalf("MoE active FLOPs %v should exceed dense %v", moeHyper.FlopsPerToken, denseHyper.FlopsPerToken)
	}
	// The depth-only entry point must still agree with the dense config.
	if got := DeriveHyperparams(8, 32000, 12, 0.28); got.TotalTokens != denseHyper.TotalTokens {
		tester.Fatalf("DeriveHyperparams should match DeriveHyperparamsForConfig for the derived dense config")
	}
}
