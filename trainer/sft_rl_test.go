package trainer

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optimizer"
	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

func newSFTModel() (*model.Transformer, *tokenizer.Tokenizer) {
	config := model.Config{
		SequenceLen: 32, VocabSize: 265, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(1))
	ranks := make(map[string]int, 256)
	for index := 0; index < 256; index++ {
		ranks[string([]byte{byte(index)})] = index
	}
	tokenizerInstance := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return transformer, tokenizerInstance
}

func TestTrainSFT(tester *testing.T) {
	transformer, tokenizerInstance := newSFTModel()
	conversations := []*tokenizer.Conversation{
		{Messages: []tokenizer.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}},
		{Messages: []tokenizer.Message{{Role: "user", Content: "how are you"}, {Role: "assistant", Content: "fine thanks"}}},
	}
	conversationIndex := 0
	provider := func() ([]*tokenizer.Conversation, bool) {
		conversation := conversations[conversationIndex%len(conversations)]
		conversationIndex++
		return []*tokenizer.Conversation{conversation}, true
	}
	loader := data.NewSFTLoader(tokenizerInstance, 1, 32, provider, 5)

	groups := transformer.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1)
	losses := TrainSFT(transformer, groups, loader, 20, nil)
	if len(losses) == 0 {
		tester.Fatal("no losses recorded")
	}
	if math.IsNaN(float64(losses[len(losses)-1])) {
		tester.Fatal("loss became NaN")
	}
}

func TestPolicyGradientStep(tester *testing.T) {
	transformer, _ := newSFTModel()
	inputs := tensors.NewInt32sWithData([]int{2, 4}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	targets := tensors.NewInt32sWithData([]int{2, 4}, []int32{2, 3, 4, 5, 6, 7, 8, 9})
	advantages := []float32{1.0, -0.5}
	transformer.ZeroGrad()
	loss := PolicyGradientStep(transformer, inputs, targets, advantages, 8.0)
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		tester.Fatalf("pg loss = %v, want finite", loss)
	}
	// Gradients must have been accumulated.
	hasGradient := false
	for _, parameter := range transformer.Parameters() {
		for _, gradientValue := range parameter.Grad {
			if gradientValue != 0 {
				hasGradient = true
			}
		}
	}
	if !hasGradient {
		tester.Fatal("no gradients accumulated")
	}
}

func TestRLStep(tester *testing.T) {
	transformer, _ := newSFTModel()
	groups := transformer.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1)
	optimizer := optimizer.NewMuonAdamW(groups)
	inputs := tensors.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	targets := tensors.NewInt32sWithData([]int{1, 4}, []int32{2, 3, 4, 5})
	loss := RLStep(transformer, optimizer, inputs, targets, []float32{1.0}, 4.0)
	if math.IsNaN(float64(loss)) {
		tester.Fatal("rl loss NaN")
	}
}
