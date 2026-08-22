package train

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/data"
	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/optim"
	"github.com/cookiengineer/gonano/tensor"
	"github.com/cookiengineer/gonano/tokenizer"
)

func sftModel() (*model.Transformer, *tokenizer.Tokenizer) {
	cfg := model.Config{
		SequenceLen: 32, VocabSize: 265, NumLayer: 1, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(1))
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	tok := tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
	return m, tok
}

func TestTrainSFT(t *testing.T) {
	m, tok := sftModel()
	convs := []*tokenizer.Conversation{
		{Messages: []tokenizer.Message{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}}},
		{Messages: []tokenizer.Message{{Role: "user", Content: "how are you"}, {Role: "assistant", Content: "fine thanks"}}},
	}
	i := 0
	provider := func() ([]*tokenizer.Conversation, bool) {
		c := convs[i%len(convs)]
		i++
		return []*tokenizer.Conversation{c}, true
	}
	loader := data.NewSFTLoader(tok, 1, 32, provider, 5)

	groups := m.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1)
	losses := TrainSFT(m, groups, loader, 20, nil)
	if len(losses) == 0 {
		t.Fatal("no losses recorded")
	}
	if math.IsNaN(float64(losses[len(losses)-1])) {
		t.Fatal("loss became NaN")
	}
}

func TestPolicyGradientStep(t *testing.T) {
	m, _ := sftModel()
	inputs := tensor.NewInt32sWithData([]int{2, 4}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
	targets := tensor.NewInt32sWithData([]int{2, 4}, []int32{2, 3, 4, 5, 6, 7, 8, 9})
	advantages := []float32{1.0, -0.5}
	m.ZeroGrad()
	loss := PolicyGradientStep(m, inputs, targets, advantages, 8.0)
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("pg loss = %v, want finite", loss)
	}
	// Gradients must have been accumulated.
	any := false
	for _, p := range m.Parameters() {
		for _, g := range p.Grad {
			if g != 0 {
				any = true
			}
		}
	}
	if !any {
		t.Fatal("no gradients accumulated")
	}
}

func TestRLStep(t *testing.T) {
	m, _ := sftModel()
	groups := m.SetupOptimizer(0.01, 0.1, 0.01, 0.0, 0.1)
	opt := optim.NewMuonAdamW(groups)
	inputs := tensor.NewInt32sWithData([]int{1, 4}, []int32{1, 2, 3, 4})
	targets := tensor.NewInt32sWithData([]int{1, 4}, []int32{2, 3, 4, 5})
	loss := RLStep(m, opt, inputs, targets, []float32{1.0}, 4.0)
	if math.IsNaN(float64(loss)) {
		t.Fatal("rl loss NaN")
	}
}
