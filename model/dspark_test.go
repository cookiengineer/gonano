package model

import (
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestDrafterConfigBounded(t *testing.T) {
	backbone := testConfig()
	backbone.SequenceLen = 2048
	config := DrafterConfig(backbone)
	if config.NumLayer != DrafterLayers {
		t.Fatalf("drafter layers = %d, want %d", config.NumLayer, DrafterLayers)
	}
	if config.SequenceLen > 128 {
		t.Fatalf("drafter context = %d, want <= 128", config.SequenceLen)
	}
	if config.VocabSize != backbone.VocabSize || config.HeadDim() != backbone.HeadDim() {
		t.Fatal("drafter must keep the backbone vocab and head geometry")
	}
}

func TestDraftTokensLengthAndRange(t *testing.T) {
	backbone := NewTransformer(testConfig())
	drafter := NewDrafter(backbone)
	drafter.InitWeights(tensors.NewRNG(7))

	context := []int{1, 5, 2, 8, 3, 7, 4, 6}
	drafts := drafter.DraftTokens(context, 4)
	if len(drafts) != 4 {
		t.Fatalf("drafts = %d, want 4", len(drafts))
	}
	for _, token := range drafts {
		if token < 0 || token >= drafter.Config.VocabSize {
			t.Fatalf("draft token %d out of vocab range", token)
		}
	}
}

func TestDraftTokensEmptyContext(t *testing.T) {
	drafter := NewDrafter(NewTransformer(testConfig()))
	drafter.InitWeights(tensors.NewRNG(1))
	if drafts := drafter.DraftTokens(nil, 3); drafts != nil {
		t.Fatalf("empty context should draft nothing, got %v", drafts)
	}
}

func TestDraftTokensBoundedByContext(t *testing.T) {
	backbone := NewTransformer(testConfig())
	drafter := NewDrafter(backbone)
	drafter.InitWeights(tensors.NewRNG(2))

	long := make([]int, 500)
	for index := range long {
		long[index] = index % backbone.Config.VocabSize
	}
	// A long context must still draft (using only the bounded window) rather
	// than panic on the rotary/context limit.
	drafts := drafter.DraftTokens(long, 3)
	if len(drafts) != 3 {
		t.Fatalf("drafts = %d, want 3", len(drafts))
	}
}
