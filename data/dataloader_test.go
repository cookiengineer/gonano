package data

import (
	"testing"

	"github.com/cookiengineer/gonano/tokenizer"
)

// simpleTokenizer builds a byte-level tokenizer with no merges (256 byte
// tokens + special tokens), so encode maps each byte to its id.
func simpleTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for i := 0; i < 256; i++ {
		ranks[string([]byte{byte(i)})] = i
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func cyclicProvider(docs []string) DocProvider {
	i := 0
	return func() ([]string, State) {
		batch := []string{docs[i%len(docs)]}
		i++
		return batch, State{PQIndex: 0, RGIndex: 0, Epoch: 1}
	}
}

func TestPretrainLoaderBOSAligned(t *testing.T) {
	tok := simpleTokenizer()
	bos := tok.BOSTokenID()
	provider := cyclicProvider([]string{"hello", "world", "hi", "there"})
	loader := NewPretrainLoader(tok, 2, 8, provider, 10)

	inputs, targets, _ := loader.Next()
	if inputs.Shape[0] != 2 || inputs.Shape[1] != 8 {
		t.Fatalf("inputs shape = %v, want [2 8]", inputs.Shape)
	}
	// Every row starts with BOS.
	for r := 0; r < 2; r++ {
		if inputs.Get2(r, 0) != int32(bos) {
			t.Fatalf("row %d does not start with BOS: %d", r, inputs.Get2(r, 0))
		}
	}
	// targets[i] == inputs[i+1] (autoregressive shift).
	for r := 0; r < 2; r++ {
		for i := 0; i < 7; i++ {
			if targets.Get2(r, i) != inputs.Get2(r, i+1) {
				t.Fatalf("shift mismatch at (%d,%d)", r, i)
			}
		}
	}
	// All ids are valid.
	vocab := tok.VocabSize()
	for r := 0; r < 2; r++ {
		for i := 0; i < 8; i++ {
			if int(inputs.Get2(r, i)) >= vocab {
				t.Fatalf("invalid input id %d at (%d,%d)", inputs.Get2(r, i), r, i)
			}
		}
	}
}

func TestPretrainLoaderMultipleBatches(t *testing.T) {
	tok := simpleTokenizer()
	provider := cyclicProvider([]string{"a", "b", "c"})
	loader := NewPretrainLoader(tok, 3, 4, provider, 5)

	// Pull several batches; each must be BOS-aligned and consistently shifted.
	for b := 0; b < 10; b++ {
		inputs, targets, _ := loader.Next()
		for r := 0; r < 3; r++ {
			if inputs.Get2(r, 0) != int32(tok.BOSTokenID()) {
				t.Fatalf("batch %d row %d not BOS-aligned", b, r)
			}
			for i := 0; i < 3; i++ {
				if targets.Get2(r, i) != inputs.Get2(r, i+1) {
					t.Fatalf("batch %d shift mismatch at (%d,%d)", b, r, i)
				}
			}
		}
	}
}

func TestPretrainLoaderCropsToFill(t *testing.T) {
	// Documents longer than T must be cropped; the row is always full.
	tok := simpleTokenizer()
	provider := cyclicProvider([]string{"aaaaaaaaaaaaaaaaaaaa"}) // 20 a's
	loader := NewPretrainLoader(tok, 1, 6, provider, 5)
	inputs, targets, _ := loader.Next()
	if inputs.Get2(0, 0) != int32(tok.BOSTokenID()) {
		t.Fatal("row must start with BOS")
	}
	for i := 0; i < 5; i++ {
		if targets.Get2(0, i) != inputs.Get2(0, i+1) {
			t.Fatalf("shift mismatch at %d", i)
		}
	}
}

func finiteConvProvider(convs []*tokenizer.Conversation) ConvProvider {
	i := 0
	return func() ([]*tokenizer.Conversation, bool) {
		if i >= len(convs) {
			return nil, false
		}
		c := convs[i]
		i++
		return []*tokenizer.Conversation{c}, true
	}
}

func TestSFTLoaderMasking(t *testing.T) {
	tok := simpleTokenizer()
	conv := &tokenizer.Conversation{Messages: []tokenizer.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}
	loader := NewSFTLoader(tok, 1, 32, finiteConvProvider([]*tokenizer.Conversation{conv}), 5)

	inputs, targets, ok := loader.Next()
	if !ok {
		t.Fatal("expected a batch")
	}
	// Reconstruct the expected rendered ids.
	wantIDs, _ := tok.RenderConversation(conv, 1<<30)
	if int(inputs.Numel()) != 32 {
		t.Fatalf("inputs numel = %d, want 32", inputs.Numel())
	}
	// The first len(wantIDs) inputs should match the rendered ids.
	for i := 0; i < len(wantIDs); i++ {
		if inputs.Data[i] != int32(wantIDs[i]) {
			t.Fatalf("input %d = %d, want %d", i, inputs.Data[i], wantIDs[i])
		}
	}
	// Padding inputs after the conversation should be BOS.
	bos := int32(tok.BOSTokenID())
	for i := len(wantIDs); i < 32; i++ {
		if inputs.Data[i] != bos {
			t.Fatalf("padding input %d = %d, want BOS", i, inputs.Data[i])
		}
	}
	// The first target (BOS->user_start shift) must be masked -1 since mask[1]==0.
	// Verify at least one -1 target exists (non-assistant positions).
	sawMask := false
	sawValid := false
	for i := 0; i < 32; i++ {
		if targets.Data[i] == -1 {
			sawMask = true
		} else {
			sawValid = true
		}
	}
	if !sawMask || !sawValid {
		t.Fatalf("expected both masked and valid targets, sawMask=%v sawValid=%v", sawMask, sawValid)
	}
}

func TestSFTLoaderExhaustion(t *testing.T) {
	tok := simpleTokenizer()
	conv := &tokenizer.Conversation{Messages: []tokenizer.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}
	loader := NewSFTLoader(tok, 1, 32, finiteConvProvider([]*tokenizer.Conversation{conv}), 5)

	// Keep pulling until exhausted.
	pulled := 0
	for {
		_, _, ok := loader.Next()
		if !ok {
			break
		}
		pulled++
		if pulled > 100 {
			t.Fatal("loader did not terminate")
		}
	}
}
