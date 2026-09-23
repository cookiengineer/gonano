package data

import (
	"testing"

	"github.com/cookiengineer/gonano/tokenizer"
)

// newSimpleTokenizer builds a byte-level tokenizer with no merges (256 byte
// tokens + special tokens), so encode maps each byte to its id.
func newSimpleTokenizer() *tokenizer.Tokenizer {
	ranks := make(map[string]int, 256)
	for byteValue := 0; byteValue < 256; byteValue++ {
		ranks[string([]byte{byte(byteValue)})] = byteValue
	}
	return tokenizer.NewTokenizer(ranks, tokenizer.SpecialTokens)
}

func newCyclicProvider(docs []string) DocProvider {
	position := 0
	return func() ([]string, State) {
		batch := []string{docs[position%len(docs)]}
		position++
		return batch, State{PQIndex: 0, RGIndex: 0, Epoch: 1}
	}
}

func TestPretrainLoaderBOSAligned(tests *testing.T) {
	tok := newSimpleTokenizer()
	bos := tok.BOSTokenID()
	provider := newCyclicProvider([]string{"hello", "world", "hi", "there"})
	loader := NewPretrainLoader(tok, 2, 8, provider, 10)

	inputs, targets, _ := loader.Next()
	if inputs.Shape[0] != 2 || inputs.Shape[1] != 8 {
		tests.Fatalf("inputs shape = %v, want [2 8]", inputs.Shape)
	}
	// Every row starts with BOS.
	for row := 0; row < 2; row++ {
		if inputs.Get2(row, 0) != int32(bos) {
			tests.Fatalf("row %d does not start with BOS: %d", row, inputs.Get2(row, 0))
		}
	}
	// targets[i] == inputs[i+1] (autoregressive shift).
	for row := 0; row < 2; row++ {
		for index := 0; index < 7; index++ {
			if targets.Get2(row, index) != inputs.Get2(row, index+1) {
				tests.Fatalf("shift mismatch at (%d,%d)", row, index)
			}
		}
	}
	// All ids are valid.
	vocab := tok.VocabSize()
	for row := 0; row < 2; row++ {
		for index := 0; index < 8; index++ {
			if int(inputs.Get2(row, index)) >= vocab {
				tests.Fatalf("invalid input id %d at (%d,%d)", inputs.Get2(row, index), row, index)
			}
		}
	}
}

func TestPretrainLoaderMultipleBatches(tests *testing.T) {
	tok := newSimpleTokenizer()
	provider := newCyclicProvider([]string{"a", "b", "c"})
	loader := NewPretrainLoader(tok, 3, 4, provider, 5)

	// Pull several batches; each must be BOS-aligned and consistently shifted.
	for batch := 0; batch < 10; batch++ {
		inputs, targets, _ := loader.Next()
		for row := 0; row < 3; row++ {
			if inputs.Get2(row, 0) != int32(tok.BOSTokenID()) {
				tests.Fatalf("batch %d row %d not BOS-aligned", batch, row)
			}
			for index := 0; index < 3; index++ {
				if targets.Get2(row, index) != inputs.Get2(row, index+1) {
					tests.Fatalf("batch %d shift mismatch at (%d,%d)", batch, row, index)
				}
			}
		}
	}
}

func TestPretrainLoaderCropsToFill(tests *testing.T) {
	// Documents longer than T must be cropped; the row is always full.
	tok := newSimpleTokenizer()
	provider := newCyclicProvider([]string{"aaaaaaaaaaaaaaaaaaaa"}) // 20 a's
	loader := NewPretrainLoader(tok, 1, 6, provider, 5)
	inputs, targets, _ := loader.Next()
	if inputs.Get2(0, 0) != int32(tok.BOSTokenID()) {
		tests.Fatal("row must start with BOS")
	}
	for index := 0; index < 5; index++ {
		if targets.Get2(0, index) != inputs.Get2(0, index+1) {
			tests.Fatalf("shift mismatch at %d", index)
		}
	}
}

// TestPretrainLoaderSegmentsAlignWithBOS checks that the segment ids returned by
// NextSegments change exactly at document boundaries (each packed document
// starts with BOS) and that the inputs/targets match Next.
func TestPretrainLoaderSegmentsAlignWithBOS(tests *testing.T) {
	tok := newSimpleTokenizer()
	bos := int32(tok.BOSTokenID())
	provider := newCyclicProvider([]string{"hello", "world", "hi", "there", "ok"})
	loader := NewPretrainLoader(tok, 3, 12, provider, 10)

	inputs, targets, segments, _ := loader.NextSegments()
	if segments.Shape[0] != inputs.Shape[0] || segments.Shape[1] != inputs.Shape[1] {
		tests.Fatalf("segments shape = %v, want %v", segments.Shape, inputs.Shape)
	}
	for row := 0; row < inputs.Shape[0]; row++ {
		if segments.Get2(row, 0) != 0 {
			tests.Fatalf("row %d must start in segment 0, got %d", row, segments.Get2(row, 0))
		}
		for index := 1; index < inputs.Shape[1]; index++ {
			previous := segments.Get2(row, index-1)
			current := segments.Get2(row, index)
			if inputs.Get2(row, index) == bos {
				if current != previous+1 {
					tests.Fatalf("row %d index %d: BOS should start a new segment (%d -> %d)", row, index, previous, current)
				}
			} else if current != previous {
				tests.Fatalf("row %d index %d: non-BOS token changed segment (%d -> %d)", row, index, previous, current)
			}
		}
	}

	// The inputs/targets of Next must be identical (Next is NextSegments
	// without the segment tensor); the same provider/loader state makes the
	// comparison valid only across a fresh loader, so use one.
	fresh := NewPretrainLoader(tok, 3, 12, newCyclicProvider([]string{"hello", "world", "hi", "there", "ok"}), 10)
	plainInputs, plainTargets, _ := fresh.Next()
	for index := range plainInputs.Data {
		if plainInputs.Data[index] != inputs.Data[index] || plainTargets.Data[index] != targets.Data[index] {
			tests.Fatalf("Next and NextSegments differ at %d", index)
		}
	}
}

func newFiniteConvProvider(convs []*tokenizer.Conversation) ConvProvider {
	position := 0
	return func() ([]*tokenizer.Conversation, bool) {
		if position >= len(convs) {
			return nil, false
		}
		conversation := convs[position]
		position++
		return []*tokenizer.Conversation{conversation}, true
	}
}

func TestSFTLoaderMasking(tests *testing.T) {
	tok := newSimpleTokenizer()
	conv := &tokenizer.Conversation{Messages: []tokenizer.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}
	loader := NewSFTLoader(tok, 1, 32, newFiniteConvProvider([]*tokenizer.Conversation{conv}), 5)

	inputs, targets, ok := loader.Next()
	if !ok {
		tests.Fatal("expected a batch")
	}
	// Reconstruct the expected rendered ids.
	wantIDs, _ := tok.RenderConversation(conv, 1<<30)
	if int(inputs.Numel()) != 32 {
		tests.Fatalf("inputs numel = %d, want 32", inputs.Numel())
	}
	// The first len(wantIDs) inputs should match the rendered ids.
	for index := 0; index < len(wantIDs); index++ {
		if inputs.Data[index] != int32(wantIDs[index]) {
			tests.Fatalf("input %d = %d, want %d", index, inputs.Data[index], wantIDs[index])
		}
	}
	// Padding inputs after the conversation should be BOS.
	bos := int32(tok.BOSTokenID())
	for index := len(wantIDs); index < 32; index++ {
		if inputs.Data[index] != bos {
			tests.Fatalf("padding input %d = %d, want BOS", index, inputs.Data[index])
		}
	}
	// The first target (BOS->user_start shift) must be masked -1 since mask[1]==0.
	// Verify at least one -1 target exists (non-assistant positions).
	sawMask := false
	sawValid := false
	for index := 0; index < 32; index++ {
		if targets.Data[index] == -1 {
			sawMask = true
		} else {
			sawValid = true
		}
	}
	if !sawMask || !sawValid {
		tests.Fatalf("expected both masked and valid targets, sawMask=%v sawValid=%v", sawMask, sawValid)
	}
}

func TestSFTLoaderExhaustion(tests *testing.T) {
	tok := newSimpleTokenizer()
	conv := &tokenizer.Conversation{Messages: []tokenizer.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}}
	loader := NewSFTLoader(tok, 1, 32, newFiniteConvProvider([]*tokenizer.Conversation{conv}), 5)

	// Keep pulling until exhausted.
	pulled := 0
	for {
		_, _, ok := loader.Next()
		if !ok {
			break
		}
		pulled++
		if pulled > 100 {
			tests.Fatal("loader did not terminate")
		}
	}
}
