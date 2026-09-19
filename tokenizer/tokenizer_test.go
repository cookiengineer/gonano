package tokenizer

import (
	"reflect"
	"testing"
)

func TestSplitPiecesKnown(tests *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"hello, world", []string{"hello", ",", " world"}},
		{"12345", []string{"12", "34", "5"}},
		{"I'm", []string{"I", "'m"}},
		{"don't", []string{"don", "'t"}},
		{"a\nb", []string{"a", "\n", "b"}},
		{"a\n\nb", []string{"a", "\n\n", "b"}},
		{"(abc)", []string{"(abc", ")"}},
		{"hey!", []string{"hey", "!"}},
		{"  spaced  ", []string{"  ", "spaced", "  "}},
		{"tab\there", []string{"tab", "\there"}},
		{"a \n b", []string{"a", " \n", " b"}},
	}
	for _, testCase := range cases {
		got := SplitPieces(testCase.in)
		if !reflect.DeepEqual(got, testCase.want) {
			tests.Errorf("splitPieces(%q) = %q, want %q", testCase.in, got, testCase.want)
		}
	}
}

func TestSplitPiecesCoverage(tests *testing.T) {
	// Every character of the input must appear in exactly the same order in
	// the concatenated pieces (no loss, no duplication).
	inputs := []string{
		"The quick brown fox jumps over the lazy dog.",
		"Numbers: 123, 4567, 89. Done!",
		"Contractions: I'm, you're, it's, we'll, they've.",
		"Special chars: @#$%^&*()_+=-[]{}|;:,.<>?/~`",
		"Unicode: 你好世界 🌍 café naïve",
		"newline\nmixed\r\nwindows\rtabs\tand  spaces\n\n\n",
		"",
		"a",
	}
	for _, input := range inputs {
		pieces := SplitPieces(input)
		var joined string
		for _, piece := range pieces {
			joined += piece
		}
		if joined != input {
			tests.Errorf("splitPieces(%q) joined = %q, want original", input, joined)
		}
	}
}

func TestTrainBPEAndRoundtrip(tests *testing.T) {
	pieces := []string{
		"the quick brown fox",
		"the quick brown fox jumps",
		"quick brown",
		"fox",
	}
	ranks := TrainBPE(pieces, 10)
	if len(ranks) != 256+10 {
		tests.Fatalf("ranks size = %d, want %d", len(ranks), 256+10)
	}
	tok := NewTokenizer(ranks, SpecialTokens)
	for _, piece := range pieces {
		ids := tok.Encode(piece)
		back := tok.Decode(ids)
		if back != piece {
			tests.Errorf("roundtrip %q -> %q", piece, back)
		}
	}
}

func TestTrainBPEMergesMostFrequentPair(tests *testing.T) {
	// "aa" appears twice, "b" once. The first merge must produce "aa".
	ranks := TrainBPE([]string{"aa", "aa", "b"}, 1)
	if _, ok := ranks["aa"]; !ok {
		tests.Fatal("expected 'aa' merge to be produced first")
	}
	tok := NewTokenizer(ranks, SpecialTokens)
	if got := tok.Encode("aa"); len(got) != 1 {
		tests.Fatalf("encode(aa) = %v, want single token", got)
	}
	if got := tok.Decode(tok.Encode("aa")); got != "aa" {
		tests.Fatalf("roundtrip aa = %q", got)
	}
}

func TestTokenizerEncodeDecodeUnicode(tests *testing.T) {
	ranks := TrainBPE([]string{"hello world", "你好世界", "12345"}, 20)
	tok := NewTokenizer(ranks, SpecialTokens)
	for _, text := range []string{
		"hello world",
		"你好世界",
		"12345",
		"mixed 你好 and 123",
	} {
		if got := tok.Decode(tok.Encode(text)); got != text {
			tests.Errorf("roundtrip %q -> %q", text, got)
		}
	}
}

func TestSpecialTokens(tests *testing.T) {
	ranks := TrainBPE([]string{"abcdefghij"}, 5)
	tok := NewTokenizer(ranks, SpecialTokens)
	if tok.VocabSize() != len(ranks)+len(SpecialTokens) {
		tests.Fatalf("vocab = %d, want %d", tok.VocabSize(), len(ranks)+len(SpecialTokens))
	}
	if len(ranks) != 256+5 {
		tests.Fatalf("ranks = %d, want %d", len(ranks), 256+5)
	}
	bos := tok.EncodeSpecial("<|bos|>")
	if bos != tok.BOSTokenID() {
		tests.Fatal("bos id mismatch")
	}
	if !tok.IsSpecial(bos) {
		tests.Fatal("bos should be special")
	}
	// Decode of a special token yields its literal name.
	if tok.Decode([]int{bos}) != "<|bos|>" {
		tests.Fatal("special decode mismatch")
	}
	// Special token byte count is zero.
	if tok.ByteCount(bos) != 0 {
		tests.Fatal("special byte count should be 0")
	}
}

func TestSaveLoad(tests *testing.T) {
	ranks := TrainBPE([]string{"the quick brown fox", "jumps over"}, 20)
	tok := NewTokenizer(ranks, SpecialTokens)
	path := tests.TempDir() + "/tokenizer.json"
	if err := tok.Save(path); err != nil {
		tests.Fatalf("Save: %v", err)
	}
	loaded, err := LoadTokenizer(path)
	if err != nil {
		tests.Fatalf("Load: %v", err)
	}
	for _, text := range []string{"the quick", "brown fox jumps"} {
		if !reflect.DeepEqual(tok.Encode(text), loaded.Encode(text)) {
			tests.Errorf("encode(%q) differs after roundtrip", text)
		}
	}
	if loaded.VocabSize() != tok.VocabSize() {
		tests.Fatal("vocab size differs after roundtrip")
	}
}
