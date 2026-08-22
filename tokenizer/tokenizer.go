package tokenizer

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"sort"

	"github.com/cookiengineer/gonano/parallel"
)

// Tokenizer is a byte-level BPE tokenizer: it encodes text to token ids and
// decodes ids back to bytes. Special tokens are assigned ids after the
// mergeable vocabulary, exactly as nanochat orders them.
type Tokenizer struct {
	// mergeableRanks maps token bytes to their id (rank). Byte tokens are
	// 0..255; merged tokens continue upward.
	mergeableRanks map[string]int
	// specialIDs maps a special token name to its id.
	specialIDs map[string]int
	// specialNames maps a special id to its name.
	specialNames map[int]string
	// tokenBytes maps a token id to its raw bytes (used for decode).
	tokenBytes [][]byte
	vocabSize  int
	bosTokenID int
}

// NewTokenizer builds a tokenizer from mergeable ranks (bytes -> id) and the
// ordered special-token list. Special tokens are assigned ids starting at
// len(mergeableRanks).
func NewTokenizer(mergeableRanks map[string]int, specialTokens []string) *Tokenizer {
	vocabNoSpecial := len(mergeableRanks)
	vocabSize := vocabNoSpecial + len(specialTokens)

	t := &Tokenizer{
		mergeableRanks: mergeableRanks,
		specialIDs:     make(map[string]int, len(specialTokens)),
		specialNames:   make(map[int]string, len(specialTokens)),
		tokenBytes:     make([][]byte, vocabSize),
		vocabSize:      vocabSize,
	}
	for bytes, rank := range mergeableRanks {
		t.tokenBytes[rank] = []byte(bytes)
	}
	for i, name := range specialTokens {
		id := vocabNoSpecial + i
		t.specialIDs[name] = id
		t.specialNames[id] = name
		t.tokenBytes[id] = []byte(name)
	}
	t.bosTokenID = t.specialIDs["<|bos|>"]
	return t
}

// VocabSize returns the total vocabulary size (mergeable + special).
func (t *Tokenizer) VocabSize() int { return t.vocabSize }

// BOSTokenID returns the id of the <|bos|> token.
func (t *Tokenizer) BOSTokenID() int { return t.bosTokenID }

// EncodeSpecial returns the id of a single special token by name.
func (t *Tokenizer) EncodeSpecial(name string) int {
	id, ok := t.specialIDs[name]
	if !ok {
		panic("tokenizer: unknown special token " + name)
	}
	return id
}

// IsSpecial reports whether id is a special token.
func (t *Tokenizer) IsSpecial(id int) bool {
	_, ok := t.specialNames[id]
	return ok
}

// SpecialTokenIDs returns the sorted ids of all special tokens.
func (t *Tokenizer) SpecialTokenIDs() []int {
	ids := make([]int, 0, len(t.specialIDs))
	for _, id := range t.specialIDs {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

// Encode tokenizes text into token ids using the split pattern and BPE merge.
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, piece := range SplitPieces(text) {
		for _, part := range t.bpeMergePiece([]byte(piece)) {
			ids = append(ids, t.mergeableRanks[string(part)])
		}
	}
	return ids
}

// EncodeBatch tokenizes many texts in parallel. It is goroutine-safe.
func (t *Tokenizer) EncodeBatch(texts []string) [][]int {
	out := make([][]int, len(texts))
	parallel.Default().For(0, len(texts), func(i int) {
		out[i] = t.Encode(texts[i])
	})
	return out
}

// Decode reconstructs the byte string for a sequence of token ids. Special
// tokens decode to their literal name; unknown ids are skipped.
func (t *Tokenizer) Decode(ids []int) string {
	buf := make([]byte, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && id < len(t.tokenBytes) {
			buf = append(buf, t.tokenBytes[id]...)
		}
	}
	return string(buf)
}

// DecodeSingleTokenBytes returns the raw bytes of a single token id.
func (t *Tokenizer) DecodeSingleTokenBytes(id int) []byte {
	if id >= 0 && id < len(t.tokenBytes) {
		return t.tokenBytes[id]
	}
	return nil
}

// ByteCount returns the number of bytes a token represents, or 0 for unknown
// ids and special tokens (special tokens carry no bytes for loss accounting).
func (t *Tokenizer) ByteCount(id int) int {
	if t.IsSpecial(id) {
		return 0
	}
	return len(t.DecodeSingleTokenBytes(id))
}

// bpeMergePiece greedily merges a piece into mergeable tokens using the ranks
// (lowest rank = earliest merge wins).
func (t *Tokenizer) bpeMergePiece(piece []byte) [][]byte {
	parts := make([][]byte, len(piece))
	for i, b := range piece {
		parts[i] = []byte{b}
	}
	var buf []byte
	for {
		bestIdx := -1
		bestRank := math.MaxInt
		for i := 0; i+1 < len(parts); i++ {
			buf = buf[:0]
			buf = append(buf, parts[i]...)
			buf = append(buf, parts[i+1]...)
			if r, ok := t.mergeableRanks[string(buf)]; ok && r < bestRank {
				bestRank = r
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		parts[bestIdx] = append(parts[bestIdx], parts[bestIdx+1]...)
		parts = append(parts[:bestIdx+1], parts[bestIdx+2:]...)
	}
	return parts
}

// serialized is the on-disk JSON format for a tokenizer.
type serialized struct {
	Version int
	Ranks   []rankEntry
	Specials []string
}

type rankEntry struct {
	Rank  int    `json:"rank"`
	Bytes string `json:"bytes"` // hex-encoded token bytes
}

// Save writes the tokenizer to path as JSON.
func (t *Tokenizer) Save(path string) error {
	s := serialized{Version: 1, Specials: append([]string(nil), SpecialTokens...)}
	for bytes, rank := range t.mergeableRanks {
		s.Ranks = append(s.Ranks, rankEntry{Rank: rank, Bytes: hex.EncodeToString([]byte(bytes))})
	}
	sort.Slice(s.Ranks, func(i, j int) bool { return s.Ranks[i].Rank < s.Ranks[j].Rank })
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// LoadTokenizer reads a tokenizer saved by Save.
func LoadTokenizer(path string) (*Tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s serialized
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	ranks := make(map[string]int, len(s.Ranks))
	for _, e := range s.Ranks {
		raw, err := hex.DecodeString(e.Bytes)
		if err != nil {
			return nil, err
		}
		ranks[string(raw)] = e.Rank
	}
	return NewTokenizer(ranks, s.Specials), nil
}
