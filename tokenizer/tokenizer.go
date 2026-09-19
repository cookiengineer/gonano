package tokenizer

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"sort"

	"github.com/cookiengineer/gonano/internal/parallel"
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

	tok := &Tokenizer{
		mergeableRanks: mergeableRanks,
		specialIDs:     make(map[string]int, len(specialTokens)),
		specialNames:   make(map[int]string, len(specialTokens)),
		tokenBytes:     make([][]byte, vocabSize),
		vocabSize:      vocabSize,
	}
	for tokenText, rank := range mergeableRanks {
		tok.tokenBytes[rank] = []byte(tokenText)
	}
	for index, name := range specialTokens {
		id := vocabNoSpecial + index
		tok.specialIDs[name] = id
		tok.specialNames[id] = name
		tok.tokenBytes[id] = []byte(name)
	}
	tok.bosTokenID = tok.specialIDs["<|bos|>"]
	return tok
}

// VocabSize returns the total vocabulary size (mergeable + special).
func (tokenizer *Tokenizer) VocabSize() int { return tokenizer.vocabSize }

// BOSTokenID returns the id of the <|bos|> token.
func (tokenizer *Tokenizer) BOSTokenID() int { return tokenizer.bosTokenID }

// EncodeSpecial returns the id of a single special token by name.
func (tokenizer *Tokenizer) EncodeSpecial(name string) int {
	id, ok := tokenizer.specialIDs[name]
	if !ok {
		panic("tokenizer: unknown special token " + name)
	}
	return id
}

// IsSpecial reports whether id is a special token.
func (tokenizer *Tokenizer) IsSpecial(id int) bool {
	_, ok := tokenizer.specialNames[id]
	return ok
}

// SpecialTokenIDs returns the sorted ids of all special tokens.
func (tokenizer *Tokenizer) SpecialTokenIDs() []int {
	ids := make([]int, 0, len(tokenizer.specialIDs))
	for _, id := range tokenizer.specialIDs {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}

// Encode tokenizes text into token ids using the split pattern and BPE merge.
func (tokenizer *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, piece := range SplitPieces(text) {
		for _, part := range tokenizer.bpeMergePiece([]byte(piece)) {
			ids = append(ids, tokenizer.mergeableRanks[string(part)])
		}
	}
	return ids
}

// EncodeBatch tokenizes many texts in parallel. It is goroutine-safe.
func (tokenizer *Tokenizer) EncodeBatch(texts []string) [][]int {
	out := make([][]int, len(texts))
	parallel.Default().For(0, len(texts), func(index int) {
		out[index] = tokenizer.Encode(texts[index])
	})
	return out
}

// Decode reconstructs the byte string for a sequence of token ids. Special
// tokens decode to their literal name; unknown ids are skipped.
func (tokenizer *Tokenizer) Decode(ids []int) string {
	buf := make([]byte, 0, len(ids))
	for _, id := range ids {
		if id >= 0 && id < len(tokenizer.tokenBytes) {
			buf = append(buf, tokenizer.tokenBytes[id]...)
		}
	}
	return string(buf)
}

// DecodeSingleTokenBytes returns the raw bytes of a single token id.
func (tokenizer *Tokenizer) DecodeSingleTokenBytes(id int) []byte {
	if id >= 0 && id < len(tokenizer.tokenBytes) {
		return tokenizer.tokenBytes[id]
	}
	return nil
}

// ByteCount returns the number of bytes a token represents, or 0 for unknown
// ids and special tokens (special tokens carry no bytes for loss accounting).
func (tokenizer *Tokenizer) ByteCount(id int) int {
	if tokenizer.IsSpecial(id) {
		return 0
	}
	return len(tokenizer.DecodeSingleTokenBytes(id))
}

// bpeMergePiece greedily merges a piece into mergeable tokens using the ranks
// (lowest rank = earliest merge wins).
func (tokenizer *Tokenizer) bpeMergePiece(piece []byte) [][]byte {
	parts := make([][]byte, len(piece))
	for index, currentByte := range piece {
		parts[index] = []byte{currentByte}
	}
	var buf []byte
	for {
		bestIdx := -1
		bestRank := math.MaxInt
		for index := 0; index+1 < len(parts); index++ {
			buf = buf[:0]
			buf = append(buf, parts[index]...)
			buf = append(buf, parts[index+1]...)
			if rank, ok := tokenizer.mergeableRanks[string(buf)]; ok && rank < bestRank {
				bestRank = rank
				bestIdx = index
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
	Version  int
	Ranks    []rankEntry
	Specials []string
}

type rankEntry struct {
	Rank  int    `json:"rank"`
	Bytes string `json:"bytes"` // hex-encoded token bytes
}

// Save writes the tokenizer to path as JSON.
func (tokenizer *Tokenizer) Save(path string) error {
	saved := serialized{Version: 1, Specials: append([]string(nil), SpecialTokens...)}
	for tokenText, rank := range tokenizer.mergeableRanks {
		saved.Ranks = append(saved.Ranks, rankEntry{Rank: rank, Bytes: hex.EncodeToString([]byte(tokenText))})
	}
	sort.Slice(saved.Ranks, func(first, second int) bool { return saved.Ranks[first].Rank < saved.Ranks[second].Rank })
	data, err := json.MarshalIndent(saved, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

// LoadTokenizer reads a tokenizer saved by Save.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var saved serialized
	if err := json.Unmarshal(data, &saved); err != nil {
		return nil, err
	}
	ranks := make(map[string]int, len(saved.Ranks))
	for _, entry := range saved.Ranks {
		raw, err := hex.DecodeString(entry.Bytes)
		if err != nil {
			return nil, err
		}
		ranks[string(raw)] = entry.Rank
	}
	return NewTokenizer(ranks, saved.Specials), nil
}
