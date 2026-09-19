package data

import "github.com/cookiengineer/gonano/tokenizer"

// TrainTokenizer trains a byte-level BPE tokenizer over a document provider
// (Parquet or Markdown), collecting up to maxChars characters. It stops after
// one full epoch of the dataset or once maxChars is reached, whichever comes
// first. The returned map is the mergeable ranks (token bytes -> id).
func TrainTokenizer(provider DocProvider, vocabSize, maxChars int) map[string]int {
	numMerges := vocabSize - 256 - len(tokenizer.SpecialTokens)
	if numMerges < 0 {
		numMerges = 0
	}

	var pieces []string
	total := 0
	startEpoch := -1
	for {
		docs, state := provider()
		if docs == nil {
			break // empty source
		}
		if startEpoch < 0 {
			startEpoch = state.Epoch
		} else if state.Epoch > startEpoch {
			break // wrapped around: we have seen the whole dataset once
		}
		for _, doc := range docs {
			for _, piece := range tokenizer.SplitPieces(doc) {
				pieces = append(pieces, piece)
				total += len(piece)
			}
		}
		if total >= maxChars {
			break
		}
	}
	return tokenizer.TrainBPE(pieces, numMerges)
}
