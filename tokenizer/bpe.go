package tokenizer

import "container/heap"

// pair is an adjacent pair of byte symbols.
type pair [2]int

type pairEntry struct {
	symbolPair pair
	count      int
}

// pairHeap is a max-heap ordered by count (then by symbols for determinism).
type pairHeap []pairEntry

func (entries pairHeap) Len() int { return len(entries) }
func (entries pairHeap) Less(first, second int) bool {
	if entries[first].count != entries[second].count {
		return entries[first].count > entries[second].count
	}
	if entries[first].symbolPair[0] != entries[second].symbolPair[0] {
		return entries[first].symbolPair[0] < entries[second].symbolPair[0]
	}
	return entries[first].symbolPair[1] < entries[second].symbolPair[1]
}
func (entries pairHeap) Swap(first, second int) {
	entries[first], entries[second] = entries[second], entries[first]
}
func (entries *pairHeap) Push(value any) { *entries = append(*entries, value.(pairEntry)) }
func (entries *pairHeap) Pop() any {
	old := *entries
	count := len(old)
	value := old[count-1]
	*entries = old[:count-1]
	return value
}

// TrainBPE trains byte-level BPE over the given pieces (already split by the
// GPT-4 pattern). It performs numMerges merges and returns the mergeable
// ranks: a map from token bytes (as a string) to the token id (rank). The 256
// single-byte tokens are assigned ids 0..255; each merge adds the next id.
//
// The implementation maintains an inverted index of pair occurrences so that
// each merge updates only the pieces that actually contain the pair, making
// training tractable for large corpora.
func TrainBPE(pieces []string, numMerges int) map[string]int {
	symbols := make([][]int, len(pieces))
	for pieceIndex, piece := range pieces {
		symbols[pieceIndex] = make([]int, len(piece))
		for byteIndex := 0; byteIndex < len(piece); byteIndex++ {
			symbols[pieceIndex][byteIndex] = int(piece[byteIndex])
		}
	}

	tokenBytes := make([][]byte, 256+numMerges)
	for tokenID := 0; tokenID < 256; tokenID++ {
		tokenBytes[tokenID] = []byte{byte(tokenID)}
	}

	pairCount := map[pair]int{}
	pairMembers := map[pair]map[int]struct{}{}
	touched := map[pair]bool{}

	addPiecePairs := func(pieceIndex int) {
		sequence := symbols[pieceIndex]
		for pairIndex := 0; pairIndex+1 < len(sequence); pairIndex++ {
			symbolPair := pair{sequence[pairIndex], sequence[pairIndex+1]}
			pairCount[symbolPair]++
			touched[symbolPair] = true
			memberSet := pairMembers[symbolPair]
			if memberSet == nil {
				memberSet = map[int]struct{}{}
				pairMembers[symbolPair] = memberSet
			}
			memberSet[pieceIndex] = struct{}{}
		}
	}
	removePiecePairs := func(pieceIndex int) {
		sequence := symbols[pieceIndex]
		for pairIndex := 0; pairIndex+1 < len(sequence); pairIndex++ {
			symbolPair := pair{sequence[pairIndex], sequence[pairIndex+1]}
			pairCount[symbolPair]--
			touched[symbolPair] = true
			if memberSet := pairMembers[symbolPair]; memberSet != nil {
				delete(memberSet, pieceIndex)
				if len(memberSet) == 0 {
					delete(pairMembers, symbolPair)
				}
			}
		}
	}

	for pieceIndex := range symbols {
		addPiecePairs(pieceIndex)
	}

	mergeHeap := &pairHeap{}
	heap.Init(mergeHeap)
	for symbolPair, count := range pairCount {
		heap.Push(mergeHeap, pairEntry{symbolPair, count})
	}

	ranks := make(map[string]int, 256+numMerges)
	for tokenID := 0; tokenID < 256; tokenID++ {
		ranks[string(tokenBytes[tokenID])] = tokenID
	}

	for merge := 0; merge < numMerges; merge++ {
		var topEntry pairEntry
		found := false
		for mergeHeap.Len() > 0 {
			topEntry = heap.Pop(mergeHeap).(pairEntry)
			if pairCount[topEntry.symbolPair] == topEntry.count && topEntry.count > 0 {
				found = true
				break
			}
		}
		if !found {
			break // no more mergeable pairs
		}

		leftSymbol, rightSymbol := topEntry.symbolPair[0], topEntry.symbolPair[1]
		newSymbol := 256 + merge
		tokenBytes[newSymbol] = append(append([]byte(nil), tokenBytes[leftSymbol]...), tokenBytes[rightSymbol]...)
		ranks[string(tokenBytes[newSymbol])] = newSymbol

		// Snapshot the pieces containing this pair, then update them.
		members := pairMembers[topEntry.symbolPair]
		pieceList := make([]int, 0, len(members))
		for pieceIndex := range members {
			pieceList = append(pieceList, pieceIndex)
		}
		clear(touched)
		for _, pieceIndex := range pieceList {
			removePiecePairs(pieceIndex)
			symbols[pieceIndex] = mergePairs(symbols[pieceIndex], leftSymbol, rightSymbol, newSymbol)
			addPiecePairs(pieceIndex)
		}
		// Re-insert fresh entries for every pair whose count changed.
		for symbolPair := range touched {
			heap.Push(mergeHeap, pairEntry{symbolPair, pairCount[symbolPair]})
		}
	}
	return ranks
}

// mergePairs replaces all non-overlapping occurrences of (leftSymbol,
// rightSymbol) with mergedSymbol in a left-to-right pass, matching the
// standard BPE merge semantics.
func mergePairs(symbols []int, leftSymbol, rightSymbol, mergedSymbol int) []int {
	out := make([]int, 0, len(symbols))
	index := 0
	for index < len(symbols) {
		if index+1 < len(symbols) && symbols[index] == leftSymbol && symbols[index+1] == rightSymbol {
			out = append(out, mergedSymbol)
			index += 2
		} else {
			out = append(out, symbols[index])
			index++
		}
	}
	return out
}
