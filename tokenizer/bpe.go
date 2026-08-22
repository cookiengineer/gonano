package tokenizer

import "container/heap"

// pair is an adjacent pair of byte symbols.
type pair [2]int

type pairEntry struct {
	p     pair
	count int
}

// pairHeap is a max-heap ordered by count (then by symbols for determinism).
type pairHeap []pairEntry

func (h pairHeap) Len() int { return len(h) }
func (h pairHeap) Less(i, j int) bool {
	if h[i].count != h[j].count {
		return h[i].count > h[j].count
	}
	if h[i].p[0] != h[j].p[0] {
		return h[i].p[0] < h[j].p[0]
	}
	return h[i].p[1] < h[j].p[1]
}
func (h pairHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *pairHeap) Push(x any)        { *h = append(*h, x.(pairEntry)) }
func (h *pairHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
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
	for i, p := range pieces {
		symbols[i] = make([]int, len(p))
		for j := 0; j < len(p); j++ {
			symbols[i][j] = int(p[j])
		}
	}

	tokenBytes := make([][]byte, 256+numMerges)
	for i := 0; i < 256; i++ {
		tokenBytes[i] = []byte{byte(i)}
	}

	pairCount := map[pair]int{}
	pairMembers := map[pair]map[int]struct{}{}
	touched := map[pair]bool{}

	addPiecePairs := func(pi int) {
		s := symbols[pi]
		for j := 0; j+1 < len(s); j++ {
			p := pair{s[j], s[j+1]}
			pairCount[p]++
			touched[p] = true
			m := pairMembers[p]
			if m == nil {
				m = map[int]struct{}{}
				pairMembers[p] = m
			}
			m[pi] = struct{}{}
		}
	}
	removePiecePairs := func(pi int) {
		s := symbols[pi]
		for j := 0; j+1 < len(s); j++ {
			p := pair{s[j], s[j+1]}
			pairCount[p]--
			touched[p] = true
			if m := pairMembers[p]; m != nil {
				delete(m, pi)
				if len(m) == 0 {
					delete(pairMembers, p)
				}
			}
		}
	}

	for pi := range symbols {
		addPiecePairs(pi)
	}

	h := &pairHeap{}
	heap.Init(h)
	for p, c := range pairCount {
		heap.Push(h, pairEntry{p, c})
	}

	ranks := make(map[string]int, 256+numMerges)
	for i := 0; i < 256; i++ {
		ranks[string(tokenBytes[i])] = i
	}

	for merge := 0; merge < numMerges; merge++ {
		var pe pairEntry
		ok := false
		for h.Len() > 0 {
			pe = heap.Pop(h).(pairEntry)
			if pairCount[pe.p] == pe.count && pe.count > 0 {
				ok = true
				break
			}
		}
		if !ok {
			break // no more mergeable pairs
		}

		a, b := pe.p[0], pe.p[1]
		c := 256 + merge
		tokenBytes[c] = append(append([]byte(nil), tokenBytes[a]...), tokenBytes[b]...)
		ranks[string(tokenBytes[c])] = c

		// Snapshot the pieces containing this pair, then update them.
		members := pairMembers[pe.p]
		pieceList := make([]int, 0, len(members))
		for pi := range members {
			pieceList = append(pieceList, pi)
		}
		clear(touched)
		for _, pi := range pieceList {
			removePiecePairs(pi)
			symbols[pi] = mergePairs(symbols[pi], a, b, c)
			addPiecePairs(pi)
		}
		// Re-insert fresh entries for every pair whose count changed.
		for p := range touched {
			heap.Push(h, pairEntry{p, pairCount[p]})
		}
	}
	return ranks
}

// mergePairs replaces all non-overlapping occurrences of (a, b) with c in a
// left-to-right pass, matching the standard BPE merge semantics.
func mergePairs(s []int, a, b, c int) []int {
	out := make([]int, 0, len(s))
	i := 0
	for i < len(s) {
		if i+1 < len(s) && s[i] == a && s[i+1] == b {
			out = append(out, c)
			i += 2
		} else {
			out = append(out, s[i])
			i++
		}
	}
	return out
}
