package inference

import (
	"sort"
	"sync"

	"github.com/cookiengineer/gonano/tokenizer"
)

// tokenTrieNode is one node in the compact token-byte trie. Edges live in a
// shared slice so nodes stay small even for 128k-token vocabularies.
type tokenTrieNode struct {
	edgeStart  int32
	edgeCount  int32
	tokenStart int32
	tokenCount int32
}

type tokenTrieEdge struct {
	label byte
	child int32
}

// tokenMaskTrie is an immutable byte trie over the tokenizer's vocabulary.
type tokenMaskTrie struct {
	nodes       []tokenTrieNode
	edges       []tokenTrieEdge
	tokens      []int32
	emptyTokens []int32
}

var tokenMaskTrieStore sync.Map

type temporaryTrieNode struct {
	children map[byte]int
	tokens   []int32
}

// tokenTrieFor returns the cached trie for a tokenizer, building it on first
// use. Tokenizers are long-lived, so the pointer is a stable cache key.
func tokenTrieFor(tokenizerImpl *tokenizer.Tokenizer) *tokenMaskTrie {
	if cached, ok := tokenMaskTrieStore.Load(tokenizerImpl); ok {
		return cached.(*tokenMaskTrie)
	}
	trie := buildTokenMaskTrie(tokenizerImpl)
	actual, _ := tokenMaskTrieStore.LoadOrStore(tokenizerImpl, trie)
	return actual.(*tokenMaskTrie)
}

func buildTokenMaskTrie(tokenizerImpl *tokenizer.Tokenizer) *tokenMaskTrie {
	temporary := []temporaryTrieNode{{children: make(map[byte]int)}}
	emptyTokens := []int32{}
	vocab := tokenizerImpl.VocabSize()
	for id := 0; id < vocab; id++ {
		if tokenizerImpl.IsSpecial(id) {
			continue
		}
		tokenBytes := tokenizerImpl.DecodeSingleTokenBytes(id)
		if len(tokenBytes) == 0 {
			emptyTokens = append(emptyTokens, int32(id))
			continue
		}
		node := 0
		for _, value := range tokenBytes {
			child, ok := temporary[node].children[value]
			if !ok {
				child = len(temporary)
				temporary[node].children[value] = child
				temporary = append(temporary, temporaryTrieNode{children: make(map[byte]int)})
			}
			node = child
		}
		temporary[node].tokens = append(temporary[node].tokens, int32(id))
	}
	trie := &tokenMaskTrie{emptyTokens: emptyTokens}
	compactTokenTrie(temporary, 0, trie)
	return trie
}

// compactTokenTrie copies the temporary map-based trie into flat node, edge and
// token arrays. Parent edges are reserved before descending so a node's outgoing
// edges stay contiguous even though child edges are appended first.
func compactTokenTrie(temporary []temporaryTrieNode, index int, trie *tokenMaskTrie) int32 {
	nodeIndex := int32(len(trie.nodes))
	trie.nodes = append(trie.nodes, tokenTrieNode{})

	if tokens := temporary[index].tokens; len(tokens) > 0 {
		trie.nodes[nodeIndex].tokenStart = int32(len(trie.tokens))
		trie.nodes[nodeIndex].tokenCount = int32(len(tokens))
		trie.tokens = append(trie.tokens, tokens...)
	}

	labels := make([]byte, 0, len(temporary[index].children))
	for label := range temporary[index].children {
		labels = append(labels, label)
	}
	sort.Slice(labels, func(left, right int) bool { return labels[left] < labels[right] })

	edgeStart := int32(len(trie.edges))
	trie.edges = append(trie.edges, make([]tokenTrieEdge, len(labels))...)
	for offset, label := range labels {
		child := compactTokenTrie(temporary, temporary[index].children[label], trie)
		trie.edges[edgeStart+int32(offset)] = tokenTrieEdge{label: label, child: child}
	}
	trie.nodes[nodeIndex].edgeStart = edgeStart
	trie.nodes[nodeIndex].edgeCount = int32(len(labels))
	return nodeIndex
}

// constraintMaskFromTrie builds the mask by walking the token trie and advancing
// a cloned constraint once per visited byte edge. An invalid edge prunes its
// entire subtree, so tokens that share a valid prefix share the work.
func constraintMaskFromTrie(tokenizerImpl *tokenizer.Tokenizer, trie *tokenMaskTrie, constraint Constraint, vocab int, stopTokens []int) []bool {
	mask := make([]bool, vocab)
	finished := constraint.Finished()
	if finished {
		for _, id := range stopTokens {
			if id >= 0 && id < vocab && tokenizerImpl.IsSpecial(id) {
				mask[id] = true
			}
		}
	}
	for _, id := range trie.emptyTokens {
		if id >= 0 && int(id) < vocab {
			mask[int(id)] = true
		}
	}

	type trieFrame struct {
		node  int32
		state Constraint
	}
	stack := []trieFrame{{node: 0, state: constraint}}
	probe := [1]byte{}
	for len(stack) > 0 {
		frame := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		node := trie.nodes[frame.node]
		for offset := int32(0); offset < node.tokenCount; offset++ {
			id := int(trie.tokens[node.tokenStart+offset])
			if id >= 0 && id < vocab {
				mask[id] = true
			}
		}
		for offset := int32(0); offset < node.edgeCount; offset++ {
			edge := trie.edges[node.edgeStart+offset]
			probe[0] = edge.label
			next := frame.state.Clone()
			if !next.Allowed(probe[:]) {
				continue
			}
			next.Advance(probe[:])
			stack = append(stack, trieFrame{node: edge.child, state: next})
		}
	}
	return mask
}
