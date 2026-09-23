package inference

import (
	"sync"

	"github.com/cookiengineer/gonano/model"
)

// PrefixCache is a single-entry, in-memory KV prefix cache for autoregressive
// serving. It snapshots the KV buffer produced by a prefill and reuses it when
// a later request strictly extends the same token prefix (the common multi-turn
// pattern: a conversation is appended to).
//
// The cache is intentionally conservative: it only hits when the stored token
// sequence is an exact prefix of the new request and at least one token remains
// to be prefilled, which keeps the reused KV state exactly consistent with a
// full prefill. CED models are not cached because their decoder global KV is
// derived from the full encoder hidden state; callers should keep the cache
// disabled for them.
//
// It is safe for concurrent use.
type PrefixCache struct {
	mutex     sync.Mutex
	maxPrefix int
	tokens    []int
	state     *model.KVBuffer
}

// NewPrefixCache returns a prefix cache that stores at most maxPrefix tokens. A
// non-positive maxPrefix is treated as unbounded.
func NewPrefixCache(maxPrefix int) *PrefixCache {
	return &PrefixCache{maxPrefix: maxPrefix}
}

// Lookup returns a clone of the stored KV state and the number of matched
// prefix tokens when the stored sequence is a strict prefix of tokens. It
// returns nil, 0 on a miss.
func (cache *PrefixCache) Lookup(tokens []int) (*model.KVBuffer, int) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	if cache.state == nil {
		return nil, 0
	}
	if cache.maxPrefix > 0 && len(cache.tokens) > cache.maxPrefix {
		return nil, 0
	}
	// The stored sequence must be a strict prefix: at least one token of the
	// new request must remain to be prefilled, so the caller always has logits
	// for the last prompt token.
	if len(cache.tokens) == 0 || len(cache.tokens) >= len(tokens) {
		return nil, 0
	}
	for index := range cache.tokens {
		if cache.tokens[index] != tokens[index] {
			return nil, 0
		}
	}
	return cache.state.Clone(), len(cache.tokens)
}

// Store replaces the cached snapshot with a clone of the given KV state.
func (cache *PrefixCache) Store(tokens []int, state *model.KVBuffer) {
	if state == nil {
		return
	}
	clonedTokens := append([]int(nil), tokens...)
	cache.mutex.Lock()
	cache.tokens = clonedTokens
	cache.state = state.Clone()
	cache.mutex.Unlock()
}

// Invalidate drops the cached snapshot.
func (cache *PrefixCache) Invalidate() {
	cache.mutex.Lock()
	cache.tokens = nil
	cache.state = nil
	cache.mutex.Unlock()
}
