package inference

import (
	"sort"
	"strconv"
	"sync"

	"github.com/cookiengineer/gonano/tokenizer"
)

const constraintMaskCacheEntries = 2048

type constraintStateKeyer interface {
	StateKey() string
}

type constraintMaskKey struct {
	tokenizer *tokenizer.Tokenizer
	state     string
	finished  bool
	stops     string
	vocab     int
}

type constraintMaskStore struct {
	mutex   sync.Mutex
	entries map[constraintMaskKey][]bool
	order   []constraintMaskKey
	hits    int64
	misses  int64
}

var constraintMaskCache = constraintMaskStore{entries: make(map[constraintMaskKey][]bool)}

// ConstraintMaskCached builds the allowed-token mask, reusing the result for an
// identical constraint state, tokenizer and stop-token set. Constraints that do
// not expose a StateKey fall back to a full scan.
func ConstraintMaskCached(tokenizerImpl *tokenizer.Tokenizer, constraint Constraint, vocab int, stopTokens []int) []bool {
	keyer, ok := constraint.(constraintStateKeyer)
	if constraint == nil || !ok {
		return ConstraintMask(tokenizerImpl, constraint, vocab, stopTokens)
	}
	key := constraintMaskKey{
		tokenizer: tokenizerImpl,
		state:     keyer.StateKey(),
		finished:  constraint.Finished(),
		stops:     stopTokenKey(stopTokens),
		vocab:     vocab,
	}
	constraintMaskCache.mutex.Lock()
	if mask, exists := constraintMaskCache.entries[key]; exists {
		constraintMaskCache.hits++
		constraintMaskCache.touchLocked(key)
		constraintMaskCache.mutex.Unlock()
		return mask
	}
	constraintMaskCache.mutex.Unlock()

	mask := ConstraintMask(tokenizerImpl, constraint, vocab, stopTokens)

	constraintMaskCache.mutex.Lock()
	defer constraintMaskCache.mutex.Unlock()
	if existing, exists := constraintMaskCache.entries[key]; exists {
		constraintMaskCache.hits++
		return existing
	}
	constraintMaskCache.misses++
	constraintMaskCache.entries[key] = mask
	constraintMaskCache.order = append(constraintMaskCache.order, key)
	for len(constraintMaskCache.entries) > constraintMaskCacheEntries && len(constraintMaskCache.order) > 0 {
		oldest := constraintMaskCache.order[0]
		constraintMaskCache.order = constraintMaskCache.order[1:]
		delete(constraintMaskCache.entries, oldest)
	}
	return mask
}

// ConstraintMaskCacheStats reports cache hits, misses and current entries.
func ConstraintMaskCacheStats() (hits, misses, entries int64) {
	constraintMaskCache.mutex.Lock()
	defer constraintMaskCache.mutex.Unlock()
	return constraintMaskCache.hits, constraintMaskCache.misses, int64(len(constraintMaskCache.entries))
}

func (cache *constraintMaskStore) touchLocked(key constraintMaskKey) {
	for index, candidate := range cache.order {
		if candidate == key {
			cache.order = append(cache.order[:index], cache.order[index+1:]...)
			break
		}
	}
	cache.order = append(cache.order, key)
}

func stopTokenKey(stops []int) string {
	values := append([]int(nil), stops...)
	sort.Ints(values)
	builder := make([]byte, 0, len(values)*4)
	for _, value := range values {
		builder = strconv.AppendInt(builder, int64(value), 10)
		builder = append(builder, ',')
	}
	return string(builder)
}
