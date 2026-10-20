package inference

import (
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

func TestBlockPrefixCacheStrictBlockAlignment(t *testing.T) {
	cache := NewBlockPrefixCache(2, 0)
	state := model.NewKVBuffer(1, 512, 1, 1, 4)
	state.Advance(model.KVBlockSize)
	first := make([]int, model.KVBlockSize)
	second := make([]int, model.KVBlockSize)
	for index := range first {
		first[index] = index%13 + 1
		second[index] = index%7 + 2
	}

	cache.Store(first, state)
	if cache.Len() != 1 {
		t.Fatalf("entries = %d, want 1", cache.Len())
	}
	if _, matched := cache.Lookup(first); matched != 0 {
		t.Fatalf("equal prefix matched %d, want 0 (strict prefix required)", matched)
	}
	longer := append(append([]int(nil), first...), 9)
	clone, matched := cache.Lookup(longer)
	if clone == nil || matched != model.KVBlockSize {
		t.Fatalf("matched = %d, want %d", matched, model.KVBlockSize)
	}
	if clone.Position() != model.KVBlockSize {
		t.Fatalf("clone position = %d", clone.Position())
	}
	clone.Release()

	if _, matched := cache.Lookup(first[:model.KVBlockSize-1]); matched != 0 {
		t.Fatalf("partial block matched %d, want 0", matched)
	}

	state2 := model.NewKVBuffer(1, 512, 1, 1, 4)
	state2.Advance(model.KVBlockSize)
	cache.Store(second, state2)
	if cache.Len() != 2 {
		t.Fatalf("entries = %d, want 2", cache.Len())
	}
	cache.Store(append(first, 0), state2) // not block aligned
	if cache.Len() != 2 {
		t.Fatalf("unaligned store changed entries: %d", cache.Len())
	}
	cache.Invalidate()
	if cache.Len() != 0 {
		t.Fatalf("entries after invalidate = %d", cache.Len())
	}
}

func TestEngineBlockPrefixReusesCompleteBlocks(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*model.Config)
	}{
		{"plain", func(config *model.Config) {}},
		{"compressed", func(config *model.Config) { config.CompressionRatio = 2; config.SWAWindow = 4 }},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := model.Config{
				SequenceLen: 256, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
				EmbedDim: 32, WindowPattern: "L",
			}
			testCase.mutate(&config)
			backbone := model.NewTransformer(config)
			backbone.InitWeights(tensors.NewRNG(42))
			tokenizerImpl := testSpeculativeTokenizer()

			prompt1 := make([]int, 200)
			for index := range prompt1 {
				prompt1[index] = index%11 + 1
			}
			prompt2 := make([]int, 160)
			copy(prompt2, prompt1[:model.KVBlockSize])
			for index := model.KVBlockSize; index < len(prompt2); index++ {
				prompt2[index] = (index%9 + 2)
			}

			blockEngine := NewEngine(backbone, tokenizerImpl)
			blockEngine.BlockPrefix = NewBlockPrefixCache(8, 0)
			blockEngine.GenerateBatch(prompt1, 1, 3, 0, 0, 42)
			if blockEngine.BlockPrefix.Len() == 0 {
				t.Fatal("no block-aligned snapshot was stored")
			}
			cached, matched := blockEngine.BlockPrefix.Lookup(prompt2)
			if cached == nil || matched != model.KVBlockSize {
				t.Fatalf("lookup matched = %d, want %d", matched, model.KVBlockSize)
			}
			cached.Release()

			got, _ := blockEngine.GenerateBatch(prompt2, 1, 4, 0, 0, 7)
			referenceEngine := NewEngine(backbone, tokenizerImpl)
			want, _ := referenceEngine.GenerateBatch(prompt2, 1, 4, 0, 0, 7)
			if len(got) != len(want) || len(got) != 1 {
				t.Fatalf("result shape = %d/%d", len(got), len(want))
			}
			for index := range want[0] {
				if got[0][index] != want[0][index] {
					t.Fatalf("token %d = %d, want %d", index, got[0][index], want[0][index])
				}
			}
			hits, _ := blockEngine.BlockPrefix.Stats()
			if hits == 0 {
				t.Fatal("block prefix cache was never hit")
			}
		})
	}
}
