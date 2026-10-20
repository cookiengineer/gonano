package model

import (
	"math"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

func TestKVBlockManagerRefCounts(t *testing.T) {
	manager := NewKVBlockManager(2)
	first := manager.Allocate()
	second := manager.Allocate()
	if first < 0 || second < 0 {
		t.Fatal("expected two blocks")
	}
	if manager.Allocate() != -1 {
		t.Fatal("allocate must fail when the pool is exhausted")
	}
	manager.Retain(first)
	if manager.RefCount(first) != 2 {
		t.Fatalf("refcount = %d, want 2", manager.RefCount(first))
	}
	manager.Free(first)
	if manager.RefCount(first) != 1 {
		t.Fatalf("refcount = %d, want 1", manager.RefCount(first))
	}
	manager.Free(first)
	manager.Free(second)
	if manager.FreeBlocks() != 2 || manager.InUse() != 0 {
		t.Fatalf("pool not drained: free=%d inUse=%d", manager.FreeBlocks(), manager.InUse())
	}
	manager.Allocate()
	manager.Reset()
	if manager.FreeBlocks() != 2 || manager.InUse() != 0 {
		t.Fatalf("reset did not restore the pool: free=%d inUse=%d", manager.FreeBlocks(), manager.InUse())
	}
}

func TestPagedKVCloneSharesUntilWritten(t *testing.T) {
	manager := NewKVBlockManager(4)
	cache := NewPagedKVBuffer(manager, 1, 16, 1, 1, 4)
	key := []float32{1, 2, 3, 4}
	value := []float32{5, 6, 7, 8}
	cache.storage.appendRaw(0, 0, 0, 0, key, value)

	clone := cache.Clone()
	original := cache.storage.(*pagedKVStorage)
	copied := clone.storage.(*pagedKVStorage)
	if len(original.tables[0]) != 1 || len(copied.tables[0]) != 1 {
		t.Fatal("expected one shared block")
	}
	shared := original.tables[0][0]
	if copied.tables[0][0] != shared || manager.RefCount(shared) != 2 {
		t.Fatalf("clone did not share the block: refcount=%d", manager.RefCount(shared))
	}

	cloneKey := []float32{9, 9, 9, 9}
	cloneValue := []float32{10, 10, 10, 10}
	clone.storage.appendRaw(0, 0, 0, 0, cloneKey, cloneValue)
	if copied.tables[0][0] == shared {
		t.Fatal("writing a shared block must copy it first")
	}
	gotKey, _ := cache.storage.readRaw(0, 0, 0, 1)
	if gotKey[0] != 1 {
		t.Fatalf("original block was mutated: %v", gotKey)
	}
	gotCloneKey, _ := clone.storage.readRaw(0, 0, 0, 1)
	if gotCloneKey[0] != 9 {
		t.Fatalf("clone block was not updated: %v", gotCloneKey)
	}
	if manager.RefCount(shared) != 1 {
		t.Fatalf("shared block refcount = %d, want 1", manager.RefCount(shared))
	}

	clone.Release()
	cache.Release()
	if manager.InUse() != 0 {
		t.Fatalf("blocks leaked: inUse=%d", manager.InUse())
	}
}

func enableParityCache(cache *KVBuffer, config Config, capacity int) {
	if config.MLAEnabled() {
		cache.EnableMLA(config.MLALatent, config.NumKVHead*config.MLARotaryDimension())
	}
	if config.AnyCompression() {
		compressionMask := make([]bool, config.NumLayer)
		indexerMask := make([]bool, config.NumLayer)
		for layer := 0; layer < config.NumLayer; layer++ {
			compressionMask[layer] = config.OwnsCompressed(layer)
			indexerMask[layer] = config.OwnsIndexer(layer)
		}
		cache.EnableCompressionLayout(config.LayerRatios(), config.EmbedDim,
			config.NumKVHead*config.HeadDim(), config.CompressedMaxBlocks(capacity), compressionMask)
		if config.AnySparse() {
			dim := config.IndexerDim
			if dim <= 0 {
				dim = 64
			}
			heads := config.IndexerHeads
			if heads <= 0 {
				heads = 1
			}
			cache.EnableIndexerKeysLayers(dim*heads, indexerMask)
		}
	}
}

func TestPagedMatchesDense(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"plain", func(config *Config) {}},
		{"compressed", func(config *Config) { config.CompressionRatio = 2; config.SWAWindow = 4 }},
		{"sparse", func(config *Config) {
			config.CompressionRatio = 2
			config.SWAWindow = 4
			config.SparseTopK = 2
			config.IndexerDim = 4
			config.IndexerHeads = 1
			config.IndexerPool = 2
		}},
		{"mla", func(config *Config) { config.MLALatent = 8; config.MLARotaryDims = 8 }},
	}
	prompt := []int32{1, 5, 2, 8, 3, 7, 4}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := testConfig()
			testCase.mutate(&config)
			transformer := NewTransformer(config)
			transformer.InitWeights(tensors.NewRNG(42))

			paged := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			dense := NewDenseKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(paged, config, config.SequenceLen)
			enableParityCache(dense, config, config.SequenceLen)

			ids := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)
			pagedLogits := transformer.Forward(ids, paged)
			denseLogits := transformer.Forward(ids, dense)
			assertSameLogits(t, denseLogits.Data, pagedLogits.Data)

			next := tensors.NewInt32sWithData([]int{1, 1}, []int32{6})
			pagedNext := transformer.Forward(next, paged)
			denseNext := transformer.Forward(next, dense)
			assertSameLogits(t, denseNext.Data, pagedNext.Data)
		})
	}
}

func assertSameLogits(t *testing.T, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("logit length = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if math.Abs(float64(want[index]-got[index])) > 1e-4 {
			t.Fatalf("logit %d = %v, want %v", index, got[index], want[index])
		}
	}
}

func TestPagedRingsMatchDense(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
		prompt []int32
	}{
		{"compressed", func(config *Config) {
			config.SequenceLen = 64
			config.CompressionRatio = 2
			config.SWAWindow = 4
		}, []int32{1, 5, 2, 8, 3, 7, 4, 6, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6}},
		{"plain-swa", func(config *Config) {
			config.SequenceLen = 512
			config.WindowPattern = "SS"
		}, func() []int32 {
			prompt := make([]int32, 300)
			for index := range prompt {
				prompt[index] = int32(index%13 + 1)
			}
			return prompt
		}()},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := testConfig()
			testCase.mutate(&config)
			transformer := NewTransformer(config)
			transformer.InitWeights(tensors.NewRNG(42))

			paged := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(paged, config, config.SequenceLen)
			paged.EnableSWARings(config.RingWindows())
			dense := NewDenseKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(dense, config, config.SequenceLen)

			ids := tensors.NewInt32sWithData([]int{1, len(testCase.prompt)}, testCase.prompt)
			pagedLogits := transformer.Forward(ids, paged)
			denseLogits := transformer.Forward(ids, dense)
			assertSameLogits(t, denseLogits.Data, pagedLogits.Data)

			next := tensors.NewInt32sWithData([]int{1, 1}, []int32{7})
			pagedNext := transformer.Forward(next, paged)
			denseNext := transformer.Forward(next, dense)
			assertSameLogits(t, denseNext.Data, pagedNext.Data)
		})
	}
}

func TestSWARingsBoundRawMemory(t *testing.T) {
	config := testConfig()
	config.SequenceLen = 512
	config.CompressionRatio = 2
	config.SWAWindow = 4
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	prompt := make([]int32, 256)
	for index := range prompt {
		prompt[index] = int32(index%13 + 1)
	}
	ids := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)

	full := NewDenseKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableParityCache(full, config, config.SequenceLen)
	transformer.Forward(ids, full)

	ring := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableParityCache(ring, config, config.SequenceLen)
	ring.EnableSWARings(config.RingWindows())
	transformer.Forward(ids, ring)

	if ring.BytesAllocated() >= full.BytesAllocated() {
		t.Fatalf("ring cache %d bytes must be smaller than full %d", ring.BytesAllocated(), full.BytesAllocated())
	}
	storage := ring.storage.(*pagedKVStorage)
	for layer := 0; layer < config.NumLayer; layer++ {
		if ring.RingEnabled(layer) && storage.rawKey != nil && storage.rawKey[layer] != nil {
			// Compressed entries share the block table, but ring layers must not
			// store raw token rows in the paged pools.
			for _, head := range storage.rawKey[layer] {
				for _, block := range head {
					if block != nil {
						t.Fatalf("layer %d stored raw rows in paged blocks despite having a ring", layer)
					}
				}
			}
		}
	}
}

func TestBatchedRowsMatchSequential(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"compressed", func(config *Config) { config.CompressionRatio = 2; config.SWAWindow = 4 }},
		{"mla", func(config *Config) { config.MLALatent = 8; config.MLARotaryDims = 8 }},
	}
	promptA := []int32{1, 5, 2, 8, 3}
	promptB := []int32{1, 9, 4, 6, 2, 7, 3, 8}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := testConfig()
			testCase.mutate(&config)
			transformer := NewTransformer(config)
			transformer.InitWeights(tensors.NewRNG(42))
			manager := NewKVBlockManager(256)
			pools := NewKVBlockPools(manager, config.NumLayer, config.NumKVHead, config.HeadDim())

			prefillA := NewPagedKVBufferWithPools(manager, pools, 1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(prefillA, config, config.SequenceLen)
			prefillA.EnableSWARings(config.RingWindows())
			transformer.Forward(tensors.NewInt32sWithData([]int{1, len(promptA)}, promptA), prefillA)
			referenceA := prefillA.Clone()

			prefillB := NewPagedKVBufferWithPools(manager, pools, 1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(prefillB, config, config.SequenceLen)
			prefillB.EnableSWARings(config.RingWindows())
			transformer.Forward(tensors.NewInt32sWithData([]int{1, len(promptB)}, promptB), prefillB)
			referenceB := prefillB.Clone()

			shared := NewPagedKVBufferWithPools(manager, pools, 2, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(shared, config, config.SequenceLen)
			shared.EnableSWARings(config.RingWindows())
			if !shared.AdoptRow(0, prefillA) || !shared.AdoptRow(1, prefillB) {
				t.Fatal("adopt row failed")
			}
			if shared.PositionAt(0) != len(promptA) || shared.PositionAt(1) != len(promptB) {
				t.Fatalf("row positions = %d,%d", shared.PositionAt(0), shared.PositionAt(1))
			}
			prefillA.Release()
			prefillB.Release()

			for step := 0; step < 3; step++ {
				tokenA := int32((step*3 + 1) % 9)
				tokenB := int32((step*5 + 2) % 9)
				batched := transformer.Forward(tensors.NewInt32sWithData([]int{2, 1}, []int32{tokenA, tokenB}), shared)
				seqA := transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{tokenA}), referenceA)
				seqB := transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{tokenB}), referenceB)
				vocab := config.VocabSize
				assertSameLogits(t, seqA.Data, batched.Data[0*vocab:1*vocab])
				assertSameLogits(t, seqB.Data, batched.Data[1*vocab:2*vocab])
			}

			if !shared.MoveRow(0, 1) {
				t.Fatal("move row failed")
			}
			if shared.PositionAt(0) != referenceB.Position() {
				t.Fatalf("moved row position = %d, want %d", shared.PositionAt(0), referenceB.Position())
			}
			token := int32(4)
			moved := transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{token}), shared)
			expected := transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{token}), referenceB)
			assertSameLogits(t, expected.Data, moved.Data)

			shared.Release()
			referenceA.Release()
			referenceB.Release()
			if manager.InUse() != 0 {
				t.Fatalf("blocks leaked: inUse=%d", manager.InUse())
			}
		})
	}
}

func TestPagedDecodeAttentionMatchesDense(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
		rings  bool
	}{
		{"raw-multiblock", func(config *Config) { config.SequenceLen = 512 }, false},
		{"compressed-multiblock", func(config *Config) {
			config.SequenceLen = 512
			config.CompressionRatio = 2
			config.SWAWindow = 4
		}, true},
		{"mla-multiblock", func(config *Config) {
			config.SequenceLen = 512
			config.MLALatent = 8
			config.MLARotaryDims = 8
		}, false},
	}
	prompt := make([]int32, 300)
	for index := range prompt {
		prompt[index] = int32(index%13 + 1)
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := testConfig()
			testCase.mutate(&config)
			transformer := NewTransformer(config)
			transformer.InitWeights(tensors.NewRNG(42))

			paged := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(paged, config, config.SequenceLen)
			if testCase.rings {
				paged.EnableSWARings(config.RingWindows())
			}
			dense := NewDenseKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(dense, config, config.SequenceLen)

			ids := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)
			transformer.Forward(ids, paged)
			transformer.Forward(ids, dense)

			for step := 0; step < 3; step++ {
				token := int32((step*4 + 3) % 9)
				next := tensors.NewInt32sWithData([]int{1, 1}, []int32{token})
				got := transformer.Forward(next, paged)
				want := transformer.Forward(next, dense)
				assertSameLogits(t, want.Data, got.Data)
			}
		})
	}
}

func TestBatchedPrefillMatchesSequential(t *testing.T) {
	config := testConfig()
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	manager := NewKVBlockManager(64)
	pools := NewKVBlockPools(manager, config.NumLayer, config.NumKVHead, config.HeadDim())

	shared := NewPagedKVBufferWithPools(manager, pools, 2, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableParityCache(shared, config, config.SequenceLen)
	referenceA := NewPagedKVBufferWithPools(manager, pools, 1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableParityCache(referenceA, config, config.SequenceLen)
	referenceB := NewPagedKVBufferWithPools(manager, pools, 1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableParityCache(referenceB, config, config.SequenceLen)

	promptA := []int32{1, 5, 2, 8, 3, 7}
	promptB := []int32{1, 2}
	// Batch A and B for the first two tokens.
	transformer.Forward(tensors.NewInt32sWithData([]int{2, 2}, []int32{promptA[0], promptA[1], promptB[0], promptB[1]}), shared)
	// Finish A in two-token chunks.
	transformer.Forward(tensors.NewInt32sWithData([]int{1, 2}, promptA[2:4]), shared)
	logits := transformer.Forward(tensors.NewInt32sWithData([]int{1, 2}, promptA[4:6]), shared)

	referenceLogits := transformer.Forward(tensors.NewInt32sWithData([]int{1, len(promptA)}, promptA), referenceA)
	vocab := config.VocabSize
	assertSameLogits(t, referenceLogits.Data[(len(promptA)-1)*vocab:], logits.Data[(len(promptA[4:])-1)*vocab:])

	referenceBLogits := transformer.Forward(tensors.NewInt32sWithData([]int{1, len(promptB)}, promptB), referenceB)
	_ = referenceBLogits
}

func TestPagedParallelAttentionMatchesDense(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Config)
	}{
		{"raw", func(config *Config) { config.SequenceLen = 1024 }},
		{"compressed", func(config *Config) {
			config.SequenceLen = 1024
			config.CompressionRatio = 2
			config.SWAWindow = 4
		}},
		{"mla", func(config *Config) {
			config.SequenceLen = 1024
			config.MLALatent = 8
			config.MLARotaryDims = 8
		}},
	}
	prompt := make([]int32, 700)
	for index := range prompt {
		prompt[index] = int32(index%13 + 1)
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			config := testConfig()
			testCase.mutate(&config)
			transformer := NewTransformer(config)
			transformer.InitWeights(tensors.NewRNG(42))
			paged := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(paged, config, config.SequenceLen)
			dense := NewDenseKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
			enableParityCache(dense, config, config.SequenceLen)
			ids := tensors.NewInt32sWithData([]int{1, len(prompt)}, prompt)
			transformer.Forward(ids, paged)
			transformer.Forward(ids, dense)
			next := tensors.NewInt32sWithData([]int{1, 1}, []int32{4})
			got := transformer.Forward(next, paged)
			want := transformer.Forward(next, dense)
			assertSameLogits(t, want.Data, got.Data)
		})
	}
}

func TestReleasedRowReuseMatchesFreshCache(t *testing.T) {
	config := testConfig()
	config.SequenceLen = 32
	config.CompressionRatio = 2
	config.SWAWindow = 4
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	enable := func(cache *KVBuffer) {
		enableParityCache(cache, config, config.SequenceLen)
		cache.EnableSWARings(config.RingWindows())
	}
	old := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enable(old)
	transformer.Forward(tensors.NewInt32sWithData([]int{1, 6}, []int32{1, 5, 2, 8, 3, 7}), old)
	for _, token := range []int32{15, 4, 3, 3} {
		transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{token}), old)
	}
	old.ReleaseRow(0)
	old.SetPositionAt(0, 0)

	fresh := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enable(fresh)

	prompt := []int32{1, 2}
	oldLogits := transformer.Forward(tensors.NewInt32sWithData([]int{1, 2}, prompt), old)
	freshLogits := transformer.Forward(tensors.NewInt32sWithData([]int{1, 2}, prompt), fresh)
	assertSameLogits(t, freshLogits.Data, oldLogits.Data)
	for _, token := range []int32{18, 27, 4, 21} {
		oldLogits = transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{token}), old)
		freshLogits = transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{token}), fresh)
		assertSameLogits(t, freshLogits.Data, oldLogits.Data)
	}
}

func TestMixedPrefillDecodeCompressedMatchesSingleRow(t *testing.T) {
	config := testConfig()
	config.SequenceLen = 32
	config.CompressionRatio = 2
	config.SWAWindow = 4
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	manager := NewKVBlockManager(128)
	pools := NewKVBlockPools(manager, config.NumLayer, config.NumKVHead, config.HeadDim())
	shared := NewPagedKVBufferWithPools(manager, pools, 2, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableParityCache(shared, config, config.SequenceLen)
	shared.EnableSWARings(config.RingWindows())
	reference := NewPagedKVBufferWithPools(manager, pools, 1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enableParityCache(reference, config, config.SequenceLen)
	reference.EnableSWARings(config.RingWindows())

	longPrompt := []int32{1, 5, 2, 8, 3, 7}
	shortPrompt := []int32{1, 2}
	// Shared chunked prefill for the first two tokens of each row.
	sharedLogits := transformer.Forward(tensors.NewInt32sWithData([]int{2, 2}, []int32{longPrompt[0], longPrompt[1], shortPrompt[0], shortPrompt[1]}), shared)
	referenceLogits := transformer.Forward(tensors.NewInt32sWithData([]int{1, 2}, shortPrompt), reference)
	vocab := config.VocabSize
	assertSameLogits(t, referenceLogits.Data[vocab:], sharedLogits.Data[3*vocab:])

	shortTokens := []int32{18, 27, 4, 21}
	for step := 0; step < len(shortTokens); step++ {
		sharedLogits = transformer.Forward(tensors.NewInt32sWithData([]int{2, 1}, []int32{longPrompt[2+step], shortTokens[step]}), shared)
		referenceLogits = transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{shortTokens[step]}), reference)
		assertSameLogits(t, referenceLogits.Data, sharedLogits.Data[vocab:])
	}
}

func TestPrefillFromMatchesDirectCompressedDecode(t *testing.T) {
	config := testConfig()
	config.SequenceLen = 32
	config.CompressionRatio = 2
	config.SWAWindow = 4
	transformer := NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	enable := func(cache *KVBuffer) {
		enableParityCache(cache, config, config.SequenceLen)
		cache.EnableSWARings(config.RingWindows())
	}
	prompt := []int32{1, 2}
	direct := NewKVBuffer(1, len(prompt), config.NumLayer, config.NumKVHead, config.HeadDim())
	enable(direct)
	transformer.Forward(tensors.NewInt32sWithData([]int{1, 2}, prompt), direct)

	copied := NewKVBuffer(1, config.SequenceLen, config.NumLayer, config.NumKVHead, config.HeadDim())
	enable(copied)
	PrefillFrom(copied, direct)

	for _, token := range []int32{18, 27, 4, 21} {
		directLogits := transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{token}), direct)
		copiedLogits := transformer.Forward(tensors.NewInt32sWithData([]int{1, 1}, []int32{token}), copied)
		assertSameLogits(t, directLogits.Data, copiedLogits.Data)
	}
}
