package model

import (
	"errors"
	"io"
	"testing"

	"github.com/cookiengineer/gonano/tensors"
)

// populatedCache builds a compressed, sparse, windowed KV cache with every
// state section filled so the codec round-trip covers all branches.
func populatedCache() *KVBuffer {
	cache := NewKVBuffer(1, 16, 3, 2, 4)
	cache.Advance(3)
	cache.EnableCompression(2, 4, 8, 8)
	cache.EnableIndexerKeys(6)

	fill := tensors.NewRNG(7)
	for layer := 0; layer < cache.layerCount; layer++ {
		for head := 0; head < cache.keyValueHeadCount; head++ {
			key := make([]float32, 3*cache.headDimension)
			value := make([]float32, 3*cache.headDimension)
			for element := range key {
				key[element] = fill.NormFloat32()
				value[element] = fill.NormFloat32()
			}
			cache.storage.appendRaw(layer, 0, head, 0, key, value)
		}
		if cache.tailHidden[layer] == nil {
			continue
		}
		cache.tailLength[layer][0] = 1
		cache.compressedCount[layer][0] = 2
		for element := range cache.tailHidden[layer][0][:cache.compressedEmbeddingDim] {
			cache.tailHidden[layer][0][element] = fill.NormFloat32()
		}
		for index := range cache.tailKey[layer] {
			for element := range cache.tailKey[layer][index][:cache.headDimension] {
				cache.tailKey[layer][index][element] = fill.NormFloat32()
				cache.tailValue[layer][index][element] = fill.NormFloat32()
			}
		}
		for head := 0; head < cache.keyValueHeadCount; head++ {
			for entry := 0; entry < 2; entry++ {
				key := make([]float32, cache.headDimension)
				value := make([]float32, cache.headDimension)
				for element := range key {
					key[element] = fill.NormFloat32()
					value[element] = fill.NormFloat32()
				}
				cache.storage.appendCompressed(layer, 0, head, entry, key, value)
				indexerKey := make([]float32, cache.indexerKeyWidth)
				for element := range indexerKey {
					indexerKey[element] = fill.NormFloat32()
				}
				cache.storage.appendIndexer(layer, 0, head, entry, indexerKey)
			}
		}
	}
	cache.previousEmbedding = tensors.New(1, 1, cache.headDimension)
	for index := range cache.previousEmbedding.Data {
		cache.previousEmbedding.Data[index] = fill.NormFloat32()
	}
	return cache
}

func TestKVCacheCodecRoundTrip(t *testing.T) {
	cache := populatedCache()
	data, err := cache.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := UnmarshalKVBuffer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.batchSize != cache.batchSize || decoded.maximumSequenceLength != cache.maximumSequenceLength ||
		decoded.layerCount != cache.layerCount || decoded.keyValueHeadCount != cache.keyValueHeadCount ||
		decoded.headDimension != cache.headDimension || decoded.sequenceLength != cache.sequenceLength {
		t.Fatal("geometry mismatch after round-trip")
	}
	if decoded.compressionRatio != cache.compressionRatio || decoded.compressedMaxBlocks != cache.compressedMaxBlocks ||
		decoded.compressedEmbeddingDim != cache.compressedEmbeddingDim || decoded.indexerKeyWidth != cache.indexerKeyWidth {
		t.Fatal("compression geometry mismatch after round-trip")
	}

	position := int(cache.sequenceLength)
	for layer := 0; layer < cache.layerCount; layer++ {
		for head := 0; head < cache.keyValueHeadCount; head++ {
			wantKey, wantValue := cache.storage.readRaw(layer, 0, head, position)
			gotKey, gotValue := decoded.storage.readRaw(layer, 0, head, position)
			for element := 0; element < position*cache.headDimension; element++ {
				if gotKey[element] != wantKey[element] {
					t.Fatalf("key cache mismatch at layer %d head %d element %d", layer, head, element)
				}
				if gotValue[element] != wantValue[element] {
					t.Fatalf("value cache mismatch at layer %d head %d element %d", layer, head, element)
				}
			}
			wantCompressedKey, wantCompressedValue := cache.storage.readCompressed(layer, 0, head, 2)
			gotCompressedKey, gotCompressedValue := decoded.storage.readCompressed(layer, 0, head, 2)
			for element := 0; element < 2*cache.headDimension; element++ {
				if gotCompressedKey[element] != wantCompressedKey[element] {
					t.Fatalf("compressed key mismatch at layer %d head %d", layer, head)
				}
				if gotCompressedValue[element] != wantCompressedValue[element] {
					t.Fatalf("compressed value mismatch at layer %d head %d", layer, head)
				}
			}
			wantIndexer := cache.storage.readIndexer(layer, 0, head, 2)
			gotIndexer := decoded.storage.readIndexer(layer, 0, head, 2)
			for element := 0; element < 2*cache.indexerKeyWidth; element++ {
				if gotIndexer[element] != wantIndexer[element] {
					t.Fatalf("indexer key mismatch at layer %d head %d", layer, head)
				}
			}
		}
		if cache.tailHidden[layer] == nil {
			if decoded.tailHidden[layer] != nil {
				t.Fatalf("layer %d should have no tail", layer)
			}
			continue
		}
		for element := 0; element < cache.compressedEmbeddingDim; element++ {
			if decoded.tailHidden[layer][0][element] != cache.tailHidden[layer][0][element] {
				t.Fatalf("tail hidden mismatch at layer %d element %d", layer, element)
			}
		}
		for index := range cache.tailKey[layer] {
			for element := 0; element < cache.headDimension; element++ {
				if decoded.tailKey[layer][index][element] != cache.tailKey[layer][index][element] {
					t.Fatalf("tail key mismatch at layer %d index %d", layer, index)
				}
				if decoded.tailValue[layer][index][element] != cache.tailValue[layer][index][element] {
					t.Fatalf("tail value mismatch at layer %d index %d", layer, index)
				}
			}
		}
	}
	if decoded.previousEmbedding == nil || decoded.previousEmbedding.Numel() != cache.previousEmbedding.Numel() {
		t.Fatal("previous embedding missing after round-trip")
	}
	for index := range cache.previousEmbedding.Data {
		if decoded.previousEmbedding.Data[index] != cache.previousEmbedding.Data[index] {
			t.Fatal("previous embedding mismatch after round-trip")
		}
	}
}

func TestKVCacheCodecPlainRoundTrip(t *testing.T) {
	cache := NewKVBuffer(2, 8, 1, 1, 4)
	cache.Advance(2)
	for row := 0; row < 2; row++ {
		key := make([]float32, 2*4)
		value := make([]float32, 2*4)
		for element := range key {
			key[element] = float32(element + row*100)
			value[element] = -float32(element + row*100)
		}
		cache.storage.appendRaw(0, row, 0, 0, key, value)
	}
	data, err := cache.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := UnmarshalKVBuffer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.CompressionEnabled() {
		t.Fatal("plain cache should not report compression")
	}
	for row := 0; row < 2; row++ {
		wantKey, wantValue := cache.storage.readRaw(0, row, 0, 2)
		gotKey, gotValue := decoded.storage.readRaw(0, row, 0, 2)
		for element := 0; element < 2*4; element++ {
			if gotKey[element] != wantKey[element] || gotValue[element] != wantValue[element] {
				t.Fatal("plain key/value cache mismatch")
			}
		}
	}
	if decoded.previousEmbedding != nil {
		t.Fatal("plain cache should have no previous embedding")
	}
}

func TestKVCacheCodecRejectsInvalidData(t *testing.T) {
	if _, err := UnmarshalKVBuffer([]byte("not a kv cache")); !errors.Is(err, ErrInvalidKVCache) {
		t.Fatalf("invalid magic error = %v, want ErrInvalidKVCache", err)
	}

	cache := populatedCache()
	data, err := cache.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if _, err := UnmarshalKVBuffer(data[:len(data)-4]); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("truncated error = %v, want io.ErrUnexpectedEOF", err)
	}
}

func TestKVCacheCodecStrippedRoundTrip(t *testing.T) {
	cache := populatedCache()
	cache.StripRaw()
	if !cache.RawStripped() {
		t.Fatal("StripRaw must mark the cache as stripped")
	}
	data, err := cache.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := UnmarshalKVBuffer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !decoded.RawStripped() {
		t.Fatal("decoded cache lost the stripped flag")
	}
	if decoded.sequenceLength != cache.sequenceLength {
		t.Fatalf("position = %d, want %d", decoded.sequenceLength, cache.sequenceLength)
	}
	// Raw buffers are gone.
	for layer := 0; layer < decoded.layerCount; layer++ {
		for head := 0; head < decoded.keyValueHeadCount; head++ {
			key, value := decoded.storage.readRaw(layer, 0, head, int(decoded.sequenceLength))
			for _, entry := range key {
				if entry != 0 {
					t.Fatal("stripped key cache is not zeroed")
				}
			}
			for _, entry := range value {
				if entry != 0 {
					t.Fatal("stripped value cache is not zeroed")
				}
			}
		}
	}
	// Global compressed, indexer and tail state survive.
	for layer := 0; layer < cache.layerCount; layer++ {
		if !cache.compressedAlloc[layer] {
			continue
		}
		for head := 0; head < cache.keyValueHeadCount; head++ {
			wantKey, wantValue := cache.storage.readCompressed(layer, 0, head, 2)
			gotKey, gotValue := decoded.storage.readCompressed(layer, 0, head, 2)
			for element := 0; element < 2*cache.headDimension; element++ {
				if gotKey[element] != wantKey[element] {
					t.Fatal("compressed key lost after strip round-trip")
				}
				if gotValue[element] != wantValue[element] {
					t.Fatal("compressed value lost after strip round-trip")
				}
			}
			wantIndexer := cache.storage.readIndexer(layer, 0, head, 2)
			gotIndexer := decoded.storage.readIndexer(layer, 0, head, 2)
			for element := 0; element < 2*cache.indexerKeyWidth; element++ {
				if gotIndexer[element] != wantIndexer[element] {
					t.Fatal("indexer key lost after strip round-trip")
				}
			}
		}
	}
}

func TestKVCacheCodecRingRoundTrip(t *testing.T) {
	cache := NewKVBuffer(1, 256, 2, 1, 4)
	cache.EnableCompression(2, 4, 4, 16)
	cache.EnableSWARings([]int{4, 0})
	cache.Advance(20)
	for position := 0; position < 20; position++ {
		key := []float32{float32(position), float32(position + 1), float32(position + 2), float32(position + 3)}
		value := []float32{float32(-position), float32(-position - 1), float32(-position - 2), float32(-position - 3)}
		cache.appendRawKV(0, 0, 0, position, key, value)
		cache.appendRawKV(1, 0, 0, position, key, value)
	}
	data, err := cache.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoded, err := UnmarshalKVBuffer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !decoded.RingEnabled(0) || decoded.RingEnabled(1) {
		t.Fatal("ring configuration lost after round-trip")
	}
	if decoded.ringSize != cache.ringSize {
		t.Fatalf("ring size = %d, want %d", decoded.ringSize, cache.ringSize)
	}
	for element := range cache.ringKey[0][0][0] {
		if decoded.ringKey[0][0][0][element] != cache.ringKey[0][0][0][element] {
			t.Fatal("ring key mismatch after round-trip")
		}
		if decoded.ringValue[0][0][0][element] != cache.ringValue[0][0][0][element] {
			t.Fatal("ring value mismatch after round-trip")
		}
	}
	wantKey, wantValue := cache.storage.readRaw(1, 0, 0, 20)
	gotKey, gotValue := decoded.storage.readRaw(1, 0, 0, 20)
	for element := range wantKey {
		if gotKey[element] != wantKey[element] || gotValue[element] != wantValue[element] {
			t.Fatal("full-context raw cache mismatch after round-trip")
		}
	}
}
