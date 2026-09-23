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
		for index := range cache.keyCache[layer] {
			for element := range cache.keyCache[layer][index].Data {
				cache.keyCache[layer][index].Data[element] = fill.NormFloat32()
				cache.valueCache[layer][index].Data[element] = fill.NormFloat32()
			}
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
			for element := range cache.compressedKey[layer][index].Data[:2*cache.headDimension] {
				cache.compressedKey[layer][index].Data[element] = fill.NormFloat32()
				cache.compressedValue[layer][index].Data[element] = fill.NormFloat32()
			}
			for element := range cache.indexerKey[layer][index].Data[:2*cache.indexerKeyWidth] {
				cache.indexerKey[layer][index].Data[element] = fill.NormFloat32()
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
		for index := range cache.keyCache[layer] {
			for element := 0; element < position*cache.headDimension; element++ {
				if decoded.keyCache[layer][index].Data[element] != cache.keyCache[layer][index].Data[element] {
					t.Fatalf("key cache mismatch at layer %d index %d element %d", layer, index, element)
				}
				if decoded.valueCache[layer][index].Data[element] != cache.valueCache[layer][index].Data[element] {
					t.Fatalf("value cache mismatch at layer %d index %d element %d", layer, index, element)
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
			for element := 0; element < 2*cache.headDimension; element++ {
				if decoded.compressedKey[layer][index].Data[element] != cache.compressedKey[layer][index].Data[element] {
					t.Fatalf("compressed key mismatch at layer %d index %d", layer, index)
				}
				if decoded.compressedValue[layer][index].Data[element] != cache.compressedValue[layer][index].Data[element] {
					t.Fatalf("compressed value mismatch at layer %d index %d", layer, index)
				}
			}
			for element := 0; element < 2*cache.indexerKeyWidth; element++ {
				if decoded.indexerKey[layer][index].Data[element] != cache.indexerKey[layer][index].Data[element] {
					t.Fatalf("indexer key mismatch at layer %d index %d", layer, index)
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

	// The decoded cache must report the same allocated size.
	if decoded.BytesAllocated() != cache.BytesAllocated() {
		t.Fatalf("bytes allocated = %d, want %d", decoded.BytesAllocated(), cache.BytesAllocated())
	}
}

func TestKVCacheCodecPlainRoundTrip(t *testing.T) {
	cache := NewKVBuffer(2, 8, 1, 1, 4)
	cache.Advance(2)
	for index := range cache.keyCache[0] {
		for element := range cache.keyCache[0][index].Data {
			cache.keyCache[0][index].Data[element] = float32(element)
			cache.valueCache[0][index].Data[element] = -float32(element)
		}
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
	for index := range cache.keyCache[0] {
		for element := 0; element < 2*4; element++ {
			if decoded.keyCache[0][index].Data[element] != cache.keyCache[0][index].Data[element] {
				t.Fatal("plain key cache mismatch")
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
