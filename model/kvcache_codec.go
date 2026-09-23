package model

import (
	"encoding/binary"
	"errors"
	"io"
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// kvCacheMagic identifies a serialized KV cache buffer.
const kvCacheMagic = "GONAKV\x00\x01"

// KV cache serialization flags.
const (
	kvFlagCompression       = 1 << 0
	kvFlagIndexer           = 1 << 1
	kvFlagPreviousEmbedding = 1 << 2
	kvFlagRawStripped       = 1 << 3
)

// Per-layer serialization flags.
const (
	kvLayerTail       = 1 << 0
	kvLayerCompressed = 1 << 1
	kvLayerIndexer    = 1 << 2
)

// ErrInvalidKVCache is returned when a serialized KV cache cannot be decoded.
var ErrInvalidKVCache = errors.New("model: invalid KV cache data")

// MarshalBinary serializes the KV cache into a self-describing, versioned byte
// slice. Only the used prefix of each key/value buffer is written, so a cache
// persisted for prefix reuse costs its live size rather than its capacity.
//
// The format is used by the inference persistent cache tier to snapshot a
// prefill to disk.
func (cache *KVBuffer) MarshalBinary() ([]byte, error) {
	writer := &kvWriter{}
	writer.buffer = append(writer.buffer, kvCacheMagic...)
	writer.u32(uint32(cache.batchSize))
	writer.u32(uint32(cache.maximumSequenceLength))
	writer.u32(uint32(cache.layerCount))
	writer.u32(uint32(cache.keyValueHeadCount))
	writer.u32(uint32(cache.headDimension))
	writer.i32(cache.sequenceLength)
	writer.u32(uint32(cache.compressionRatio))
	writer.u32(uint32(cache.compressedMaxBlocks))
	writer.u32(uint32(cache.compressedEmbeddingDim))
	writer.u32(uint32(cache.compressedKVWidth))
	writer.u32(uint32(cache.indexerKeyWidth))

	flags := uint32(0)
	compression := cache.compressionRatio > 1
	indexer := cache.indexerKey != nil
	if compression {
		flags |= kvFlagCompression
	}
	if indexer {
		flags |= kvFlagIndexer
	}
	if cache.previousEmbedding != nil {
		flags |= kvFlagPreviousEmbedding
	}
	if cache.rawStripped {
		flags |= kvFlagRawStripped
	}
	writer.u32(flags)

	headDimension := cache.headDimension
	// A stripped snapshot omits the raw key/value buffers; bounded replay
	// rebuilds the local sliding-window rows on load.
	if !cache.rawStripped {
		position := int(cache.sequenceLength)
		for layer := 0; layer < cache.layerCount; layer++ {
			for index := range cache.keyCache[layer] {
				writer.f32s(cache.keyCache[layer][index].Data[:position*headDimension])
				writer.f32s(cache.valueCache[layer][index].Data[:position*headDimension])
			}
		}
	}

	if compression {
		for layer := 0; layer < cache.layerCount; layer++ {
			layerFlags := uint32(0)
			if cache.tailHidden[layer] != nil {
				layerFlags |= kvLayerTail
			}
			if cache.compressedKey[layer] != nil {
				layerFlags |= kvLayerCompressed
			}
			if indexer && cache.indexerKey[layer] != nil {
				layerFlags |= kvLayerIndexer
			}
			writer.u32(layerFlags)
			for batch := 0; batch < cache.batchSize; batch++ {
				writer.i32(int32(cache.tailLength[layer][batch]))
				writer.i32(int32(cache.compressedCount[layer][batch]))
			}
			if layerFlags&kvLayerTail != 0 {
				embeddingDimension := cache.compressedEmbeddingDim
				for batch := 0; batch < cache.batchSize; batch++ {
					used := cache.tailLength[layer][batch]
					writer.f32s(cache.tailHidden[layer][batch][:used*embeddingDimension])
				}
				for index := 0; index < cache.batchSize*cache.keyValueHeadCount; index++ {
					batch := index / cache.keyValueHeadCount
					used := cache.tailLength[layer][batch]
					writer.f32s(cache.tailKey[layer][index][:used*headDimension])
					writer.f32s(cache.tailValue[layer][index][:used*headDimension])
				}
			}
			if layerFlags&kvLayerCompressed != 0 {
				for index := 0; index < cache.batchSize*cache.keyValueHeadCount; index++ {
					batch := index / cache.keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					writer.f32s(cache.compressedKey[layer][index].Data[:count*headDimension])
					writer.f32s(cache.compressedValue[layer][index].Data[:count*headDimension])
				}
			}
			if layerFlags&kvLayerIndexer != 0 {
				width := cache.indexerKeyWidth
				for index := 0; index < cache.batchSize*cache.keyValueHeadCount; index++ {
					batch := index / cache.keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					writer.f32s(cache.indexerKey[layer][index].Data[:count*width])
				}
			}
		}
	}

	if flags&kvFlagPreviousEmbedding != 0 {
		channels := cache.previousEmbedding.Shape[2]
		writer.u32(uint32(channels))
		writer.f32s(cache.previousEmbedding.Data)
	}
	return writer.buffer, nil
}

// UnmarshalKVBuffer decodes a buffer written by KVBuffer.MarshalBinary.
func UnmarshalKVBuffer(data []byte) (*KVBuffer, error) {
	reader := &kvReader{data: data}
	if len(data) < len(kvCacheMagic) || string(data[:len(kvCacheMagic)]) != kvCacheMagic {
		return nil, ErrInvalidKVCache
	}
	reader.pos = len(kvCacheMagic)
	batchSize := int(reader.u32())
	maximumSequenceLength := int(reader.u32())
	layerCount := int(reader.u32())
	keyValueHeadCount := int(reader.u32())
	headDimension := int(reader.u32())
	sequenceLength := reader.i32()
	compressionRatio := int(reader.u32())
	compressedMaxBlocks := int(reader.u32())
	compressedEmbeddingDim := int(reader.u32())
	compressedKVWidth := int(reader.u32())
	indexerKeyWidth := int(reader.u32())
	flags := reader.u32()
	if reader.err != nil {
		return nil, reader.err
	}
	if batchSize <= 0 || layerCount <= 0 || keyValueHeadCount <= 0 || headDimension <= 0 || maximumSequenceLength <= 0 {
		return nil, ErrInvalidKVCache
	}
	if sequenceLength < 0 || int(sequenceLength) > maximumSequenceLength {
		return nil, ErrInvalidKVCache
	}

	cache := NewKVBuffer(batchSize, maximumSequenceLength, layerCount, keyValueHeadCount, headDimension)
	cache.sequenceLength = sequenceLength
	cache.rawStripped = flags&kvFlagRawStripped != 0

	if !cache.rawStripped {
		position := int(sequenceLength)
		for layer := 0; layer < layerCount; layer++ {
			for index := range cache.keyCache[layer] {
				reader.f32s(cache.keyCache[layer][index].Data[:position*headDimension])
				reader.f32s(cache.valueCache[layer][index].Data[:position*headDimension])
			}
		}
	}

	compression := flags&kvFlagCompression != 0
	if compression {
		cache.compressionRatio = compressionRatio
		cache.compressedMaxBlocks = compressedMaxBlocks
		cache.compressedEmbeddingDim = compressedEmbeddingDim
		cache.compressedKVWidth = compressedKVWidth
		if flags&kvFlagIndexer != 0 {
			cache.indexerKeyWidth = indexerKeyWidth
		}
		cache.tailHidden = make([][][]float32, layerCount)
		cache.tailKey = make([][][]float32, layerCount)
		cache.tailValue = make([][][]float32, layerCount)
		cache.tailLength = make([][]int, layerCount)
		cache.compressedKey = make([][]*tensors.Tensor, layerCount)
		cache.compressedValue = make([][]*tensors.Tensor, layerCount)
		cache.compressedCount = make([][]int, layerCount)
		if flags&kvFlagIndexer != 0 {
			cache.indexerKey = make([][]*tensors.Tensor, layerCount)
		}
		for layer := 0; layer < layerCount; layer++ {
			layerFlags := reader.u32()
			cache.tailLength[layer] = make([]int, batchSize)
			cache.compressedCount[layer] = make([]int, batchSize)
			for batch := 0; batch < batchSize; batch++ {
				cache.tailLength[layer][batch] = int(reader.i32())
				cache.compressedCount[layer][batch] = int(reader.i32())
			}
			if layerFlags&kvLayerTail != 0 {
				cache.tailHidden[layer] = make([][]float32, batchSize)
				cache.tailKey[layer] = make([][]float32, batchSize*keyValueHeadCount)
				cache.tailValue[layer] = make([][]float32, batchSize*keyValueHeadCount)
				for batch := 0; batch < batchSize; batch++ {
					cache.tailHidden[layer][batch] = make([]float32, compressionRatio*compressedEmbeddingDim)
				}
				for index := range cache.tailKey[layer] {
					cache.tailKey[layer][index] = make([]float32, compressionRatio*headDimension)
					cache.tailValue[layer][index] = make([]float32, compressionRatio*headDimension)
				}
				for batch := 0; batch < batchSize; batch++ {
					used := cache.tailLength[layer][batch]
					reader.f32s(cache.tailHidden[layer][batch][:used*compressedEmbeddingDim])
				}
				for index := 0; index < batchSize*keyValueHeadCount; index++ {
					batch := index / keyValueHeadCount
					used := cache.tailLength[layer][batch]
					reader.f32s(cache.tailKey[layer][index][:used*headDimension])
					reader.f32s(cache.tailValue[layer][index][:used*headDimension])
				}
			}
			if layerFlags&kvLayerCompressed != 0 {
				cache.compressedKey[layer] = make([]*tensors.Tensor, batchSize*keyValueHeadCount)
				cache.compressedValue[layer] = make([]*tensors.Tensor, batchSize*keyValueHeadCount)
				for index := range cache.compressedKey[layer] {
					cache.compressedKey[layer][index] = tensors.New(compressedMaxBlocks, headDimension)
					cache.compressedValue[layer][index] = tensors.New(compressedMaxBlocks, headDimension)
				}
				for index := 0; index < batchSize*keyValueHeadCount; index++ {
					batch := index / keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					reader.f32s(cache.compressedKey[layer][index].Data[:count*headDimension])
					reader.f32s(cache.compressedValue[layer][index].Data[:count*headDimension])
				}
			}
			if layerFlags&kvLayerIndexer != 0 {
				cache.indexerKey[layer] = make([]*tensors.Tensor, batchSize*keyValueHeadCount)
				for index := range cache.indexerKey[layer] {
					cache.indexerKey[layer][index] = tensors.New(compressedMaxBlocks, indexerKeyWidth)
				}
				for index := 0; index < batchSize*keyValueHeadCount; index++ {
					batch := index / keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					reader.f32s(cache.indexerKey[layer][index].Data[:count*indexerKeyWidth])
				}
			}
		}
	}

	if flags&kvFlagPreviousEmbedding != 0 {
		channels := int(reader.u32())
		if channels <= 0 {
			return nil, ErrInvalidKVCache
		}
		cache.previousEmbedding = tensors.New(batchSize, 1, channels)
		reader.f32s(cache.previousEmbedding.Data)
	}
	if reader.err != nil {
		return nil, reader.err
	}
	return cache, nil
}

// kvWriter appends little-endian values to a byte buffer.
type kvWriter struct {
	buffer []byte
}

func (writer *kvWriter) u32(value uint32) {
	writer.buffer = binary.LittleEndian.AppendUint32(writer.buffer, value)
}

func (writer *kvWriter) i32(value int32) {
	writer.buffer = binary.LittleEndian.AppendUint32(writer.buffer, uint32(value))
}

func (writer *kvWriter) f32s(values []float32) {
	for _, value := range values {
		writer.buffer = binary.LittleEndian.AppendUint32(writer.buffer, math.Float32bits(value))
	}
}

// kvReader reads little-endian values, recording the first error.
type kvReader struct {
	data []byte
	pos  int
	err  error
}

func (reader *kvReader) need(count int) bool {
	if reader.err != nil {
		return false
	}
	if reader.pos+count > len(reader.data) {
		reader.err = io.ErrUnexpectedEOF
		return false
	}
	return true
}

func (reader *kvReader) u32() uint32 {
	if !reader.need(4) {
		return 0
	}
	value := binary.LittleEndian.Uint32(reader.data[reader.pos:])
	reader.pos += 4
	return value
}

func (reader *kvReader) i32() int32 { return int32(reader.u32()) }

func (reader *kvReader) f32s(destination []float32) {
	if !reader.need(4 * len(destination)) {
		return
	}
	for index := range destination {
		destination[index] = math.Float32frombits(binary.LittleEndian.Uint32(reader.data[reader.pos:]))
		reader.pos += 4
	}
}
