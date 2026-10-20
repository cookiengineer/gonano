package model

import (
	"encoding/binary"
	"errors"
	"io"
	"math"

	"github.com/cookiengineer/gonano/tensors"
)

// kvCacheMagic identifies a serialized KV cache buffer.
const kvCacheMagic = "GONAKV\x00\x03"

// KV cache serialization flags.
const (
	kvFlagCompression       = 1 << 0
	kvFlagIndexer           = 1 << 1
	kvFlagPreviousEmbedding = 1 << 2
	kvFlagRawStripped       = 1 << 3
	kvFlagMLA               = 1 << 4
	kvFlagRings             = 1 << 5
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
// slice. Only the used prefix of each buffer is written, so a cache persisted
// for prefix reuse costs its live size rather than its capacity. Paged caches
// are serialized in their logical dense form.
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
	writer.u32(uint32(cache.mlaLatentWidth))
	writer.u32(uint32(cache.mlaRopeKeyWidth))
	writer.u32(uint32(len(cache.layerRatios)))
	for layer := 0; layer < len(cache.layerRatios); layer++ {
		writer.u32(uint32(cache.layerRatios[layer]))
		maxBlocks := 0
		if layer < len(cache.maxBlocksByLayer) {
			maxBlocks = cache.maxBlocksByLayer[layer]
		}
		writer.u32(uint32(maxBlocks))
	}
	writer.u32(uint32(cache.ringSize))
	writer.u32(uint32(len(cache.ringWindow)))
	for layer := 0; layer < len(cache.ringWindow); layer++ {
		writer.u32(uint32(cache.ringWindow[layer]))
	}

	flags := uint32(0)
	compression := cache.compressionEnabled
	indexer := cache.indexerEnabled
	mla := cache.mlaEnabled
	rawStripped := cache.RawStripped()
	if mla {
		flags |= kvFlagMLA
	}
	if compression {
		flags |= kvFlagCompression
	}
	if indexer {
		flags |= kvFlagIndexer
	}
	if cache.previousEmbedding != nil {
		flags |= kvFlagPreviousEmbedding
	}
	if rawStripped {
		flags |= kvFlagRawStripped
	}
	if cache.ringSize > 0 && !rawStripped {
		flags |= kvFlagRings
	}
	writer.u32(flags)

	headDimension := cache.headDimension
	position := int(cache.sequenceLength)
	if !rawStripped {
		for layer := 0; layer < cache.layerCount; layer++ {
			for index := 0; index < cache.batchSize*cache.keyValueHeadCount; index++ {
				row := index / cache.keyValueHeadCount
				head := index % cache.keyValueHeadCount
				key, value := cache.storage.readRaw(layer, row, head, position)
				writer.f32s(key)
				writer.f32s(value)
			}
		}
	}

	if flags&kvFlagRings != 0 {
		for layer := 0; layer < cache.layerCount; layer++ {
			if !cache.RingEnabled(layer) {
				continue
			}
			for row := 0; row < cache.batchSize; row++ {
				for head := 0; head < cache.keyValueHeadCount; head++ {
					key := make([]float32, cache.ringSize*cache.headDimension)
					value := make([]float32, cache.ringSize*cache.headDimension)
					if cache.ringKey != nil && cache.ringKey[layer] != nil && cache.ringKey[layer][row] != nil && cache.ringKey[layer][row][head] != nil {
						copy(key, cache.ringKey[layer][row][head])
					}
					if cache.ringValue != nil && cache.ringValue[layer] != nil && cache.ringValue[layer][row] != nil && cache.ringValue[layer][row][head] != nil {
						copy(value, cache.ringValue[layer][row][head])
					}
					writer.f32s(key)
					writer.f32s(value)
				}
			}
		}
	}

	if compression {
		for layer := 0; layer < cache.layerCount; layer++ {
			layerFlags := uint32(0)
			if cache.tailHidden[layer] != nil {
				layerFlags |= kvLayerTail
			}
			if layer < len(cache.compressedAlloc) && cache.compressedAlloc[layer] {
				layerFlags |= kvLayerCompressed
			}
			if indexer && layer < len(cache.indexerAlloc) && cache.indexerAlloc[layer] {
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
					head := index % cache.keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					key, value := cache.storage.readCompressed(layer, batch, head, count)
					writer.f32s(key)
					writer.f32s(value)
				}
			}
			if layerFlags&kvLayerIndexer != 0 {
				width := cache.indexerKeyWidth
				for index := 0; index < cache.batchSize*cache.keyValueHeadCount; index++ {
					batch := index / cache.keyValueHeadCount
					head := index % cache.keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					writer.f32s(cache.storage.readIndexer(layer, batch, head, count)[:count*width])
				}
			}
		}
	}

	if mla {
		for layer := 0; layer < cache.layerCount; layer++ {
			for batch := 0; batch < cache.batchSize; batch++ {
				writer.f32s(cache.storage.readMLALatent(layer, batch, position))
				writer.f32s(cache.storage.readMLARope(layer, batch, position))
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
	mlaLatentWidth := int(reader.u32())
	mlaRopeKeyWidth := int(reader.u32())
	layerRatioCount := int(reader.u32())
	layerRatios := make([]int, layerRatioCount)
	layerMaxBlocks := make([]int, layerRatioCount)
	for layer := 0; layer < layerRatioCount; layer++ {
		layerRatios[layer] = int(reader.u32())
		layerMaxBlocks[layer] = int(reader.u32())
	}
	ringSize := int(reader.u32())
	ringWindowCount := int(reader.u32())
	ringWindows := make([]int, ringWindowCount)
	for layer := 0; layer < ringWindowCount; layer++ {
		ringWindows[layer] = int(reader.u32())
	}
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
	if ringWindowCount > 0 {
		cache.EnableSWARings(ringWindows)
		if ringSize > 0 {
			cache.ringSize = ringSize
		}
	}

	position := int(sequenceLength)
	if !cache.rawStripped {
		for layer := 0; layer < layerCount; layer++ {
			for index := 0; index < batchSize*keyValueHeadCount; index++ {
				row := index / keyValueHeadCount
				head := index % keyValueHeadCount
				key := make([]float32, position*headDimension)
				value := make([]float32, position*headDimension)
				reader.f32s(key)
				reader.f32s(value)
				if !cache.RingEnabled(layer) {
					cache.storage.appendRaw(layer, row, head, 0, key, value)
				}
			}
		}
	}

	if flags&kvFlagRings != 0 {
		cache.ringKey = make([][][][]float32, layerCount)
		cache.ringValue = make([][][][]float32, layerCount)
		for layer := 0; layer < layerCount; layer++ {
			if !cache.RingEnabled(layer) {
				continue
			}
			cache.ringKey[layer] = make([][][]float32, batchSize)
			cache.ringValue[layer] = make([][][]float32, batchSize)
			for row := 0; row < batchSize; row++ {
				cache.ringKey[layer][row] = make([][]float32, keyValueHeadCount)
				cache.ringValue[layer][row] = make([][]float32, keyValueHeadCount)
				for head := 0; head < keyValueHeadCount; head++ {
					key := make([]float32, cache.ringSize*headDimension)
					value := make([]float32, cache.ringSize*headDimension)
					reader.f32s(key)
					reader.f32s(value)
					cache.ringKey[layer][row][head] = key
					cache.ringValue[layer][row][head] = value
				}
			}
		}
	}

	compression := flags&kvFlagCompression != 0
	if compression {
		if len(layerRatios) != layerCount {
			return nil, ErrInvalidKVCache
		}
		cache.EnableCompressionLayout(layerRatios, compressedEmbeddingDim, compressedKVWidth, layerMaxBlocks, nil)
		cache.compressionRatio = compressionRatio
		cache.compressedMaxBlocks = compressedMaxBlocks
		if flags&kvFlagIndexer != 0 {
			cache.EnableIndexerKeysLayers(indexerKeyWidth, nil)
		}
		for layer := 0; layer < layerCount; layer++ {
			layerFlags := reader.u32()
			cache.compressedAlloc[layer] = layerFlags&kvLayerCompressed != 0
			if layer < len(cache.indexerAlloc) {
				cache.indexerAlloc[layer] = layerFlags&kvLayerIndexer != 0
			}
			if layerFlags&kvLayerTail == 0 {
				cache.tailHidden[layer] = nil
				cache.tailKey[layer] = nil
				cache.tailValue[layer] = nil
			}
			for batch := 0; batch < batchSize; batch++ {
				cache.tailLength[layer][batch] = int(reader.i32())
				cache.compressedCount[layer][batch] = int(reader.i32())
			}
			if layerFlags&kvLayerTail != 0 {
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
				for index := 0; index < batchSize*keyValueHeadCount; index++ {
					batch := index / keyValueHeadCount
					head := index % keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					key := make([]float32, count*headDimension)
					value := make([]float32, count*headDimension)
					reader.f32s(key)
					reader.f32s(value)
					for entry := 0; entry < count; entry++ {
						cache.storage.appendCompressed(layer, batch, head, entry,
							key[entry*headDimension:(entry+1)*headDimension],
							value[entry*headDimension:(entry+1)*headDimension])
					}
				}
			}
			if layerFlags&kvLayerIndexer != 0 {
				width := cache.indexerKeyWidth
				for index := 0; index < batchSize*keyValueHeadCount; index++ {
					batch := index / keyValueHeadCount
					head := index % keyValueHeadCount
					count := cache.compressedCount[layer][batch]
					keys := make([]float32, count*width)
					reader.f32s(keys)
					for entry := 0; entry < count; entry++ {
						cache.storage.appendIndexer(layer, batch, head, entry, keys[entry*width:(entry+1)*width])
					}
				}
			}
		}
	}

	if flags&kvFlagMLA != 0 {
		cache.EnableMLA(mlaLatentWidth, mlaRopeKeyWidth)
		for layer := 0; layer < layerCount; layer++ {
			for batch := 0; batch < batchSize; batch++ {
				latent := make([]float32, position*mlaLatentWidth)
				rope := make([]float32, position*mlaRopeKeyWidth)
				reader.f32s(latent)
				reader.f32s(rope)
				for token := 0; token < position; token++ {
					cache.storage.appendMLALatent(layer, batch, token, latent[token*mlaLatentWidth:(token+1)*mlaLatentWidth])
					cache.storage.appendMLARope(layer, batch, token, rope[token*mlaRopeKeyWidth:(token+1)*mlaRopeKeyWidth])
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
	if cache.rawStripped {
		cache.storage.stripRaw()
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
