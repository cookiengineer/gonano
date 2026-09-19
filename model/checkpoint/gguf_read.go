package checkpoint

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// LoadGGUF reads a GGUF file exported by ExportGGUF and reconstructs the
// metadata and parameters. The nanochat architecture metadata is read back into
// model.Config, so the result can be passed straight to LoadModel.
func LoadGGUF(path string) (Meta, map[string]*tensors.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Meta{}, nil, err
	}
	if len(data) < 4 || binary.LittleEndian.Uint32(data[0:]) != ggufMagic {
		return Meta{}, nil, fmt.Errorf("gguf: invalid magic in %s", path)
	}
	position := 4
	if int(binary.LittleEndian.Uint32(data[position:])) > len(data)-position {
		return Meta{}, nil, fmt.Errorf("gguf: truncated version")
	}
	position += 4 // version (v3)

	numTensors := binary.LittleEndian.Uint64(data[position:])
	position += 8
	numKV := binary.LittleEndian.Uint64(data[position:])
	position += 8

	metadata := make(map[string]any, numKV)
	for index := 0; index < int(numKV); index++ {
		valueType := binary.LittleEndian.Uint32(data[position:])
		position += 4
		key, nextPosition := ggufReadString(data, position)
		position = nextPosition
		value, nextPosition, err := ggufReadValue(data, position, valueType)
		if err != nil {
			return Meta{}, nil, err
		}
		position = nextPosition
		metadata[key] = value
	}

	type tensorInfo struct {
		name     string
		shape    []int
		ggmlType uint32
		offset   uint64
	}
	infos := make([]tensorInfo, 0, numTensors)
	for index := 0; index < int(numTensors); index++ {
		name, nextPosition := ggufReadString(data, position)
		position = nextPosition
		numDimensions := binary.LittleEndian.Uint32(data[position:])
		position += 4
		shape := make([]int, numDimensions)
		for dimension := 0; dimension < int(numDimensions); dimension++ {
			shape[dimension] = int(binary.LittleEndian.Uint64(data[position:]))
			position += 8
		}
		ggmlType := binary.LittleEndian.Uint32(data[position:])
		position += 4
		offset := binary.LittleEndian.Uint64(data[position:])
		position += 8
		infos = append(infos, tensorInfo{name: name, shape: shape, ggmlType: ggmlType, offset: offset})
	}

	config, err := parseConfigFromGGUF(metadata)
	if err != nil {
		return Meta{}, nil, err
	}

	params := make(map[string]*tensors.Tensor, len(infos))
	for _, info := range infos {
		if info.ggmlType != ggmlF32 {
			return Meta{}, nil, fmt.Errorf("gguf: tensor %s is not f32 (type %d)", info.name, info.ggmlType)
		}
		tensor := tensors.New(info.shape...)
		baseOffset := int(info.offset)
		for index := range tensor.Data {
			tensor.Data[index] = math.Float32frombits(binary.LittleEndian.Uint32(data[baseOffset+index*4:]))
		}
		params[info.name] = tensor
	}

	meta := Meta{ModelConfig: config}
	if value, ok := metadata["nanochat.step"].(uint32); ok {
		meta.Step = int(value)
	}
	return meta, params, nil
}

// LoadAny loads a checkpoint, auto-detecting the format by file extension:
// ".gguf" uses LoadGGUF, everything else uses Load.
func LoadAny(path string) (Meta, map[string]*tensors.Tensor, error) {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return LoadGGUF(path)
	}
	return Load(path)
}

func parseConfigFromGGUF(metadata map[string]any) (model.Config, error) {
	getUint32 := func(key string) (int, error) {
		value, ok := metadata[key].(uint32)
		if !ok {
			return 0, fmt.Errorf("gguf: missing metadata key %q", key)
		}
		return int(value), nil
	}
	getString := func(key string) (string, error) {
		value, ok := metadata[key].(string)
		if !ok {
			return "", fmt.Errorf("gguf: missing metadata key %q", key)
		}
		return value, nil
	}

	var config model.Config
	var err error
	if config.SequenceLen, err = getUint32("nanochat.context_length"); err != nil {
		return config, err
	}
	if config.VocabSize, err = getUint32("nanochat.vocab_size"); err != nil {
		return config, err
	}
	if config.NumLayer, err = getUint32("nanochat.block_count"); err != nil {
		return config, err
	}
	if config.NumHead, err = getUint32("nanochat.head_count"); err != nil {
		return config, err
	}
	if config.NumKVHead, err = getUint32("nanochat.kv_head_count"); err != nil {
		return config, err
	}
	if config.EmbedDim, err = getUint32("nanochat.embedding_length"); err != nil {
		return config, err
	}
	if config.WindowPattern, err = getString("nanochat.window_pattern"); err != nil {
		return config, err
	}
	config.Validate()
	return config, nil
}

func ggufReadString(data []byte, position int) (string, int) {
	length := binary.LittleEndian.Uint64(data[position:])
	position += 8
	return string(data[position : position+int(length)]), position + int(length)
}

func ggufReadValue(data []byte, position int, valueType uint32) (any, int, error) {
	switch valueType {
	case vUint32, vFloat32:
		if position+4 > len(data) {
			return nil, 0, fmt.Errorf("gguf: truncated value")
		}
		return binary.LittleEndian.Uint32(data[position:]), position + 4, nil
	case vUint64:
		if position+8 > len(data) {
			return nil, 0, fmt.Errorf("gguf: truncated value")
		}
		return binary.LittleEndian.Uint64(data[position:]), position + 8, nil
	case vBool:
		return data[position] != 0, position + 1, nil
	case vString:
		value, nextPosition := ggufReadString(data, position)
		return value, nextPosition, nil
	case vArray:
		elementType := binary.LittleEndian.Uint32(data[position:])
		position += 4
		length := binary.LittleEndian.Uint64(data[position:])
		position += 8
		elements := make([]any, 0, length)
		for index := 0; index < int(length); index++ {
			value, nextPosition, err := ggufReadValue(data, position, elementType)
			if err != nil {
				return nil, 0, err
			}
			elements = append(elements, value)
			position = nextPosition
		}
		return elements, position, nil
	default:
		return nil, 0, fmt.Errorf("gguf: unsupported value type %d", valueType)
	}
}
