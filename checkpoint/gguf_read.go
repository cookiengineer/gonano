package checkpoint

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
)

// LoadGGUF reads a GGUF file exported by ExportGGUF and reconstructs the
// metadata and parameters. The nanochat architecture metadata is read back into
// model.Config, so the result can be passed straight to LoadModel.
func LoadGGUF(path string) (Meta, map[string]*tensor.Tensor, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Meta{}, nil, err
	}
	if len(data) < 4 || binary.LittleEndian.Uint32(data[0:]) != ggufMagic {
		return Meta{}, nil, fmt.Errorf("gguf: invalid magic in %s", path)
	}
	pos := 4
	if int(binary.LittleEndian.Uint32(data[pos:])) > len(data)-pos {
		return Meta{}, nil, fmt.Errorf("gguf: truncated version")
	}
	pos += 4 // version (v3)

	nTensors := binary.LittleEndian.Uint64(data[pos:])
	pos += 8
	nKV := binary.LittleEndian.Uint64(data[pos:])
	pos += 8

	kv := make(map[string]any, nKV)
	for i := 0; i < int(nKV); i++ {
		typ := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		key, p := ggufReadString(data, pos)
		pos = p
		val, p, err := ggufReadValue(data, pos, typ)
		if err != nil {
			return Meta{}, nil, err
		}
		pos = p
		kv[key] = val
	}

	type tinfo struct {
		name   string
		shape  []int
		ggtype uint32
		offset uint64
	}
	infos := make([]tinfo, 0, nTensors)
	for i := 0; i < int(nTensors); i++ {
		name, p := ggufReadString(data, pos)
		pos = p
		nDims := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		shape := make([]int, nDims)
		for d := 0; d < int(nDims); d++ {
			shape[d] = int(binary.LittleEndian.Uint64(data[pos:]))
			pos += 8
		}
		ggtype := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		offset := binary.LittleEndian.Uint64(data[pos:])
		pos += 8
		infos = append(infos, tinfo{name: name, shape: shape, ggtype: ggtype, offset: offset})
	}

	cfg, err := configFromGGUF(kv)
	if err != nil {
		return Meta{}, nil, err
	}

	params := make(map[string]*tensor.Tensor, len(infos))
	for _, in := range infos {
		if in.ggtype != ggmlF32 {
			return Meta{}, nil, fmt.Errorf("gguf: tensor %s is not f32 (type %d)", in.name, in.ggtype)
		}
		t := tensor.New(in.shape...)
		base := int(in.offset)
		for i := range t.Data {
			t.Data[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[base+i*4:]))
		}
		params[in.name] = t
	}

	meta := Meta{ModelConfig: cfg}
	if v, ok := kv["nanochat.step"].(uint32); ok {
		meta.Step = int(v)
	}
	return meta, params, nil
}

// LoadAny loads a checkpoint, auto-detecting the format by file extension:
// ".gguf" uses LoadGGUF, everything else uses Load.
func LoadAny(path string) (Meta, map[string]*tensor.Tensor, error) {
	if strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return LoadGGUF(path)
	}
	return Load(path)
}

func configFromGGUF(kv map[string]any) (model.Config, error) {
	getU32 := func(key string) (int, error) {
		v, ok := kv[key].(uint32)
		if !ok {
			return 0, fmt.Errorf("gguf: missing metadata key %q", key)
		}
		return int(v), nil
	}
	getStr := func(key string) (string, error) {
		v, ok := kv[key].(string)
		if !ok {
			return "", fmt.Errorf("gguf: missing metadata key %q", key)
		}
		return v, nil
	}

	var cfg model.Config
	var err error
	if cfg.SequenceLen, err = getU32("nanochat.context_length"); err != nil {
		return cfg, err
	}
	if cfg.VocabSize, err = getU32("nanochat.vocab_size"); err != nil {
		return cfg, err
	}
	if cfg.NumLayer, err = getU32("nanochat.block_count"); err != nil {
		return cfg, err
	}
	if cfg.NumHead, err = getU32("nanochat.head_count"); err != nil {
		return cfg, err
	}
	if cfg.NumKVHead, err = getU32("nanochat.kv_head_count"); err != nil {
		return cfg, err
	}
	if cfg.EmbedDim, err = getU32("nanochat.embedding_length"); err != nil {
		return cfg, err
	}
	if cfg.WindowPattern, err = getStr("nanochat.window_pattern"); err != nil {
		return cfg, err
	}
	cfg.Validate()
	return cfg, nil
}

func ggufReadString(data []byte, pos int) (string, int) {
	n := binary.LittleEndian.Uint64(data[pos:])
	pos += 8
	return string(data[pos : pos+int(n)]), pos + int(n)
}

func ggufReadValue(data []byte, pos int, typ uint32) (any, int, error) {
	switch typ {
	case vUint32, vFloat32:
		if pos+4 > len(data) {
			return nil, 0, fmt.Errorf("gguf: truncated value")
		}
		return binary.LittleEndian.Uint32(data[pos:]), pos + 4, nil
	case vUint64:
		if pos+8 > len(data) {
			return nil, 0, fmt.Errorf("gguf: truncated value")
		}
		return binary.LittleEndian.Uint64(data[pos:]), pos + 8, nil
	case vBool:
		return data[pos] != 0, pos + 1, nil
	case vString:
		s, p := ggufReadString(data, pos)
		return s, p, nil
	case vArray:
		elemType := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		n := binary.LittleEndian.Uint64(data[pos:])
		pos += 8
		elems := make([]any, 0, n)
		for i := 0; i < int(n); i++ {
			v, p, err := ggufReadValue(data, pos, elemType)
			if err != nil {
				return nil, 0, err
			}
			elems = append(elems, v)
			pos = p
		}
		return elems, pos, nil
	default:
		return nil, 0, fmt.Errorf("gguf: unsupported value type %d", typ)
	}
}
