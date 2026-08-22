package checkpoint

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensor"
)

// TestExportGGUF writes a GGUF file and reads it back with a minimal parser to
// verify the container is well-formed and the tensor data matches.
func TestExportGGUF(t *testing.T) {
	cfg := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(42))
	params := m.NamedParameters()

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := ExportGGUF(path, Meta{Step: 7, ModelConfig: cfg}, params); err != nil {
		t.Fatalf("ExportGGUF: %v", err)
	}

	data := readFileBytes(t, path)
	pos := 0
	magic := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	if magic != ggufMagic {
		t.Fatalf("magic = %#x, want %#x", magic, ggufMagic)
	}
	version := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	if version != ggufVersion {
		t.Fatalf("version = %d, want %d", version, ggufVersion)
	}
	nTensors := binary.LittleEndian.Uint64(data[pos:])
	pos += 8
	nKV := binary.LittleEndian.Uint64(data[pos:])
	pos += 8
	if nTensors != uint64(len(params)) {
		t.Fatalf("tensors = %d, want %d", nTensors, len(params))
	}
	if nKV == 0 {
		t.Fatal("no metadata KV pairs")
	}

	// Skip metadata KVs.
	for i := 0; i < int(nKV); i++ {
		typ := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		pos = skipString(data, pos)
		pos = skipValue(data, pos, typ)
	}

	// Read tensor infos and verify data.
	type info struct {
		name   string
		dims   []uint64
		offset uint64
	}
	infos := make([]info, 0, nTensors)
	for i := 0; i < int(nTensors); i++ {
		name, p := readString(data, pos)
		pos = p
		nDims := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		dims := make([]uint64, nDims)
		for d := 0; d < int(nDims); d++ {
			dims[d] = binary.LittleEndian.Uint64(data[pos:])
			pos += 8
		}
		ggmlType := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		offset := binary.LittleEndian.Uint64(data[pos:])
		pos += 8
		if ggmlType != ggmlF32 {
			t.Fatalf("tensor %s type = %d, want f32", name, ggmlType)
		}
		infos = append(infos, info{name: name, dims: dims, offset: offset})
	}

	// Verify each tensor's data matches the original.
	for _, in := range infos {
		p := params[in.name]
		if p == nil {
			t.Fatalf("unknown tensor %s", in.name)
		}
		if len(in.dims) != len(p.Shape) {
			t.Fatalf("tensor %s dims = %v, want %v", in.name, in.dims, p.Shape)
		}
		for d := range in.dims {
			if in.dims[d] != uint64(p.Shape[d]) {
				t.Fatalf("tensor %s dim %d = %d, want %d", in.name, d, in.dims[d], p.Shape[d])
			}
		}
		for i, v := range p.Data {
			got := math.Float32frombits(binary.LittleEndian.Uint32(data[int(in.offset)+i*4:]))
			if got != v {
				t.Fatalf("tensor %s[%d] = %v, want %v", in.name, i, got, v)
			}
		}
	}
}

func readFileBytes(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// TestExportGGUFRoundtrip writes a GGUF, reads it back with LoadGGUF, rebuilds
// the model, and checks the forward pass matches the original model exactly.
func TestExportGGUFRoundtrip(t *testing.T) {
	cfg := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	m := model.NewTransformer(cfg)
	m.InitWeights(tensor.NewRNG(7))

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := ExportGGUF(path, Meta{Step: 11, ModelConfig: cfg}, m.NamedParameters()); err != nil {
		t.Fatalf("ExportGGUF: %v", err)
	}

	meta, params, err := LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	if meta.Step != 11 {
		t.Fatalf("step = %d, want 11", meta.Step)
	}
	if meta.ModelConfig != cfg {
		t.Fatalf("config = %+v, want %+v", meta.ModelConfig, cfg)
	}

	rebuilt := LoadModel(meta, params)
	idx := tensor.NewInt32sWithData([]int{1, 5}, []int32{1, 2, 3, 4, 5})
	a := m.Forward(idx, nil)
	b := rebuilt.Forward(idx, nil)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("forward mismatch at %d after GGUF round-trip", i)
		}
	}
}

func skipString(data []byte, pos int) int {
	n := binary.LittleEndian.Uint64(data[pos:])
	return pos + 8 + int(n)
}

func readString(data []byte, pos int) (string, int) {
	n := binary.LittleEndian.Uint64(data[pos:])
	pos += 8
	return string(data[pos : pos+int(n)]), pos + int(n)
}

func skipValue(data []byte, pos int, typ uint32) int {
	switch typ {
	case vUint32, vFloat32:
		return pos + 4
	case vUint64:
		return pos + 8
	case vBool:
		return pos + 1
	case vString:
		return skipString(data, pos)
	case vArray:
		elemType := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		n := binary.LittleEndian.Uint64(data[pos:])
		pos += 8
		for i := 0; i < int(n); i++ {
			pos = skipValue(data, pos, elemType)
		}
		return pos
	}
	return pos
}
