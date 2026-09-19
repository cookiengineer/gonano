package checkpoint

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/gonano/model"
	"github.com/cookiengineer/gonano/tensors"
)

// TestExportGGUF writes a GGUF file and reads it back with a minimal parser to
// verify the container is well-formed and the tensor data matches.
func TestExportGGUF(tester *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(42))
	params := transformer.NamedParameters()

	path := filepath.Join(tester.TempDir(), "model.gguf")
	if err := ExportGGUF(path, Meta{Step: 7, ModelConfig: config}, params); err != nil {
		tester.Fatalf("ExportGGUF: %v", err)
	}

	data := readFileBytes(tester, path)
	position := 0
	magic := binary.LittleEndian.Uint32(data[position:])
	position += 4
	if magic != ggufMagic {
		tester.Fatalf("magic = %#x, want %#x", magic, ggufMagic)
	}
	version := binary.LittleEndian.Uint32(data[position:])
	position += 4
	if version != ggufVersion {
		tester.Fatalf("version = %d, want %d", version, ggufVersion)
	}
	numTensors := binary.LittleEndian.Uint64(data[position:])
	position += 8
	numKV := binary.LittleEndian.Uint64(data[position:])
	position += 8
	if numTensors != uint64(len(params)) {
		tester.Fatalf("tensors = %d, want %d", numTensors, len(params))
	}
	if numKV == 0 {
		tester.Fatal("no metadata KV pairs")
	}

	// Skip metadata KVs.
	for index := 0; index < int(numKV); index++ {
		valueType := binary.LittleEndian.Uint32(data[position:])
		position += 4
		position = skipString(data, position)
		position = skipValue(data, position, valueType)
	}

	// Read tensor infos and verify data.
	type info struct {
		name   string
		dims   []uint64
		offset uint64
	}
	infos := make([]info, 0, numTensors)
	for index := 0; index < int(numTensors); index++ {
		name, nextPosition := readString(data, position)
		position = nextPosition
		numDimensions := binary.LittleEndian.Uint32(data[position:])
		position += 4
		dims := make([]uint64, numDimensions)
		for dimension := 0; dimension < int(numDimensions); dimension++ {
			dims[dimension] = binary.LittleEndian.Uint64(data[position:])
			position += 8
		}
		ggmlType := binary.LittleEndian.Uint32(data[position:])
		position += 4
		offset := binary.LittleEndian.Uint64(data[position:])
		position += 8
		if ggmlType != ggmlF32 {
			tester.Fatalf("tensor %s type = %d, want f32", name, ggmlType)
		}
		infos = append(infos, info{name: name, dims: dims, offset: offset})
	}

	// Verify each tensor's data matches the original.
	for _, tensorInfo := range infos {
		param := params[tensorInfo.name]
		if param == nil {
			tester.Fatalf("unknown tensor %s", tensorInfo.name)
		}
		if len(tensorInfo.dims) != len(param.Shape) {
			tester.Fatalf("tensor %s dims = %v, want %v", tensorInfo.name, tensorInfo.dims, param.Shape)
		}
		for dimension := range tensorInfo.dims {
			if tensorInfo.dims[dimension] != uint64(param.Shape[dimension]) {
				tester.Fatalf("tensor %s dim %d = %d, want %d", tensorInfo.name, dimension, tensorInfo.dims[dimension], param.Shape[dimension])
			}
		}
		for index, value := range param.Data {
			got := math.Float32frombits(binary.LittleEndian.Uint32(data[int(tensorInfo.offset)+index*4:]))
			if got != value {
				tester.Fatalf("tensor %s[%d] = %v, want %v", tensorInfo.name, index, got, value)
			}
		}
	}
}

func readFileBytes(tester *testing.T, path string) []byte {
	tester.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		tester.Fatal(err)
	}
	return data
}

// TestExportGGUFRoundtrip writes a GGUF, reads it back with LoadGGUF, rebuilds
// the model, and checks the forward pass matches the original model exactly.
func TestExportGGUFRoundtrip(tester *testing.T) {
	config := model.Config{
		SequenceLen: 16, VocabSize: 32, NumLayer: 2, NumHead: 2, NumKVHead: 2,
		EmbedDim: 32, WindowPattern: "L",
	}
	transformer := model.NewTransformer(config)
	transformer.InitWeights(tensors.NewRNG(7))

	path := filepath.Join(tester.TempDir(), "model.gguf")
	if err := ExportGGUF(path, Meta{Step: 11, ModelConfig: config}, transformer.NamedParameters()); err != nil {
		tester.Fatalf("ExportGGUF: %v", err)
	}

	meta, params, err := LoadGGUF(path)
	if err != nil {
		tester.Fatalf("LoadGGUF: %v", err)
	}
	if meta.Step != 11 {
		tester.Fatalf("step = %d, want 11", meta.Step)
	}
	if meta.ModelConfig != config {
		tester.Fatalf("config = %+v, want %+v", meta.ModelConfig, config)
	}

	reloaded := LoadModel(meta, params)
	inputIDs := tensors.NewInt32sWithData([]int{1, 5}, []int32{1, 2, 3, 4, 5})
	original := transformer.Forward(inputIDs, nil)
	rebuilt := reloaded.Forward(inputIDs, nil)
	for index := range original.Data {
		if original.Data[index] != rebuilt.Data[index] {
			tester.Fatalf("forward mismatch at %d after GGUF round-trip", index)
		}
	}
}

func skipString(data []byte, position int) int {
	length := binary.LittleEndian.Uint64(data[position:])
	return position + 8 + int(length)
}

func readString(data []byte, position int) (string, int) {
	length := binary.LittleEndian.Uint64(data[position:])
	position += 8
	return string(data[position : position+int(length)]), position + int(length)
}

func skipValue(data []byte, position int, valueType uint32) int {
	switch valueType {
	case vUint32, vFloat32:
		return position + 4
	case vUint64:
		return position + 8
	case vBool:
		return position + 1
	case vString:
		return skipString(data, position)
	case vArray:
		elementType := binary.LittleEndian.Uint32(data[position:])
		position += 4
		length := binary.LittleEndian.Uint64(data[position:])
		position += 8
		for index := 0; index < int(length); index++ {
			position = skipValue(data, position, elementType)
		}
		return position
	}
	return position
}
