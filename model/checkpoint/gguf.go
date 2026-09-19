package checkpoint

import (
	"encoding/binary"
	"math"
	"os"
	"sort"

	"github.com/cookiengineer/gonano/tensors"
	"github.com/cookiengineer/gonano/tokenizer"
)

// GGUF export. GGUF is a self-describing container for tensor weights. gonano
// stores its weights in GGUF with a custom architecture tag ("nanochat"),
// because the nanochat transformer (RoPE, QK-norm, ReLU² MLP, value embeddings,
// smear/backout) is not a standard llama.cpp architecture. The file is a valid
// GGUF container that any GGUF-aware tool can inspect; executing it requires a
// nanochat-compatible runner (see the deployment guide).

const (
	ggufMagic   uint32 = 0x46554747 // "GGUF"
	ggufVersion uint32 = 3

	ggmlF32 uint32 = 0
)

// GGUF metadata value types.
const (
	vUint32  uint32 = 4
	vFloat32 uint32 = 6
	vBool    uint32 = 7
	vString  uint32 = 8
	vArray   uint32 = 9
	vUint64  uint32 = 10
)

// ExportGGUF writes the model weights to a GGUF file. The tensor names match
// Transformer.NamedParameters, so a custom loader can reconstruct the model
// unambiguously.
func ExportGGUF(path string, meta Meta, params map[string]*tensors.Tensor) error {
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)

	config := meta.ModelConfig

	var header []byte
	header = binary.LittleEndian.AppendUint32(header, ggufMagic)
	header = binary.LittleEndian.AppendUint32(header, ggufVersion)
	header = binary.LittleEndian.AppendUint64(header, uint64(len(names)))
	header = binary.LittleEndian.AppendUint64(header, uint64(15)) // metadata KV count

	// Metadata.
	header = appendGGUFStringKV(header, "general.architecture", "nanochat")
	header = appendGGUFStringKV(header, "general.name", "gonano")
	header = appendGGUFUint32KV(header, "nanochat.context_length", uint32(config.SequenceLen))
	header = appendGGUFUint32KV(header, "nanochat.embedding_length", uint32(config.EmbedDim))
	header = appendGGUFUint32KV(header, "nanochat.block_count", uint32(config.NumLayer))
	header = appendGGUFUint32KV(header, "nanochat.head_count", uint32(config.NumHead))
	header = appendGGUFUint32KV(header, "nanochat.kv_head_count", uint32(config.NumKVHead))
	header = appendGGUFUint32KV(header, "nanochat.head_dim", uint32(config.HeadDim()))
	header = appendGGUFUint32KV(header, "nanochat.vocab_size", uint32(config.VocabSize))
	header = appendGGUFStringKV(header, "nanochat.window_pattern", config.WindowPattern)
	header = appendGGUFUint32KV(header, "nanochat.step", uint32(meta.Step))
	header = appendGGUFStringKV(header, "tokenizer.ggml.model", "gpt2")
	header = appendGGUFUint32KV(header, "tokenizer.ggml.bos_token_id", uint32(config.VocabSize-len(tokenizer.SpecialTokens)))
	header = appendGGUFStringArrayKV(header, "nanochat.special_tokens", tokenizer.SpecialTokens)
	header = appendGGUFUint32KV(header, "nanochat.value_embedding_layers", uint32((config.NumLayer+1)/2))

	// Tensor infos (offsets are patched after the aligned header length is
	// known).
	type tensorInfo struct {
		name     string
		offsetAt int
		size     uint64
	}
	infos := make([]tensorInfo, len(names))
	for index, name := range names {
		tensor := params[name]
		header = appendGGUFString(header, name)
		header = binary.LittleEndian.AppendUint32(header, uint32(len(tensor.Shape)))
		for _, dimension := range tensor.Shape {
			header = binary.LittleEndian.AppendUint64(header, uint64(dimension))
		}
		header = binary.LittleEndian.AppendUint32(header, ggmlF32)
		offsetAt := len(header)
		header = binary.LittleEndian.AppendUint64(header, 0) // placeholder
		infos[index] = tensorInfo{name: name, offsetAt: offsetAt, size: uint64(tensor.Numel()) * 4}
	}

	// Align the tensor data section to 32 bytes and patch the offsets.
	alignedHeaderLen := (len(header) + 31) &^ 31
	var dataOffset uint64 = uint64(alignedHeaderLen)
	for _, info := range infos {
		binary.LittleEndian.PutUint64(header[info.offsetAt:], dataOffset)
		dataOffset += info.size
	}
	for len(header) < alignedHeaderLen {
		header = append(header, 0)
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	if _, err := file.Write(header); err != nil {
		return err
	}

	scratch := make([]byte, 0, 1<<20)
	for _, name := range names {
		scratch = scratch[:0]
		for _, value := range params[name].Data {
			scratch = binary.LittleEndian.AppendUint32(scratch, math.Float32bits(value))
		}
		if _, err := file.Write(scratch); err != nil {
			return err
		}
	}
	return nil
}

func appendGGUFUint32KV(buffer []byte, key string, value uint32) []byte {
	buffer = binary.LittleEndian.AppendUint32(buffer, vUint32)
	buffer = appendGGUFString(buffer, key)
	return binary.LittleEndian.AppendUint32(buffer, value)
}

func appendGGUFStringKV(buffer []byte, key, value string) []byte {
	buffer = binary.LittleEndian.AppendUint32(buffer, vString)
	buffer = appendGGUFString(buffer, key)
	return appendGGUFString(buffer, value)
}

func appendGGUFStringArrayKV(buffer []byte, key string, values []string) []byte {
	buffer = binary.LittleEndian.AppendUint32(buffer, vArray)
	buffer = appendGGUFString(buffer, key)
	buffer = binary.LittleEndian.AppendUint32(buffer, vString) // element type
	buffer = binary.LittleEndian.AppendUint64(buffer, uint64(len(values)))
	for _, value := range values {
		buffer = appendGGUFString(buffer, value)
	}
	return buffer
}

func appendGGUFString(buffer []byte, value string) []byte {
	buffer = binary.LittleEndian.AppendUint64(buffer, uint64(len(value)))
	return append(buffer, value...)
}
