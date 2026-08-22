package checkpoint

import (
	"encoding/binary"
	"math"
	"os"
	"sort"

	"github.com/cookiengineer/gonano/tensor"
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
func ExportGGUF(path string, meta Meta, params map[string]*tensor.Tensor) error {
	names := make([]string, 0, len(params))
	for name := range params {
		names = append(names, name)
	}
	sort.Strings(names)

	cfg := meta.ModelConfig

	var header []byte
	header = binary.LittleEndian.AppendUint32(header, ggufMagic)
	header = binary.LittleEndian.AppendUint32(header, ggufVersion)
	header = binary.LittleEndian.AppendUint64(header, uint64(len(names)))
	header = binary.LittleEndian.AppendUint64(header, uint64(15)) // metadata KV count

	// Metadata.
	header = ggufKVString(header, "general.architecture", "nanochat")
	header = ggufKVString(header, "general.name", "gonano")
	header = ggufKVU32(header, "nanochat.context_length", uint32(cfg.SequenceLen))
	header = ggufKVU32(header, "nanochat.embedding_length", uint32(cfg.EmbedDim))
	header = ggufKVU32(header, "nanochat.block_count", uint32(cfg.NumLayer))
	header = ggufKVU32(header, "nanochat.head_count", uint32(cfg.NumHead))
	header = ggufKVU32(header, "nanochat.kv_head_count", uint32(cfg.NumKVHead))
	header = ggufKVU32(header, "nanochat.head_dim", uint32(cfg.HeadDim()))
	header = ggufKVU32(header, "nanochat.vocab_size", uint32(cfg.VocabSize))
	header = ggufKVString(header, "nanochat.window_pattern", cfg.WindowPattern)
	header = ggufKVU32(header, "nanochat.step", uint32(meta.Step))
	header = ggufKVString(header, "tokenizer.ggml.model", "gpt2")
	header = ggufKVU32(header, "tokenizer.ggml.bos_token_id", uint32(cfg.VocabSize-len(tokenizer.SpecialTokens)))
	header = ggufKVArrayString(header, "nanochat.special_tokens", tokenizer.SpecialTokens)
	header = ggufKVU32(header, "nanochat.value_embedding_layers", uint32((cfg.NumLayer+1)/2))

	// Tensor infos (offsets are patched after the aligned header length is
	// known).
	type tensorInfo struct {
		name     string
		offsetAt int
		size     uint64
	}
	infos := make([]tensorInfo, len(names))
	for i, name := range names {
		t := params[name]
		header = ggufString(header, name)
		header = binary.LittleEndian.AppendUint32(header, uint32(len(t.Shape)))
		for _, d := range t.Shape {
			header = binary.LittleEndian.AppendUint64(header, uint64(d))
		}
		header = binary.LittleEndian.AppendUint32(header, ggmlF32)
		offsetAt := len(header)
		header = binary.LittleEndian.AppendUint64(header, 0) // placeholder
		infos[i] = tensorInfo{name: name, offsetAt: offsetAt, size: uint64(t.Numel()) * 4}
	}

	// Align the tensor data section to 32 bytes and patch the offsets.
	alignedHeaderLen := (len(header) + 31) &^ 31
	var dataOffset uint64 = uint64(alignedHeaderLen)
	for _, in := range infos {
		binary.LittleEndian.PutUint64(header[in.offsetAt:], dataOffset)
		dataOffset += in.size
	}
	for len(header) < alignedHeaderLen {
		header = append(header, 0)
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(header); err != nil {
		return err
	}

	scratch := make([]byte, 0, 1<<20)
	for _, name := range names {
		scratch = scratch[:0]
		for _, v := range params[name].Data {
			scratch = binary.LittleEndian.AppendUint32(scratch, math.Float32bits(v))
		}
		if _, err := f.Write(scratch); err != nil {
			return err
		}
	}
	return nil
}

func ggufKVU32(b []byte, key string, v uint32) []byte {
	b = binary.LittleEndian.AppendUint32(b, vUint32)
	b = ggufString(b, key)
	return binary.LittleEndian.AppendUint32(b, v)
}

func ggufKVString(b []byte, key, v string) []byte {
	b = binary.LittleEndian.AppendUint32(b, vString)
	b = ggufString(b, key)
	return ggufString(b, v)
}

func ggufKVArrayString(b []byte, key string, v []string) []byte {
	b = binary.LittleEndian.AppendUint32(b, vArray)
	b = ggufString(b, key)
	b = binary.LittleEndian.AppendUint32(b, vString) // element type
	b = binary.LittleEndian.AppendUint64(b, uint64(len(v)))
	for _, s := range v {
		b = ggufString(b, s)
	}
	return b
}

func ggufString(b []byte, s string) []byte {
	b = binary.LittleEndian.AppendUint64(b, uint64(len(s)))
	return append(b, s...)
}
