# gonano — Export Guide (GGUF)

This guide explains how to get your trained weights out of gonano, in two
formats:

1. **`.gn`** — gonano's native checkpoint (used for gonano deployment).
2. **`.gguf`** — the GGUF container format, for interop/archival.

> **Setup.** All commands below run from the repo root with
> `export GOEXPERIMENT=simd`. See [00-quickstart.md](00-quickstart.md) for the
> copy-pasteable ArchLinux install.

> **Important architectural note.** GGUF is a *container*, not a model
> definition. gonano's transformer is a **custom architecture** (rotary
> embeddings, QK-normalization, ReLU² MLP, group-query attention, value
> embeddings, smear/backout) that does **not** map onto a stock llama.cpp
> architecture. The exported GGUF is therefore a faithful, self-describing
> weight container tagged `general.architecture = "nanochat"`. A stock
> llama.cpp/ollama build will *not* execute it; you need a nanochat-compatible
> runner (see the deployment guide for the supported gonano runner).

---

## 1. The `.gn` checkpoint format

gonano already saves checkpoints during training
(`$GONANO_BASE_DIR/base_checkpoints/<tag>/model_<step>.gn`). This is the
format you should use to move weights between gonano processes.

Layout (`checkpoint/checkpoint.go`):

```
magic "GONANO\x00\x01"          (8 bytes)
meta length                    (4 bytes, little-endian uint32)
meta JSON                      (config, step, val_bpb, param names/shapes)
raw float32 tensor blobs       (in sorted-name order)
```

Reading it back:

```go
meta, params, err := checkpoint.Load(path)
model := checkpoint.LoadModel(meta, params) // *model.Transformer
```

The metadata carries the full `model.Config` (`sequence_len`, `vocab_size`,
`n_layer`, `n_head`, `n_kv_head`, `n_embd`, `window_pattern`), so a `.gn` file
is fully self-describing.

---

## 2. Exporting to GGUF

The built-in exporter writes a valid GGUF v3 file with all weights in **float32**:

```bash
go run ./cmd/export \
  --model ~/.cache/gonano/base_checkpoints/d4/model_000200.gn \
  --out    ~/models/d4.gguf
```

Or programmatically:

```go
meta, params, err := checkpoint.Load("model_000200.gn")
err = checkpoint.ExportGGUF("d4.gguf", meta, params)
```

Implementation: `checkpoint/gguf.go` (`ExportGGUF`).

### What is written

**Metadata keys** (all prefixed for the nanochat architecture):

| Key | Type | Value |
|---|---|---|
| `general.architecture` | string | `"nanochat"` |
| `general.name` | string | `"gonano"` |
| `nanochat.context_length` | uint32 | `sequence_len` |
| `nanochat.embedding_length` | uint32 | `n_embd` |
| `nanochat.block_count` | uint32 | `n_layer` |
| `nanochat.head_count` | uint32 | `n_head` |
| `nanochat.kv_head_count` | uint32 | `n_kv_head` |
| `nanochat.head_dim` | uint32 | `n_embd / n_head` |
| `nanochat.vocab_size` | uint32 | `vocab_size` |
| `nanochat.window_pattern` | string | e.g. `"SSSL"` |
| `nanochat.step` | uint32 | training step |
| `nanochat.value_embedding_layers` | uint32 | number of ResFormer layers |
| `nanochat.special_tokens` | array[string] | the 9 special tokens |
| `tokenizer.ggml.model` | string | `"gpt2"` (byte-level BPE) |
| `tokenizer.ggml.bos_token_id` | uint32 | `<|bos|>` id |

**Tensors** — one GGUF tensor per `model.Transformer.NamedParameters()` entry,
keeping the exact names so a custom loader can reconstruct the model
unambiguously:

```
transformer.wte.weight                     [padded_vocab, n_embd]
transformer.h.{i}.attn.c_q.weight          [n_head*d,  n_embd]
transformer.h.{i}.attn.c_k.weight          [n_kv_head*d, n_embd]
transformer.h.{i}.attn.c_v.weight          [n_kv_head*d, n_embd]
transformer.h.{i}.attn.c_proj.weight       [n_embd, n_head*d]
transformer.h.{i}.attn.ve_gate.weight      [n_kv_head, 12]        (ResFormer layers only)
transformer.h.{i}.mlp.c_fc.weight          [4*n_embd, n_embd]
transformer.h.{i}.mlp.c_proj.weight        [n_embd, 4*n_embd]
value_embeds.{i}.weight                    [padded_vocab, n_kv_head*d]
lm_head.weight                             [padded_vocab, n_embd]
resid_lambdas                              [n_layer]
x0_lambdas                                 [n_layer]
smear_gate.weight                          [1, 24]
smear_lambda                               [1]
backout_lambda                             [1]
```

### Validating an export

The GGUF writer is unit-tested (`checkpoint/gguf_test.go`): it writes the file
and reads it back with a minimal GGUF parser to verify the magic, version,
metadata, tensor count, shapes, and raw float32 data. To inspect a file with a
third-party tool:

```bash
python -c "import gguf; print(gguf.GGUFReader('d4.gguf').fields)"
```

---

## 3. (Optional) Quantization

gonano exports **float32** weights. Quantization (int8/fp8/fp16) is not
implemented in the exporter; the model and training are float32 by design (see
the implementation plan's "precision" note). If you need a smaller artifact,
quantize *after* export with an external tool, or store the `.gn`/`.gguf` as-is
— the f32 weights are the ground truth.

---

## 4. Which format should I use?

| Goal | Format | Command |
|---|---|---|
| Move weights between gonano runs | `.gn` | `base_train` writes it; `checkpoint.Load` reads it |
| Inspect/archive weights portably | `.gguf` | `go run ./cmd/export` |
| Run inference **now** | `.gn` or `.gguf` | `chat_cli` / `infer_bench` (both accept either) |
| Run in llama.cpp/ollama | — | not supported (custom architecture) |

`chat_cli` and `infer_bench` load either format automatically
(`checkpoint.LoadAny`, which dispatches to `LoadGGUF` for `.gguf` and `Load`
for `.gn`), so an exported GGUF is a complete, loadable artifact for gonano
itself — the export round-trips losslessly.

Next: [Deployment & usage guide](03-deployment.md).
