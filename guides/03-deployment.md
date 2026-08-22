# gonano — Deployment & Usage Guide

This guide shows how to load trained weights and run inference, both from the
command line and from your own Go code (gonano is a library).

---

## 1. What you need

- A trained checkpoint: `$GONANO_BASE_DIR/base_checkpoints/<tag>/model_<step>.gn`
  (or an SFT checkpoint).
- The tokenizer: `$GONANO_BASE_DIR/tokenizer/tokenizer.json`.
- `export GOEXPERIMENT=simd` for every build/run.

Both are produced by the training guide. The base directory defaults to
`~/.cache/gonano` and is overridable with `GONANO_BASE_DIR`.

> **New to the project?** [00-quickstart.md](00-quickstart.md) walks through
> install + a smoke run from a fresh ArchLinux host.

---

## 2. Chat from the command line

```bash
export GOEXPERIMENT=simd

# Single-shot prompt:
go run ./cmd/chat_cli \
  --model ~/.cache/gonano/base_checkpoints/d4/model_000200.gn \
  --prompt "the capital of France is" \
  --max-tokens 64 --temperature 0.6

# Interactive:
go run ./cmd/chat_cli \
  --model ~/.cache/gonano/chatsft_checkpoints/d4/model_000200.gn
```

`chat_cli` wraps a conversation in the special tokens
(`<|bos|> <|user_start|> … <|user_end|> <|assistant_start|>`) and streams the
assistant response until `<|assistant_end|>`.

Flags: `--temperature`, `--top-k`, `--max-tokens`, `--model`.

`--model` accepts either a `.gn` checkpoint or an exported `.gguf` file —
`checkpoint.LoadAny` auto-detects the format, so this works too:

```bash
go run ./cmd/export --model .../model_000200.gn --out d4.gguf
go run ./cmd/chat_cli --model d4.gguf --prompt "why is the sky blue?"
```

---

## 3. Benchmark throughput

```bash
go run ./cmd/infer_bench \
  --model ~/.cache/gonano/base_checkpoints/d4/model_000200.gn \
  --batch-sizes 1,4,16 --decode-tokens 64
```

Reports TTFT (time-to-first-token), TPOT (per-token latency), overall `tok/s`,
and pure decode `decode tok/s` per batch size. Decode is
memory-bandwidth-bound, so larger batches give more tok/s until
compute saturates — this is where goroutine-per-op parallelism (`parallel`)
pays off.

---

## 4. Load weights and infer from Go

The minimal program: load the checkpoint, rebuild the model, and generate.

```go
package main

import (
    "fmt"

    "github.com/cookiengineer/gonano/checkpoint"
    "github.com/cookiengineer/gonano/infer"
    "github.com/cookiengineer/gonano/tokenizer"
)

func main() {
    meta, params, err := checkpoint.LoadAny("model_000200.gn") // or model.gguf
    check(err)
    model := checkpoint.LoadModel(meta, params) // *model.Transformer

    tok, err := tokenizer.LoadTokenizer("tokenizer.json")
    check(err)

    engine := infer.NewEngine(model, tok)

    prompt := []int{tok.BOSTokenID()}
    prompt = append(prompt, tok.Encode("the capital of France is")...)

    results, _ := engine.GenerateBatch(prompt, 1, 64, 0.6, 50, 42)
    fmt.Println(tok.Decode(results[0]))
}
```

### The pieces

| Step | API |
|---|---|
| Load weights (`.gn` or `.gguf`) | `checkpoint.LoadAny(path)` → `(Meta, map[string]*tensor.Tensor)` |
| Rebuild model | `checkpoint.LoadModel(meta, params)` → `*model.Transformer` |
| Load tokenizer | `tokenizer.LoadTokenizer(path)` |
| Build engine | `infer.NewEngine(model, tokenizer)` |
| Generate | `engine.GenerateBatch(prompt, numSamples, maxTokens, temperature, topK, seed)` |

### How inference works under the hood (`infer/engine.go`)

1. **Prefill** — the prompt is run once through `model.Forward` with a batch-1
   KV cache, populating keys/values.
2. **Replicate** — `model.PrefillFrom` clones the KV cache across `numSamples`
   rows (and expands the smear state).
3. **Decode loop** — for each step, `infer.SampleNextToken` samples the next
   token per row (temperature/top-k/argmax), the tool-use state machine
   (`infer.UseCalculator`) handles `<|python_start|>…<|python_end|>`, and the
   next single-token column is forwarded against the cache.

The KV cache lives in `model.KVBuffer` (`model/kvcache.go`).

---

## 5. Streaming generation

For a server, use `Engine.Generate` (the streaming iterator) instead of
`GenerateBatch`:

```go
gen := engine.Generate(prompt, numSamples, maxTokens, temp, topK, seed)
gen(func(column []int, mask []int) bool {
    // column[i] is the next token for row i; mask[i]==1 means "sampled",
    // mask[i]==0 means "forced" (tool output).
    fmt.Print(tok.Decode(column))
    return true // continue; return false to stop
})
```

Each call to the callback is one decode step. `GenerateBatch` is a convenience
wrapper over this that accumulates the final sequences.

---

## 6. Model configuration for reference

The architecture is defined entirely by `model.Config`, stored in the
checkpoint metadata:

```go
type Config struct {
    SequenceLen   int    // context length
    VocabSize     int    // vocabulary size (mergeable + special)
    NumLayer      int    // depth
    NumHead       int    // query heads
    NumKVHead     int    // key/value heads (GQA)
    EmbedDim      int    // width
    WindowPattern string // "L" = full, "S" = sliding, tiled across layers
}
```

Key facts for anyone writing a custom loader:

- **No biases** anywhere; RMSNorm has no learned parameters.
- **Untied embeddings**: `transformer.wte.weight` ≠ `lm_head.weight`.
- **Rotary embeddings** (base 100000) with **QK-norm** and a 1.2 scale.
- **ReLU²** MLP activation; **4×** expansion.
- **Value embeddings** on alternating layers (the last layer always has one).
- **Softcap** (15·tanh(x/15)) applied to logits before sampling/loss.
- The vocabulary is padded to a multiple of 64 internally; logits are cropped
  back to `vocab_size` before sampling.

---

## 7. Serving in production

For a long-running service:

- Reuse a single `*infer.Engine` (it holds no per-request state; create one per
  request row or manage caches per session).
- For concurrent requests, run independent `Generate` calls in separate
  goroutines — the `parallel` pool sizes itself to `GOMAXPROCS`.
- Measure with `infer.Measure` before and after changes.

Next: [Debugging guide](04-debugging.md).
