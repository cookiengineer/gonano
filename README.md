# gonano

This project is a pure-Go reimplementation of [nanochat](https://github.com/karpathy/nanochat),
and contains an end-to-end LLM training and inference harness. It is built on the experimental
Go 1.27 [`simd`](https://pkg.go.dev/simd) standard-library package, and optimized for parallelization
of training and inference on CPUs that support `AVX-512`.

## Requirements

The `simd` package is experimental and gated behind the `goexperiment.simd` build tag.
**Every** build, test, and run must set it:

```bash
GOEXPERIMENT=simd go build ./...;
GOEXPERIMENT=simd go test ./...;
```

- Vector width is auto-selected from the CPU (128/256/512-bit; AVX-512 via `GODEBUG=simd=512`).
- Numeric precision is **float32** everywhere; parallelism is goroutine-per-op via the `parallel` package.

## Quickstart

```bash
export GOEXPERIMENT=simd;

# Train a tiny model (synthetic data, no dataset required)
go run ./cmd/base_train --depth 4 --max-seq-len 64 --num-iterations 50;

# Chat with it
go run ./cmd/chat_cli --model ~/.cache/gonano/base_checkpoints/d4/model_000050.gn --prompt "the capital of France is";

# Benchmark inference
go run ./cmd/infer_bench --model ~/.cache/gonano/base_checkpoints/d4/model_000050.gn;
```

Train a tokenizer from Parquet text shards:

```bash
go run ./cmd/tok_train --data-dir /path/to/parquet-shards --vocab-size 32768;
```

Download a base English corpus (FineWeb-Edu shards):

```bash
go run ./cmd/dataset --repo HuggingFaceFW/fineweb-edu --config sample-10BT --split train --num 20;
```

Pretrain on real data (Parquet or Markdown):

```bash
# Parquet shards (the "text" column)
go run ./cmd/base_train --depth 20 --data-dir ~/.cache/gonano/base_data --num-iterations 10000;

# Your own webdata encoded as Markdown (one document per .md file)
go run ./cmd/base_train --depth 20 --data-dir ~/webdata-md --data-format markdown;
```

Or use the one-command `trainer.sh` wrapper, which also sets up a tokenizer
for you (training one, loading an existing one, or copying the bundled default):

```bash
# Markdown webdata (uses the bundled default tokenizer if none exists yet)
./trainer.sh ~/webdata-md --format markdown --depth 20 --num-iterations 10000;

# Parquet shards, training a fresh tokenizer on the data
./trainer.sh ~/.cache/gonano/base_data --format parquet --train-tokenizer --depth 20;
```

`trainer.sh` wraps `cmd/trainer` (an end-to-end training CLI) and ships default
tokenizers in `tokenizer/defaults/` — `markdown.json` (BPE tuned for Markdown)
and `byte.json` (a byte-level fallback). The format-appropriate one is copied
into place when no trained tokenizer exists; regenerate both with
`go run ./cmd/tok_default`.

Export weights to GGUF:

```bash
go run ./cmd/export --model ~/.cache/gonano/base_checkpoints/d20/model_010000.gn --out d20.gguf;
```

## Benchmarking

Measure inference latency and throughput across decode batch sizes:

```bash
bash benchmark.sh;
```

`infer_bench` prints, per batch size, the time-to-first-token (`TTFT`), the
per-token decode latency (`TPOT`), overall tokens/second (`tok/s`), and pure
decode tokens/second (`decode tok/s`, excluding prefill). Decode is
memory-bandwidth-bound, so tokens/second rises with batch size. The prompt
length is clamped automatically to fit the model's context window.

The benchmark uses a synthetic prompt and never decodes text, so it works even
without a trained tokenizer (it falls back to a byte-level tokenizer and logs a
warning).

## Serving (OpenAI-compatible API)

Run an OpenAI-compatible HTTP server (chat completions, streaming, tool calls):

```bash
go run ./cmd/server --model ~/.cache/gonano/base_checkpoints/d4/model_000050.gn --addr :8080;

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gonano","messages":[{"role":"user","content":"what is 2+2?"}]}';
```

Tool calls are executed server-side in Go via the `server` package (the built-in
calculator, plus any tools you register in `cmd/server`). See
`guides/03-deployment.md` for the Go library usage.

See `guides/00-quickstart.md` for a copy-pasteable ArchLinux setup, and `guides/` for the
step-by-step training, export, deployment, and debugging guides.

## Packages

| Package             | Responsibility |
|:--------------------|:-----------------------------------------------------------------------------------------------------------------|
| `tensor`            | Dense float32/int32 tensors + SIMD kernels (matmul, softmax, RMSNorm, reductions)                                |
| `nn`                | Linear, embedding, initializers                                                                                  |
| `model`             | The nanochat GPT transformer (RoPE, QK-norm, GQA, value embeddings, sliding windows) + training forward/backward |
| `optim`             | AdamW + Muon (Polar Express) + MuonAdamW                                                                         |
| `tokenizer`         | Byte-level BPE training/inference + chat rendering                                                               |
| `data`              | Parquet reader, Snappy, HF-Hub download, Markdown source, BOS-aligned dataloaders                                |
| `train`             | Scaling laws, schedulers, pretraining/SFT/RL loops                                                               |
| `infer`             | KV-cache engine, sampler, calculator tool, benchmark                                                             |
| `server`            | OpenAI-compatible HTTP API (chat completions, streaming, tool calls)                                             |
| `eval`              | BPB, CORE, ChatCORE + `eval/tasks` (MMLU/GSM8K/ARC/HumanEval/SmolTalk)                                           |
| `exec`              | Sandboxed Python execution (HumanEval)                                                                           |
| `checkpoint`        | Versioned binary checkpoint save/load + GGUF export                                                              |
| `parallel`          | Goroutine worker pool                                                                                            |
| `device`, `logging` | Hardware detection, logging/metrics                                                                              |

See `guides/00-architecture-overview.md` for how the components fit together.

## Testing

Every feature has unit tests, including scalar-vs-SIMD parity tests, numerical
gradient checks of the backprop, and an end-to-end train, save, load, generate
test:

```bash
GOEXPERIMENT=simd go test -race ./...;
```

## License

MIT
