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
- Numeric precision is **float32** everywhere; parallelism is goroutine-per-op via the `internal/parallel` package.

## Features

gonano is optimized for CPU long-context inference and batched decode. The table
below summarizes the measured effect of each optimization. Values are speedups
(`×`, higher is better; below `1×` means slower) for the two inference phases,
split by decode batch size; each row is measured against the baseline named in
its own row, so the cells are not one single end-to-end run.

| Feature | Baseline | Prefill (TTFT) | Decode ×1 | Decode ×16 | Decode ×64 |
|:--------|:---------|---------------:|----------:|-----------:|-----------:|
| Goroutine-parallel kernels | 1 core, seq 64 | — | 1.01× | 2.26× | — |
| Split-K flash decoding | 1 core, seq 1024 | 5.75× | 1.37× | 2.89× | — |
| Vectorized SIMD `exp` | attention kernel, seq 1024 | 1.17× | — | — | — |
| Rank-1 score-tile GEMM | attention kernel, seq 1024 | 1.25× | — | — | — |
| HCA dense KV compression (÷4 sequence) | uncompressed, seq 4096 | 0.7–0.8× | 1.37× | 1.02× | 1.14× |
| CSA sparse attention + hierarchical indexer | compression-only, seq 4096 | ~1.5× | 0.96× | 1.27× | 1.29× |
| Compression + sparsity | uncompressed, seq 4096 | 1.05–1.23× | 1.31× | 1.29× | 1.48× |
| Compression + sparsity + SWA + CED (recommended, long context) | uncompressed, seq 16384 | ~3.3× | 1.6× | 1.8× | 2.1× |

**Recommended configuration:** for long-context workloads enable the full stack,
`--compression-ratio 4 --sparse-topk 8 --swa-window 128 --ced`. It is the best
combined result: prefill ≈3.3× and decode ≈1.6–2.1× versus uncompressed at seq
16384, because CED's bounded replay cuts prefill by another ≈1.5× over SWA-only
without a measurable decode cost. The one exception is a **decode-only**
workload with short prompts, where plain `--compression-ratio 4 --sparse-topk 8`
(no SWA/CED) is ~10–15% faster on decode; SWA and CED trade that decode margin
for local fidelity and the large prefill win.

Architectural features that reduce memory rather than latency:

- **Grouped-query / multi-query attention** (`--kv-head-ratio N`): one KV head
  per `N` query heads. Ratio 3 cuts KV-cache size 3× (no throughput claim; KV
  traffic is reduced proportionally).
- **Partial RoPE** (`RotaryDims`, default 64): DeepSeek-style rotary embedding
  applied only to the trailing head dimensions, with no throughput cost.
- **Hierarchical sparse indexer**: coarse-to-fine block selection bounds the
  number of entries scored per query, making deeper indexing constant-cost in
  context length.

### Local sliding-window attention

`--swa-window N` adds a **layer-local sliding-window branch** to compressed
layers: every query attends to the global compressed blocks *and* the raw
keys/values of the last `N` tokens. The two branches are merged through a single
exact softmax (they share one log-sum-exp, and each branch's backward is
corrected by the merged row term `dO·O`), so the local branch adds local context
without changing the attention semantics. Its cost is bounded by `N` and is
independent of context length, which is the key property for long context.

Because decode is compute/overhead bound, adding a second branch costs
throughput at short context; the cost does not grow with context. Depth-4 models,
`GOMAXPROCS=16`, prompt ≈ seq length, decode 16:

| Model | seq | TTFT (ms) | decode tok/s ×1 | ×16 | ×64 |
|:------|----:|----------:|----------------:|----:|----:|
| compression + sparsity | 4096 | 854–1012 | 1152 | 1610 | 2003 |
| + SWA (`--swa-window 128`) | 4096 | 1051–1173 | 912 | 1449 | 1773 |
| + CED (`--ced`) | 4096 | 739–752 | 975 | 1460 | 1785 |
| compression + sparsity | 16384 | 4566–5023 | 427 | 779 | 872 |
| + SWA (`--swa-window 128`) | 16384 | 5269–5956 | 465 | 735 | 831 |
| + CED (`--ced`) | 16384 | 3604–3641 | 530 | 745 | 855 |
| uncompressed (reference) | 16384 | 11691–12825 | 325 | 408 | 412 |

SWA costs ≈ 10–15% decode and ≈ 11–19% prefill versus compression-only at seq
4096, narrowing to ≈ 5–13% decode at seq 16384. Against the uncompressed
reference the **compression + sparsity + SWA stack is still ≈ 2.0× decode and
≈ 2.2× prefill at seq 16384** — the local branch is what retains local fidelity.

CED shares the SWA config (`--compression-ratio 4 --sparse-topk 8
--swa-window 128`) and cuts prefill TTFT by **≈1.4–1.6× at seq 4096 and
≈1.46–1.51× at seq 16384** versus SWA-only, while decode stays within the
run-to-run spread: decode still runs every layer per token, so CED only adds the
decoder's global K/V projection at decode time. The bounded replay's window
bounds the decoder prefill work independent of context length, which is why the
prefill saving holds from 4k to 16k.

### Causal encoder-decoder prefill

`--ced` activates the Causal Encoder-Decoder split (DeepSeek-V4.1 §2.2). The
bottom half of the layers `[0, d/2)` is a causal encoder; the top half
`[d/2, d)` is a decoder whose **global compressed keys/values are projected from
the encoder's final hidden state** rather than from each decoder layer's own
hidden state. The decoder's local sliding-window keys/values still come from its
own hidden state, and the layer's own query, MLP, and output projections are
unchanged. The projection reuses each decoder layer's existing K/V projections
and compressor, so `--ced` adds **no parameters and no new cache layout**.

Prefill is encoder-only plus a **bounded replay**: the encoder runs over the
full prompt and fills every decoder layer's global compressed cache (and
indexer keys) from its final hidden state; the decoder is then replayed over
only the last `--swa-window` tokens to rebuild its local state. The last prompt
position's logits remain exact when the window covers the replay segment, the
decoder SWA state is used for decoding but never persisted as prefix cache, and
decode is unchanged because every layer still runs per token.

`--ced` requires `--compression-ratio > 1` and `--swa-window > 0`. The M6.1
`--reuse-pattern` is applied independently within each half (restarting at the
decoder split), so a decoder group never borrows an encoder layer's cache; the
decoder producer projects the shared cache from the encoder hidden state.

A/B benchmark (depth 4, ratio 4, top-k 8, pool 8, `--swa-window 128`, prompt ≈
seq, decode 16, `GOMAXPROCS=16`; see the table above and the Features table):

| config | seq | TTFT (ms) | prefill tok/s ×1 | ×16 | ×64 |
|:-------|----:|----------:|-----------------:|----:|----:|
| compression + sparsity + SWA | 4096 | 1051–1173 | 13–15 | 207–210 | 600–603 |
| + CED (`--ced`) | 4096 | 739–752 | 21 | 271–275 | 716–730 |
| compression + sparsity + SWA | 16384 | 5256–5490 | 3 | 44 | 145–147 |
| + CED (`--ced`) | 16384 | 3604–3641 | 4 | 62 | 192–195 |

Decode token rates are unchanged within noise (4k ≈ 975/1460/1785 tok/s,
16k ≈ 530/745/855 tok/s at batch 1/16/64).

### Notes

- Benchmarks: 16-core AMD Ryzen 7 7840HS (AVX-512), Go 1.27.1,
  `GOEXPERIMENT=simd`, `GOMAXPROCS=16`, float32. Reproduce with `benchmark.sh`
  and `cmd/infer_bench`.
- "attention kernel" rows are single-`AttentionForward` timings at
  `seq 1024, head 128`; the other rows are end-to-end `infer_bench` timings.
- `—` means that phase was not measured for that row, not that the effect is zero.
- The HCA prefill cost is an implementation/prefill-load artifact at short
  context; compression pays off in decode and becomes more favourable at longer
  context, where sparsity is layered on top.
- **Low-bit weights/KV were evaluated and rejected**: an int8 experiment was
  slower than fp32 on this platform (the Go `simd` package has no vectorized
  int8→float32 conversion, and decode is compute-bound, not bandwidth-bound), so
  the backend is float32-only.

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
step-by-step training, export, deployment, and debugging guides. The long-context
attention design and the DeepSeek-V4.1-Flash optimizations (CED, CSA2 reuse, the
hierarchical sparse indexer, low-rank query/KV, the KV prefix cache, the persistent
multi-entry KV cache tier, head-wise Muon, Sinkhorn-balanced embeddings, on-policy
distillation, and DSpark speculative decoding) are documented in `guides/05-deepseek-v4.1-optimizations.md`.

## Packages

| Package             | Responsibility |
|:--------------------|:-----------------------------------------------------------------------------------------------------------------|
| `kernels`           | The numeric backend contract (`Backend`): elementwise, reductions, matmul, softmax/RMSNorm, flash attention      |
| `kernels/simd`      | Production SIMD implementation of `kernels.Backend` (AVX-512/AVX2, goroutine-parallel)                           |
| `kernels/scalar`    | Portable pure-Go reference implementation used for parity tests and hosts without SIMD                            |
| `tensors`           | Dense float32/int32 tensor types plus high-level ops that delegate to the active kernel backend                   |
| `model`             | The nanochat GPT transformer (RoPE, QK-norm, GQA, value embeddings, sliding windows) + training forward/backward |
| `model/layers`      | Neural-network building blocks: linear layers, embeddings, initializers                                          |
| `model/checkpoint`  | Versioned binary checkpoint save/load + GGUF export                                                              |
| `optimizer`         | AdamW + Muon (Polar Express) + Sinkhorn-balanced momentum + MuonAdamW                                            |
| `tokenizer`         | Byte-level BPE training/inference + chat rendering                                                               |
| `data`              | Parquet reader, Snappy, HF-Hub download, Markdown source, BOS-aligned dataloaders                                |
| `trainer`           | Scaling laws, schedulers, pretraining/SFT/RL loops                                                               |
| `inference`         | KV-cache engine, sampler, calculator tool, benchmark                                                             |
| `server`            | OpenAI-compatible HTTP API (chat completions, streaming, tool calls)                                             |
| `evaluator`         | BPB, CORE, ChatCORE + `evaluator/tasks` (MMLU/GSM8K/ARC/HumanEval/SmolTalk)                                      |
| `executor`          | Sandboxed Python execution (HumanEval)                                                                           |
| `internal/parallel` | Goroutine worker pool                                                                                            |
| `internal/device`, `internal/logging` | Hardware detection, logging/metrics                                                            |

Swap backends at runtime with `tensors.UseKernelBackend(scalar.New())`; the
default is `kernels/simd`. See `guides/00-architecture-overview.md` for how the
components fit together.

## Testing

Every feature has unit tests: scalar-vs-SIMD parity tests for the entire
`kernels.Backend` contract (including flash attention), numerical gradient
checks of the backprop (including grouped-query attention), and an end-to-end
train, save, load, generate test.

```bash
GOEXPERIMENT=simd go test -race ./...;
```

## License

MIT
