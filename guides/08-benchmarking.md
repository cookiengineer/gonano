# gonano -- Benchmarking

This guide collects gonano's measured prefill/decode numbers and how to
reproduce them. It is the reference for the latency and throughput claims; the
main [README](../README.md) deliberately omits them.

> **Setup.** Run everything from the repo root with `GOEXPERIMENT=simd`. See
> [00-quickstart.md](00-quickstart.md).

---

## 1. What is measured

`cmd/infer_bench` reports, per decode batch size:

- **TTFT** -- time to first token (prefill latency).
- **TPOT** -- per-token decode latency.
- **tok/s** -- overall tokens per second.
- **decode tok/s** -- pure decode throughput, excluding prefill.
- **weight/KV bytes read** -- the float32 footprint actually touched.

Decode re-reads all weights plus the KV cache for a single token, so it is
memory-bandwidth-bound on small batches and becomes compute-bound as the batch
grows; tokens/second therefore rises with batch size.

```bash
# A single checkpoint across batch sizes:
go run ./cmd/infer_bench \
  --model ~/.cache/gonano/base_checkpoints/d4/model_000200.gn \
  --batch-sizes 1,4,16 --decode-tokens 64

# The standard sweep:
bash benchmark.sh
```

`benchmark.sh` wraps `infer_bench` over a preset batch-size list. The
long-context sweep trains a tiny model for a given architecture config and
sweeps prefill/decode at one or more sequence lengths:

```bash
SEQS=4096,16384 BATCH_SIZES=1,16,64 DECODE_TOKENS=16 ./benchmark_longctx.sh
CONFIG="--preset latent" ./benchmark_longctx.sh
```

Set `STEPS=0` (the default) for a random-weight latency/throughput measurement;
set `STEPS>0` to train a few steps first.

---

## 2. Feature speedups

Values are speedups (`x`, higher is better; below `1x` means slower) for the two
inference phases, split by decode batch size. Each row is measured against the
baseline named in its own row, so the cells are not one single end-to-end run.

| Feature | Baseline | Prefill (TTFT) | Decode x1 | Decode x16 | Decode x64 |
|:--------|:---------|---------------:|----------:|-----------:|-----------:|
| Goroutine-parallel kernels | 1 core, seq 64 | -- | 1.01x | 2.26x | -- |
| Split-K flash decoding | 1 core, seq 1024 | 5.75x | 1.37x | 2.89x | -- |
| Vectorized SIMD `exp` | attention kernel, seq 1024 | 1.17x | -- | -- | -- |
| Rank-1 score-tile GEMM | attention kernel, seq 1024 | 1.25x | -- | -- | -- |
| HCA dense KV compression (/4 sequence) | uncompressed, seq 4096 | 0.7-0.8x | 1.37x | 1.02x | 1.14x |
| CSA sparse attention + hierarchical indexer | compression-only, seq 4096 | ~1.5x | 0.96x | 1.27x | 1.29x |
| Compression + sparsity | uncompressed, seq 4096 | 1.05-1.23x | 1.31x | 1.29x | 1.48x |
| Compression + sparsity + SWA + CED (recommended, long context) | uncompressed, seq 16384 | ~3.3x | 1.6x | 1.8x | 2.1x |

**Recommended configuration.** The `flash` preset is the default and the best
combined result: compression ratio 4, top-k 8, indexer pool 8, reuse `FRU`,
SWA 128, CED. It gives prefill ~3.3x and decode ~1.6-2.1x versus uncompressed at
seq 16384, because CED's bounded replay cuts prefill by another ~1.5x over
SWA-only without a measurable decode cost. The one exception is a **decode-only**
workload with short prompts, where compression + sparsity without SWA/CED is
~10-15% faster on decode; SWA and CED trade that decoder margin for local
fidelity and the large prefill win.

Architectural features that reduce memory rather than latency:

- **Grouped-query / multi-query attention** -- one KV head per `N` query heads;
  cuts KV-cache size without a throughput claim (KV traffic falls
  proportionally).
- **Partial RoPE** (`RotaryDims`, default 64) -- rotary embedding only on the
  trailing head dimensions; no throughput cost.
- **Hierarchical sparse indexer** -- block-max coarse selection bounds the
  entries scored per query, so deeper indexing is constant-cost in context
  length.

---

## 3. Local sliding-window attention (SWA)

The `flash` preset's sliding-window branch (width 128) adds a layer-local branch
to compressed layers: every query attends to the global compressed blocks *and*
the raw keys/values of the last `N` tokens. Both branches share one exact
softmax.

Its cost is bounded by `N` and independent of context length -- the key property
for long context. Because decode is compute/overhead bound, the second branch
costs throughput at short context but the cost does not grow with context.
Depth-4 models, `GOMAXPROCS=16`, prompt ~ seq length, decode 16:

| Model | seq | TTFT (ms) | decode tok/s x1 | x16 | x64 |
|:------|----:|----------:|----------------:|----:|----:|
| compression + sparsity | 4096 | 854-1012 | 1152 | 1610 | 2003 |
| + SWA (`--swa-window 128`) | 4096 | 1051-1173 | 912 | 1449 | 1773 |
| + CED (`--ced`) | 4096 | 739-752 | 975 | 1460 | 1785 |
| compression + sparsity | 16384 | 4566-5023 | 427 | 779 | 872 |
| + SWA (`--swa-window 128`) | 16384 | 5269-5956 | 465 | 735 | 831 |
| + CED (`--ced`) | 16384 | 3604-3641 | 530 | 745 | 855 |
| uncompressed (reference) | 16384 | 11691-12825 | 325 | 408 | 412 |

SWA costs ~10-15% decode and ~11-19% prefill versus compression-only at seq
4096, narrowing to ~5-13% decode at seq 16384. Against the uncompressed
reference the **compression + sparsity + SWA stack is still ~2.0x decode and
~2.2x prefill at seq 16384** -- the local branch is what retains local fidelity.

---

## 4. Causal encoder-decoder prefill (CED)

CED shares the SWA config (`--compression-ratio 4 --sparse-topk 8
--swa-window 128`) and cuts prefill TTFT by **~1.4-1.6x at seq 4096 and
~1.46-1.51x at seq 16384** versus SWA-only, while decode stays within the
run-to-run spread: decode still runs every layer per token, so CED only adds the
decoder's global K/V projection at decode time. The bounded replay's window
bounds the decoder prefill work independent of context length, which is why the
prefill saving holds from 4k to 16k.

Depth-4, ratio 4, top-k 8, pool 8, `--swa-window 128`, prompt ~ seq, decode 16,
`GOMAXPROCS=16`:

| config | seq | TTFT (ms) | prefill tok/s x1 | x16 | x64 |
|:-------|----:|----------:|-----------------:|----:|----:|
| compression + sparsity + SWA | 4096 | 1051-1173 | 13-15 | 207-210 | 600-603 |
| + CED (`--ced`) | 4096 | 739-752 | 21 | 271-275 | 716-730 |
| compression + sparsity + SWA | 16384 | 5256-5490 | 3 | 44 | 145-147 |
| + CED (`--ced`) | 16384 | 3604-3641 | 4 | 62 | 192-195 |

Decode token rates are unchanged within noise (4k ~ 975/1460/1785 tok/s,
16k ~ 530/745/855 tok/s at batch 1/16/64).

---

## 5. Notes

- Benchmarks: 16-core AMD Ryzen 7 7840HS (`AVX-512`), Go 1.27.1,
  `GOEXPERIMENT=simd`, `GOMAXPROCS=16`, float32.
- "attention kernel" rows are single-`AttentionForward` timings at
  `seq 1024, head 128`; the other rows are end-to-end `infer_bench` timings.
- `--` means that phase was not measured for that row, not that the effect is
  zero.
- The HCA prefill cost is an implementation/prefill-load artifact at short
  context; compression pays off in decode and becomes more favourable at
  longer context, where sparsity is layered on top.
- **Low-bit weights/KV are rejected by decision.** An int8 QAT experiment was
  slower than fp32 on this platform (the Go `simd` package has no vectorized
  int8->float32 conversion, and decode is compute-bound here), and a
  storage-only quantized KV cache is out of scope. The backend is float32-only;
  see [06-numeric-precision.md](06-numeric-precision.md) for the locked-in
  decision record.
