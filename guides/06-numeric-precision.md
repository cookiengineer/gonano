# gonano — Numeric Precision and Low-Bit Decisions

This document is the authoritative record of gonano's numeric-precision
requirements and the decisions that have been locked in about quantization. It
exists so the reasoning is not re-litigated from scratch every time a paper (for
example [DeepSeek-V4.1-Flash](05-deepseek-v4.1-optimizations.md)) proposes a
low-bit optimization.

Read it together with
[00-architecture-overview.md](00-architecture-overview.md) (the package layout)
and [05-deepseek-v4.1-optimizations.md](05-deepseek-v4.1-optimizations.md) (the
paper-to-code map).

---

## 1. Hard requirement: float32 for vector ops

gonano's production compute path is the SIMD kernel backend, and that backend is
**float32 only**.

1. `kernels/simd` is the default `kernels.Backend`. Every vector op it
   implements — elementwise arithmetic, reductions, matmul, fused
   softmax/RMSNorm, flash attention, the indexer, and the MoE kernels — operates
   on `[]float32`.
2. `tensors.Tensor` stores `Data` and `Grad` as `float32`; `tensors.Int32s`
   exists only for token ids and masks, never for arithmetic.
3. Checkpoints (`.gn`) and GGUF export/import store `float32` blobs
   (`model/checkpoint`).
4. `kernels/scalar` is a portable pure-Go reference implementation. It is used
   for scalar-vs-SIMD parity tests and for hosts without SIMD; it is **not** a
   low-bit fallback.
5. The environment is the Go 1.27 standard-library `simd` package on AVX-512.
   That package exposes no vectorized int8/fp4 → float32 conversion, and the
   `internal/parallel` goroutine pool parallelizes float32 work.

Because the SIMD kernel is a hard requirement, "quantize, then cast back to
float32 for the vector op" is the only shape a low-bit path could take here.
That pattern is exactly what was measured and rejected (see §2.1).

---

## 2. Locked-in decisions

### 2.1 Quantization-aware training (QAT) is rejected

The DeepSeek-V4.1-Flash paper uses QAT to accelerate the indexer with FP4
indexer queries/keys (paper §2.4.4). gonano implemented an int8 QAT experiment
and it gave **no performance improvement**, because the quantized values were
cast back to float32 before the SIMD vector op. Decode is compute-bound on this
platform, not bandwidth-bound, so removing bytes did not remove the dominant
cost.

**Decision: gonano does not implement QAT.** The float32-compatible substitutes
for low-bit indexer acceleration are the vectorized kernels
(`kernels/simd/indexer.go`, `kernels/simd/moe.go`) and the low-rank
factorizations (`--query-compression-dim`, `--kv-latent-dim`, absorbed MLA).

### 2.2 A quantized KV cache is not in scope

The paper's headline KV win is an FP4 **main KV cache** (paper §2.4.4). The paper
is explicit that this reduces *storage*, not matrix-multiply speed:
dequantizing the cached values before attention is what allows a more accurate
format without native low-bit matmul support. In principle that is compatible
with a float32 SIMD kernel — store the cache in int8/FP4 with scales, dequantize
to float32 in the attention read path, then run the existing float32 kernels.

That storage-only design was considered and is **explicitly excluded from
scope**. The backend and cache remain fully float32.

**Decision: gonano does not implement a quantized KV cache.** It is not a TODO
and should not be reintroduced without first revisiting this record and
re-benchmarking (§3).

### 2.3 Consequences

- Per-token KV bytes will **not** match the paper's 890 bytes/token. gonano's
  reductions come from architecture, not precision: GQA, HCA compression,
  cross-layer KV/index reuse, the local sliding window, and absorbed MLA.
- `Transformer.KVBytesPerToken` / `KVReadBytes` report the float32 footprint.
- `infer_bench` weight/KV byte counts are float32 byte counts.

---

## 3. Guidance for future work

1. Do **not** propose QAT or a quantized KV cache as a performance or memory
   optimization. It has been evaluated and locked out.
2. If the decision is ever revisited, the proposal must:
   - keep the SIMD kernel and all vector arithmetic float32;
   - be a measured change (long-context `infer_bench` / `benchmark_longctx.sh`
     numbers, not an assumption);
   - keep `GOEXPERIMENT=simd go test -race ./...` and the scalar-vs-SIMD parity
     tests green.
3. The approved float32-compatible levers for the paper's low-bit goals are the
   vectorized indexer/MoE/matmul kernels (§4 of the optimizations guide) and the
   low-rank factorizations plus MLA (§5 of that guide).
