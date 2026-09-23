# gonano — DeepSeek-V4.1-Flash Optimizations

gonano started as a pure-Go reimplementation of nanochat's GPT transformer.
On top of that base it implements the long-context attention design and the
KV-cache optimizations described in
[DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969).

This guide is the map between the paper and the code: which concept is
implemented where, how to turn it on, and which parts are deliberately *not*
implemented. It assumes you have read
[00-architecture-overview.md](00-architecture-overview.md) for the package
layout.

> **Setup.** Run everything from the repo root with `GOEXPERIMENT=simd`, as
> described in [00-quickstart.md](00-quickstart.md). The numeric backend is
> **float32** everywhere; there is no low-bit path (see §8).

---

## 1. Concept map

| Paper concept | § | Implemented in | Enable with |
|---|---|---|---|
| Causal Encoder-Decoder (CED) | 2.2 | `model/transformer.go` `PrefillCED`, `model/train_transformer.go`, `model/train.go` | `--ced` |
| Decoder SWA bounded replay | 2.2, 3.2.2 | `PrefillCED` phase 3 | implied by `--ced --swa-window` |
| HCA dense KV compression | 2.3 | `model/compress.go` `ChannelCompressor` | `--compression-ratio N` |
| CSA sparse attention | 2.3 | `model/indexer.go` `SparseIndexer` | `--sparse-topk K` |
| Cross-layer KV + index reuse (Full/Reindex/Reuse) | 2.3.1 | `model/config.go` `ReusePattern`, `model/share.go`, `forwardCompressedSparse`, `sparseTrainingPlan` | `--reuse-pattern FRUU` |
| Hierarchical sparse indexer | 2.3.2 | `model/indexer.go` `SelectProjected`, shared candidate pool | `--indexer-pool P` |
| Local sliding-window branch (SWA) | 2.2 | `model/compress_attention.go` `mergeAttentionBranches` | `--swa-window W` |
| Grouped-query attention | 2.1 | `model/attention.go`, `Config.NumKVHead` | `--kv-head-ratio R` |
| Partial rotary embedding | 2.1 | `model/rotary.go`, `Config.RotaryDims` | `Config.RotaryDims` (default 64) |
| Head-wise Muon for Q/K | 2.5 | `model/optimizer.go` `headWiseViews` | `--head-wise-muon` |
| Sinkhorn-balanced embeddings / lm_head | 2.5 | `optimizer/sinkhorn.go`, `model/optimizer.go` `sinkhornGroup` | `--sinkhorn-embeddings` |
| Full-vocabulary on-policy distillation (OPD) | 5.2.4 | `tensors/loss.go` (`DistillationLossPerPosition`), `trainer/distill.go` | `cmd/chat_opd` |
| MLA low-rank query / KV latent | 2.3, 4.2.1 | `model/attention.go` `projectQuery`/`projectKeyValue` | `--query-compression-dim`, `--kv-latent-dim` |
| Global-KV prefix reuse + SWA replay | 3.2.1, 3.2.2 | `inference/prefix.go` `PrefixCache`, `Engine.Prefix` | `Engine.Prefix = NewPrefixCache(...)` |
| FP4 main KV cache / FP4 indexer QAT | 2.4.4 | **not implemented** (float32-only) | — |
| Engram, MoE, DSpark, Single-Pass mHC | 2.1, 2.4 | **not applicable / not implemented** | — |

---

## 2. Architecture

### 2.1 Causal Encoder-Decoder (CED)

The paper splits the 40-layer backbone into a 20-layer causal **encoder** and a
20-layer **decoder**; the decoder's *global* keys/values are projected from the
encoder's final hidden state `H_{L/2}` instead of from each decoder layer's own
hidden state, so prefill drops from `O(N·L)` to roughly `O(N·L/2)`
(paper §2.2).

In gonano:

- `Config.CED` enables the split; `Config.CEDSplit()` returns `NumLayer/2`.
- Training (`model/train_transformer.go`) captures `encoderHidden` at the split
  and hands it to each decoder group through `compressionShare.encoderHidden`.
  `model/train.go` (`cedGlobalKeyValue`, `cedKeyProjectionBackward`,
  `cedValueProjectionBackward`) routes the global K/V gradients back into the
  encoder hidden state; `TrainBackward` injects the accumulated
  `share.gradientEncoder` at the encoder/decoder boundary.
- Inference (`model/transformer.go` `PrefillCED`) runs in three phases:
  1. the encoder over the full prompt;
  2. fill every decoder layer's global compressed cache (and Reindex layers'
     indexer-key caches) from the encoder hidden state;
  3. **decoder bounded replay** over only the last `SWAWindow` tokens to rebuild
     the decoder's local sliding-window state. The last prompt position's logits
     are exact when the window covers the replay, and decode is unchanged
     because every layer still runs per token.
- The reuse pattern (`--reuse-pattern`) restarts at the decoder split
  (`Config.reuseHalfStart`), so a decoder group never borrows an encoder
  group's compressed KV.
- `Config.Validate` requires `--compression-ratio > 1` and `--swa-window > 0`
  for CED, and `NumLayer >= 2`.

The paper's **Encoder SWA Bounded Replay** (recomputing only `n_win` tokens on a
global-KV cache hit, §3.2.2) is the serving-side counterpart; gonano exposes the
building block through the prefix cache (§6).

### 2.2 Dense KV compression (HCA) and sparse attention (CSA)

Every compressed layer owns a `ChannelCompressor` (`model/compress.go`). It maps
the hidden state to per-channel logits, adds a learnable per-position bias
`bias [ratio, channels]`, and merges each block of `ratio` consecutive rows into
one compressed entry with a per-channel softmax over the block. This is the
dense HCA path.

When `--sparse-topk K` is set, each query instead attends to the top-`K`
compressed blocks selected by a **lightning indexer** (`SparseIndexer`,
`model/indexer.go`):

```
I[t, s] = sum_h w_h * ReLU(dot(q_h[t], k_h[s]))
```

The indexer is trained with a distillation loss (`distillationTarget`,
`indexerDistillationGradient` in `model/compress_attention.go`, scaled by
`Config.IndexerLossWeight`); it is deliberately low-rank (`IndexerDim` per
head) so scoring compressed blocks is much cheaper than full attention. Indexer
K is projected from the compressed main KV, matching the CSA2 simplification in
paper §2.3 (no separate compression path, no absolute positional embedding).

### 2.3 Cross-layer KV and index reuse (Full / Reindex / Reuse)

Paper §2.3.1 assigns each layer one of three modes:

- **Full** — owns the compressor and indexer; produces compressed KV, indexer K,
  and (when sparse) the top-k selection.
- **Reindex** — reuses the most recent group's compressed KV and indexer K, but
  runs its own indexer for a fresh selection.
- **Reuse** — reuses both the compressed KV and the latest published selection;
  runs no indexer at all.

In gonano this is the cyclic `Config.ReusePattern` (`F`/`R`/`U`, must start
with `F`), resolved by `ReuseModeAt`, `ReuseProducer`, `OwnsCompressed`, and
`OwnsIndexer`. `model/share.go` carries the per-group state. The inference path
is `forwardCompressedSparse`; the training path is `sparseTrainingPlan`. This is
what actually reduces cache storage (only Full layers allocate compressed KV) as
well as indexer work.

### 2.4 Hierarchical sparse indexer

Paper §2.3.2 narrows the search domain of deeper indexers: the first Full-mode
decoder layer scores all visible positions and builds a **candidate pool**, and
later Reindex layers score only that pool, making their per-query cost bounded
rather than linear in context length.

gonano's coarse-to-fine selection lives in `SparseIndexer`:

- `Config.IndexerPool` (`--indexer-pool P`) groups compressed entries into
  super-blocks; the query is scored against pooled key representations first.
- `Config.IndexerCandidates` (`--indexer-candidates`, default
  `IndexerCandidateBudget()` = `max(8*SparseTopK, 64)`) bounds how many entries
  are fully scored.
- `SelectProjectedWithCandidates` returns both the top-k selection and the
  coarse candidate pool; the pool is published in `compressionShare.candidatePool`
  and consumed by later Reindex layers through `SelectWithinPool`. This is the
  "constant-cost deeper indexer" property from the paper.

The **reindex path also uses the cached indexer-key projections**
(`KVBuffer.IndexerKey`), so decode does not re-project every compressed key on
each step.

### 2.5 Local sliding-window attention (SWA)

`--swa-window W` adds a layer-local branch: each compressed query attends to the
global compressed blocks **and** the raw keys/values of the preceding `W` tokens.
Both branches share one exact softmax:

- `mergeAttentionBranches` (training) and `mergeAttentionRow` (inference)
  combine the branch log-sum-exps; the backward pass shares the merged
  log-sum-exp and the merged row correction `dO·O` (`rowCorrection`).

Because the window is bounded by `W`, the local branch's cost is independent of
context length — the key property for long context. SWA KV is layer-local: every
reuse mode computes its own, independently of the shared global compressed KV.

### 2.6 GQA and partial RoPE

`Config.NumKVHead` (`--kv-head-ratio`) gives one key/value head per `R` query
heads, cutting KV size and traffic. `Config.RotaryDims` (default 64) applies
rotary embedding only to the trailing head dimensions, matching DeepSeek's
partial RoPE (`model/rotary.go`).

---

## 3. Shared candidate pool (constant-cost deeper indexer)

This is the runtime complement to §2.4. Before it, every Full/Reindex layer
re-pooled and coarse-scored the whole visible context on every decode step. Now:

1. A Full layer computes the candidate pool once per step and stores it in
   `compressionShare.candidatePool` (`model/compress_attention.go`).
2. Reindex layers call `SparseIndexer.SelectWithinPool`, which skips the coarse
   pass and only fine-scores the published candidates.
3. The same restriction is applied during training (`sparseTrainingPlan`) so the
   model is optimized under the search domain it uses at inference, as the
   paper requires.

Units: `TestSelectWithinPoolMatchesFineStage`, `TestHierarchicalPoolBoundedByBudget`,
`TestHierarchicalPoolCoversAllWhenUnbounded`, `TestHierarchicalPoolTrainStepFinite`,
`TestHierarchicalPoolInferenceFinite`.

---

## 4. SIMD, parallel, and bounded top-k kernels

The paper accelerates indexing with FP4 quantization of indexer queries/keys.
gonano cannot use low-bit arithmetic (float32-only; see §8), so it accelerates
the same work with vectorization and parallelism instead:

- **`kernels.Indexer`** (new backend contract, `kernels/backend.go`):
  - `IndexerScores(destination, query, key, weight, rows, blocks, heads, dim)`
    computes the ReLU-weighted multi-head score matrix with a vectorized dot
    product, parallelized across query rows (`kernels/simd/indexer.go`,
    `kernels/scalar/indexer.go`).
  - `PooledMean(destination, source, blockCount, groupSize, width)` computes the
    coarse super-block means with SIMD.
- **Bounded top-k** (`topKIndices` in `model/indexer.go`) replaces the full
  `sort.SliceStable` in `TopK`, `SelectBlocks`, and the fine selection with a
  size-`k` heap when `k` is small, preserving the exact tie-break (larger score,
  then smaller index).
- The coarse scoring uses the batched kernel for whole-sequence queries and the
  scalar loop for single-token decode (below `indexerKernelCoarseMinRows`),
  where per-row goroutine overhead would dominate.

Units: `kernels/simd/parity_test.go` (`TestIndexerScoresParity`,
`TestPooledMeanParity`), `model/indexer_test.go` (`TestTopKIndicesMatchesFullSort`,
`TestTopKIndicesTieBreak`).

---

## 5. MLA low-rank query and KV latent

Paper §2.3 / §4.2.1 uses a query-compression dimension and a shared KV latent
(MLA). gonano implements the low-rank factorizations, disabled by default so the
existing full-rank path and checkpoints are unchanged:

- `Config.QueryCompressionDim` (`--query-compression-dim R`): the query is
  down-projected to `R` and up-projected per head. Weights
  `transformer.h.N.attn.q_down.weight` plus the usual `c_q`.
- `Config.KVLatentDim` (`--kv-latent-dim R`): key and value are up-projected
  from a **single shared latent** of width `R`. Weights `kv_down` plus `c_k`/`c_v`.

Forward and backward are fully wired (`CausalSelfAttention.projectQuery` /
`projectKeyValue`, and the CED counterparts `cedKeyProjectionBackward` /
`cedValueProjectionBackward`). It reduces projection parameters, matmul FLOPs,
and per-token weight bytes.

**Scope note:** the KV cache still stores the full per-head keys/values. The
low-rank latent reduces **compute and parameters, not cache bytes**; a latent KV
cache is future work. Also note the low-rank path is a genuine architectural
change and must be trained from scratch.

Units: `TestBackpropDirectionalGradientCheckLowRank` (query, kv, both),
`TestBackpropPerElementLowRankCED`, `TestLowRankReducesParameters`,
`TestSaveLoadRoundtripLowRank`, `TestForwardLowRankDecodePath`.

---

## 6. KV prefix cache and SWA bounded replay

For multi-turn serving, `inference/prefix.go` adds an in-memory, single-entry
prefix cache:

- `PrefixCache.Store(tokens, kv)` snapshots the KV buffer after a prefill
  (`KVBuffer.Clone` deep-copies the raw, compressed, indexer, and tail state).
- `PrefixCache.Lookup(tokens)` returns a clone plus the matched prefix length
  only when the stored tokens are a **strict prefix** of the new request, so at
  least one token remains to be prefilled and logits are always available.
- `Engine.Prefix` wires it into `Generate`: a hit reuses the cached global and
  local KV and prefills only the suffix; a miss falls back to a full prefill and
  replaces the snapshot. When enabled, the prefill buffer is sized to
  `Config.SequenceLen` so a later request can extend it in place.

This is the in-memory analogue of the paper's Encoder SWA Bounded Replay: the
cache is only valid while it exactly matches the request. It is intentionally
**disabled for CED**, whose decoder global KV depends on the full encoder hidden
state, and it is not a persistent SSD/DRAM tier.

Units: `TestPrefixCacheLookupStore`, `TestPrefixCacheGenerationMatchesFullPrefill`,
`TestPrefixCacheMissFallsBack`, `TestPrefixCacheCloneIndependence`.

---

## 7. Head-wise Muon for Q and K

Paper §2.5 splits the query (and key) weights by attention head before the Muon
update, giving each head its own preconditioner. In gonano:

- `Config.HeadWiseMuon` (`--head-wise-muon`) switches it on.
- `SetupOptimizer` (`model/optimizer.go`) replaces the whole Q/K weights with
  `headWiseViews`, one `[headDim, columns]` view per head. The views share the
  parent's `Data` and `Grad` backing arrays, so the Muon update and the
  backpropagated gradient operate in place on the same storage.
- It is training-only and adds no inference cost; `InitWeights`, `NamedParameters`,
  `SetupOptimizer`, `MatmulParams`, and the FLOP accounting all handle it.

Units: `TestHeadWiseViewsShareStorage`,
`TestSetupOptimizerHeadWiseSplitsQueryKey`, `TestHeadWiseMuonTrainStepFinite`.

**Sinkhorn-balanced embeddings and prediction head.** `--sinkhorn-embeddings`
routes the token embedding, `lm_head`, and value embeddings to the
Sinkhorn-balanced momentum update instead of AdamW (Algorithm 1 of the paper):
Nesterov momentum, masking of near-zero rows, `K=11` alternating row/column L2
normalizations, a `√n` unit-RMS rescale, and the `γ=0.18` learning-rate
correction, with no weight decay. Because it keeps only a momentum buffer, it
halves the optimizer-state memory of those large `[vocab, dim]` matrices and
matches the paper's choice for embedding tables and the prediction head. It is
training-only; the checkpoint layout is unchanged.

Units: `TestSinkhornSingleRowNormalization`, `TestSinkhornMasksNearZeroRows`,
`TestSinkhornRowsHaveUnitRMS`, `TestSinkhornStateAllocatedAndFinite`,
`TestSetupOptimizerSinkhornRoutesEmbeddings`, `TestSinkhornTrainStepFinite`.

**Full-vocabulary on-policy distillation.** `cmd/chat_opd` runs the paper's
final post-training stage (§5.2.4): the student generates rollouts, a frozen
teacher checkpoint scores them, and the student is trained to match the teacher's
full next-token distribution. The objective is the forward KL from the student
to the teacher (equivalently the cross-entropy of the teacher distribution under
the student), computed by `tensors.DistillationLossPerPosition` /
`tensors.DistillationGrad` with an optional softmax temperature and a loss mask
that supervises only the sampled completion tokens. `trainer.DistillStep` is the
per-micro-batch update and `trainer.Distill` the loop; the teacher can be any
checkpoint whose vocabulary matches the student's.

Units: `TestDistillationMatchesCrossEntropyForOneHotTeacher`,
`TestDistillationGradMatchesNumeric`, `TestDistillationSelfIsZeroGrad`,
`TestDistillationMaskZeroesPositions`, `TestDistillationTemperatureSoftensGrad`,
`TestDistillStepReducesLoss`, `TestDistillSelfZeroGradient`,
`TestDistillMaskRestrictsGradient`, `TestDistillLoop`.

---

## 8. Low-bit weights and KV were rejected

The paper's headline KV win is FP4 for the main KV cache and FP4 QAT for indexer
queries/keys (paper §2.4.4). gonano does **not** do this. An int8 experiment was
slower than float32 on this platform because the Go `simd` package has no
vectorized int8→float32 conversion and decode is compute-bound here; the backend
is therefore float32-only. The float32-compatible substitutes for the paper's
low-bit indexer acceleration are the vectorized kernels in §4 and the low-rank
factorizations in §5. Do not expect per-token KV bytes to match the paper's
890 bytes/token.

---

## 9. Configuration reference

### 9.1 Flags (`cmd/base_train`, `cmd/trainer`)

| Flag | Default | Meaning |
|---|---|---|
| `--compression-ratio N` | 0 | HCA dense KV compression; each `N` rows become one entry. Requires `N > 1` to activate. |
| `--sparse-topk K` | 0 | CSA top-k compressed blocks per query; requires compression. |
| `--indexer-dim D` | 64 | Lightning-indexer per-head dimension. |
| `--indexer-pool P` | 0 | Hierarchical indexer super-block size; `>1` enables coarse-to-fine. |
| `--indexer-candidates C` | 0 | Candidate budget; `0` = `max(8*K, 64)`. |
| `--reuse-pattern F/R/U` | "" | Cross-layer reuse cycle; must start with `F`. |
| `--swa-window W` | 0 | Local sliding-window width on compressed layers. |
| `--ced` | false | Causal encoder-decoder split; requires compression and SWA. |
| `--head-wise-muon` | false | Split Q/K by head for Muon (training-only). |
| `--sinkhorn-embeddings` | false | Sinkhorn-balanced update for embedding/lm_head/value embeds (training-only). |
| `--query-compression-dim R` | 0 | Low-rank query bottleneck width. |
| `--kv-latent-dim R` | 0 | Shared low-rank KV latent width. |
| `--kv-head-ratio R` | 1 | Query heads per KV head (GQA). |

### 9.2 Config fields (`model.Config`, persisted in the checkpoint)

`RotaryDims`, `IndexerHeads`, `IndexerLossWeight` exist as fields but are not
exposed as CLI flags; set them when building a `Config` directly. `Config.Validate`
enforces the cross-field rules (sparse requires compression, CED requires
compression + SWA, the reuse pattern must start with `F`, etc.).

### 9.3 Recommended long-context configuration

```
--compression-ratio 4 --sparse-topk 8 --indexer-pool 8 --reuse-pattern FRU \
--swa-window 128 --ced
```

Add `--query-compression-dim 64 --kv-latent-dim 64` for the low-rank variant
(requires training from scratch). Add `--swa-window`/`--ced` only when local
fidelity matters more than the last few percent of decode throughput; a
decode-only short-prompt workload is often faster with plain compression +
sparsity.

---

## 10. Benchmarking and verification

**Reproduce a long-context run.** `benchmark_longctx.sh` trains a tiny model
with a given architecture config and sweeps prefill/decode at one or more
sequence lengths:

```bash
SEQS=4096,16384 BATCH_SIZES=1,16,64 DECODE_TOKENS=16 ./benchmark_longctx.sh

CONFIG="--compression-ratio 4 --sparse-topk 8 --indexer-pool 8 \
        --reuse-pattern FRU --swa-window 128 --ced" ./benchmark_longctx.sh
```

Set `STEPS=0` (the default) for a random-weight latency/throughput measurement;
set `STEPS>0` to train a few steps first. `cmd/infer_bench` prints TTFT, per-step
latency, decode tok/s, and weight/KV bytes read.

**Unit tests.** Every feature has one, and scalar-vs-SIMD parity covers the
kernel contract:

```bash
GOEXPERIMENT=simd go test -race ./...
```

| Area | Tests |
|---|---|
| Compression / CED / SWA | `model/compress_test.go`, `model/ced_test.go`, `model/swa_test.go` |
| Indexer / hierarchy / reuse | `model/indexer_test.go`, `model/model_test.go` |
| Backprop (incl. low-rank, CED) | `model/gradcheck_test.go` |
| Head-wise Muon | `model/optimizer_test.go` |
| Checkpoint round-trip | `model/checkpoint/checkpoint_test.go` |
| Inference, CED, prefix cache | `inference/ced_test.go`, `inference/prefix_test.go` |
| Kernel parity | `kernels/simd/parity_test.go` |

---

## 11. Not implemented

For completeness, the paper components that are out of scope here:

- **FP4 main KV cache and FP4 indexer QAT** (see §8).
- **MoE backbone, Engram conditional memory, DSpark speculative decoding, and
  Single-Pass mHC.** gonano is a dense single-residual-stream model; these are
  architectural/system components of the 552B model and do not map onto it.
- **Latent KV cache** — the low-rank KV latent reduces compute, not cache bytes.
- **Persistent KV tier (SSD/host DRAM), EPD disaggregation, and the GPU kernel
  fusions** (Mega-* kernels, FlashMLA): single-process CPU serving only.

---

## 12. Quick index

| Looking for | Start here |
|---|---|
| CED prefill + bounded replay | `model/transformer.go` (`PrefillCED`) |
| Dense compressor | `model/compress.go` (`ChannelCompressor`) |
| Sparse indexer + hierarchy | `model/indexer.go` (`SparseIndexer`) |
| Reuse modes | `model/config.go` (`ReusePattern`), `model/share.go` |
| Compressed attention forward/backward | `model/compress_attention.go` |
| Attention forward/backward (training) | `model/train.go` |
| Head-wise Muon | `model/optimizer.go` (`headWiseViews`) |
| Prefix cache | `inference/prefix.go` |
| Indexer kernels | `kernels/simd/indexer.go`, `kernels/scalar/indexer.go` |
| Long-context benchmark | `benchmark_longctx.sh`, `cmd/infer_bench` |
