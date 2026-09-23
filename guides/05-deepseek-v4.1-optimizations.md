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

The whole stack below is **on by default** through the `flash` preset; the
`latent` and `dense` presets are alternates. The "Enable with" column names the
preset (or runtime API) that turns each concept on.

| Paper concept | § | Implemented in | Enable with |
|---|---|---|---|
| Causal Encoder-Decoder (CED) | 2.2 | `model/transformer.go` `PrefillCED`, `model/train_transformer.go`, `model/train.go` | `flash` |
| Decoder SWA bounded replay | 2.2, 3.2.2 | `PrefillCED` phase 3 | `flash` |
| HCA dense KV compression | 2.3 | `model/compress.go` `ChannelCompressor` | `flash` |
| CSA sparse attention | 2.3 | `model/indexer.go` `SparseIndexer` | `flash` |
| Cross-layer KV + index reuse (Full/Reindex/Reuse) | 2.3.1 | `model/config.go` `ReusePattern`, `model/share.go`, `forwardCompressedSparse`, `sparseTrainingPlan` | `flash` |
| Hierarchical sparse indexer | 2.3.2 | `model/indexer.go` `SelectProjectedWithCandidates`, block-max candidate pool | `flash` |
| Local sliding-window branch (SWA) | 2.2 | `model/compress_attention.go` `mergeAttentionBranches` | `flash` |
| Grouped-query attention | 2.1 | `model/attention.go`, `Config.NumKVHead` | `flash` / `latent` |
| Partial rotary embedding | 2.1 | `model/rotary.go`, `Config.RotaryDims` | default 64 |
| Mixture-of-Experts (DeepSeekMoE) | 2.1, 4.2.1 | `model/moe.go`, `model/mlp.go` | `flash` / `latent` |
| Clamped SwiGLU experts | 4.2.1 | `kernels/backend.go` (`SwiGLU`), `tensors.SwiGLU` | MoE (default on) |
| Auxiliary-loss-free load balancing | 2.1.1, 4.2.2 | `MoE.updateRouterBias`, `Transformer.UpdateRouterBias` | MoE (default on) |
| Head-wise Muon for Q/K | 2.5 | `model/optimizer.go` `headWiseViews` | `flash` |
| Sinkhorn-balanced embeddings / lm_head | 2.5 | `optimizer/sinkhorn.go`, `model/optimizer.go` `sinkhornGroup` | all presets except `dense` |
| Full-vocabulary on-policy distillation (OPD) | 5.2.4 | `tensors/loss.go` (`DistillationLossPerPosition`), `trainer/distill.go` | `cmd/chat_opd` |
| DSpark semi-autoregressive drafting + survival scheduler | 2.4.3 | `model/dspark.go` (`DSpark`), `inference/speculative.go` | `--drafter` + `--speculative` |
| MLA low-rank query / KV latent | 2.3, 4.2.1 | `model/attention.go` `projectQuery`/`projectKeyValue` | `model.Config` fields |
| Absorbed MLA latent cache | 2.3, 4.2.1 | `model/mla.go`, `KVBuffer.EnableMLA` | `latent` |
| Global-KV prefix reuse + SWA replay | 3.2.1, 3.2.2 | `inference/prefix.go` `PrefixCache`, `Engine.Prefix` | `Engine.Prefix = NewPrefixCache(...)` |
| Persistent multi-entry KV cache (LRU/TTL/disk) | 3.2.1 | `inference/cache.go` `CacheManager`, `model/kvcache_codec.go` | `Engine.Cache = NewCacheManager(...)` |
| SWA pool + bounded replay (global-only persistence) | 3.2.1, 3.2.2 | `KVBuffer.StripRaw`, `Transformer.ReplaySWA`, `Engine.SWACache` | `CacheOptions.StripSWA`, `Engine.SWACache` |
| FP4 main KV cache / FP4 indexer QAT | 2.4.4 | **not implemented** (float32-only) | — |
| Engram, Single-Pass mHC | 2.1, 2.4 | **not applicable / not implemented** | — |

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
  super-blocks; each super-block's score is the **maximum** index score among its
  entries (`kernels.Backend.IndexerBlockMax`), and the best super-blocks are
  selected, exactly as in the paper.
- `Config.IndexerCandidates` (`--indexer-candidates`, default
  `IndexerCandidateBudget()` = `max(8*SparseTopK, 64)`) bounds how many entries
  are fully scored.
- `SelectProjectedWithCandidates` returns both the top-k selection and the
  coarse candidate pool; the pool is published in `compressionShare.candidatePool`
  and consumed by later Reindex layers through `SelectWithinPool`. This is the
  "constant-cost deeper indexer" property from the paper.
- In training (`sparseTrainingPlan`) a Reindex layer's indexer distillation is
  restricted to the shared candidate pool (its scores are masked outside the
  pool and the target is renormalized within it), so deeper indexers are
  optimized under the same search domain they use at inference.

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

### 2.7 Mixture-of-Experts (DeepSeekMoE)

Paper §2.1 and §4.2.1 use one shared expert plus many fine-grained routed
experts in every feed-forward layer. gonano implements this as `model/moe.go`,
enabled with `--moe`:

- **Structure.** Every block keeps an always-active **shared expert** (`MLP` in
  SwiGLU mode, owned by `block.mlp`) and adds a routed `MoE` (`block.moe`). Each
  token selects its top-`k` routed experts by router affinity; the routed output
  is the affinity-weighted sum of their SwiGLU outputs. There are no biases.
- **Activation.** Experts and the shared expert use clamped SwiGLU,
  `silu(min(gate,10)) * clamp(up,-10,10)` (`kernels.Backend.SwiGLU`,
  `tensors.SwiGLU`). The clamp threshold is `Config.MoEClamp` (default 10; a
  negative value disables clamping). Plain (non-MoE) blocks keep the ReLU² MLP.
- **Weight layout.** Routed expert weights are packed contiguously as
  `[E, Hidden, Dim]` (gate/up) and `[E, Dim, Hidden]` (down) so a future grouped
  SIMD kernel can walk them without pointer chasing. The optimizer consumes
  per-expert rank-2 views (`expertViews`) that share the parent's `Data`/`Grad`,
  exactly like head-wise Muon.
- **Routing.** `router.weight` maps the hidden state to `E` affinities;
  softmax gives the output weights while a non-trainable correction bias,
  `router.bias`, participates only in expert *selection*
  (`gate_scores = logits + bias`). Selection reuses the bounded top-k heap from
  the lightning indexer.
- **Fused kernels.** The `kernels.MoE` contract fuses the hot paths, in the
  spirit of the paper's "Mega-Gate"/"Mega-MoE": `MoEGateTopK` computes the
  router softmax (vectorized) and the bias-aware top-k selection in one call,
  backed by `TopKIndices`, a dedicated selection kernel that finds the maximum
  with a vectorized reduction and its first index with a vectorized equality
  mask (the DeepSelect-style TopK), giving ~7x over the scalar scan at hundreds
  of experts. `GroupedMatMulTransposed` runs one GEMM over all experts from an
  expert-major gathered input (vectorized dot products, parallelized across
  rows), replacing one `MatMulTransposed` per expert. `MoE.forward` gathers the
  selected rows, runs the three grouped GEMMs (gate/up/down) with a fused SwiGLU
  between them, and scatter-adds the affinity-weighted outputs. Training keeps
  per-expert rank-2 views into the grouped buffers for the linear-layer
  backward. Both kernels have scalar reference implementations
  (`kernels/scalar/moe.go`) and SIMD parity tests.
- **Load balancing.** After each optimizer step,
  `Transformer.UpdateRouterBias` applies the auxiliary-loss-free update
  `bias_e += u·sign(mean_load − load_e)` with `u = Config.RouterBiasUpdate`
  (default 0.001). Overloaded experts get a lower selection bias. The sequence-
  level balance loss is not implemented.
- **Accounting.** `MatmulParams` counts every expert (total parameters), while
  `ActiveMatmulParams` counts only the top-`k` experts plus shared/router; the
  decode FLOPs and `WeightReadBytes` (and therefore `cmd/infer_bench`) use the
  active count, since only those weights are read per token.
- **Sizing from depth.** With `--moe`, the routed expert count is
  `model.MoEExpertsForDepth(depth) = max(8, 2·depth)`; `--num-experts`,
  `--experts-per-token` (default 2), and `--expert-hidden-dim` (default the
  embedding width) override the derivation. MoE is orthogonal to the attention
  design, so it composes with GQA, compression, SWA, CED, and MLA (all covered
  by tests). The scaling-law path `trainer.DeriveHyperparamsForConfig` uses
  `ScalingParamsForConfig` for the target-token count (all experts) and
  `EstimateFlopsPerTokenForConfig` for the per-token FLOPs (active experts), so
  a MoE config gets a horizon that reflects its total parameter count.
- **Deliberately omitted.** The paper's small sequence-level balance loss
  (§4.2.2, weight 0.0001) is not implemented; load balancing is purely the
  auxiliary-loss-free bias update. The grouped GEMM is fused for the forward
  pass only; the backward keeps the per-expert linear-layer path.
- **Speculative decoding.** DSpark (and the plain drafter) compose with a MoE
  backbone: `DrafterConfig` keeps only the vocabulary and head geometry, so the
  DSpark trunk is a dense small transformer regardless of the target, while
  verification runs through the MoE target and trunk distillation uses the MoE
  teacher's cache-less forward. Exact greedy speculative decoding is covered by
  `TestSpeculativeDSparkOnMoEBackboneMatchesGreedy` and
  `TestSpeculativeDraftOnMoEBackboneMatchesGreedy`; distillation from a MoE
  teacher by `TestDistillWithMoETeacher`.

Units: `model/moe_test.go` (`TestMoEConfigDefaults`, `TestMoEConfigValidation`,
`TestMoEForwardFiniteAndShapes`, `TestMoENamedParameters`,
`TestMoEActiveParameterAccounting`, `TestMoERouterSelectsDeterministically`,
`TestMoERouterBiasBalancing`, `TestMoEForwardMatchesTrainForward`,
`TestMoEDecodeMatchesPrefill`, `TestMoEGradientCheck`,
`TestMoEPerElementGradientCheck`, `TestMoEConfigAccounting`,
`TestMoEComposesWithGQA`, `TestMoEComposesWithCompression`,
`TestMoEComposesWithCED`, `TestMoEComposesWithMLA`,
`TestMoEScalarBackendMatchesSIMD`);
`kernels/scalar` `TestSwiGLU`, `TestMoEGateTopK`, `TestTopKIndices`,
`TestGroupedMatMulTransposed`; `kernels/simd` `TestElementwiseParity` (SwiGLU),
`TestMoEGateTopKParity`, `TestTopKIndicesParity`,
`TestGroupedMatMulTransposedParity`; `tensors` `TestSwiGLUForwardClamp`,
`TestSwiGLUBackwardNumeric`; `model/checkpoint` (`TestSaveLoadRoundtripMoE`,
`TestExportGGUFMoE`, `TestGGUFWithoutMoEMetadataStillLoads`);
`trainer.TestTrainerMoEOverfitsTiny`, `trainer.TestDeriveHyperparamsForMoEConfig`;
`model.TestDSparkOnMoEBackbone`; `inference`
(`TestSpeculativeDSparkOnMoEBackboneMatchesGreedy`,
`TestSpeculativeDraftOnMoEBackboneMatchesGreedy`,
`TestDSparkDrafterDenseForMoEBackbone`); root `TestEndToEndMoE`.

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
  - `IndexerBlockMax(destination, scores, rows, blocks, groupSize)` reduces the
    score matrix to per-super-block maxima with SIMD; it replaced the earlier
    pooled-mean coarse stage so the hierarchical indexer matches the paper's
    block score (DeepSeek-V4.1 §2.3.2).
- The coarse stage materializes per-entry scores in bounded row chunks
  (`selectProjectedScores`), so peak scratch memory is independent of context
  length while `IndexerBlockMax` still handles the reduction.
- **Bounded top-k** (`topKIndices` in `model/indexer.go`) replaces the full
  `sort.SliceStable` in `TopK`, `SelectBlocks`, and the fine selection with a
  size-`k` heap when `k` is small, preserving the exact tie-break (larger score,
  then smaller index).
- The batched scoring kernel is used for whole-sequence queries and the scalar
  loop for single-token decode (below `indexerKernelCoarseMinRows`), where
  per-row goroutine overhead would dominate.
- **`kernels.MoE`** (new backend contract, `kernels/backend.go`): the fused
  mixture-of-experts kernels `MoEGateTopK`, the vectorized `TopKIndices`
  selection, and `GroupedMatMulTransposed` (DeepSeek-V4.1 §2.1/§4.2.1; see
  §2.7), implemented in `kernels/simd/moe.go` and `kernels/scalar/moe.go`.

Units: `kernels/simd/parity_test.go` (`TestIndexerScoresParity`,
`TestIndexerBlockMaxParity`), `model/indexer_test.go`
(`TestTopKIndicesMatchesFullSort`, `TestTopKIndicesTieBreak`,
`TestHierarchicalPoolUsesBlockMax`, `TestReindexDistillationRestrictedToPool`).

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

**Scope note:** the low-rank query/KV latent (`--query-compression-dim`,
`--kv-latent-dim`) reduces **compute and parameters, not cache bytes**; it keeps
the full per-head KV cache. Both are genuine architectural changes and must be
trained from scratch.

**Absorbed MLA latent cache.** `--mla-latent R` enables a separate absorbed MLA
mode: the content key/value latent is shared (`c_kv`, width `R`) and a decoupled
rotary key is stored per KV head, so the cache holds one latent row per token
instead of a key and a value per head. At inference `model/mla.go` folds the
content key up-projection into the query, concatenates the folded content query
with the rotary query (and the latent with the rotary key) so the attention score
is the per-position sum of the two, and applies the value up-projection to the
attention-weighted latent sum. This is a genuine architectural change trained
from scratch; it is incompatible with compression, CED, sparsity, cross-layer
reuse, the low-rank settings, head-wise Muon, and value embeddings, and
`Config.Validate` rejects those combinations. `KVBytesPerToken`/`KVReadBytes`
report the reduced footprint. A cache-less `Forward` (used by evaluation and
distillation) allocates a scratch latent cache, so an MLA model also works as a
frozen teacher.

Units: `TestBackpropDirectionalGradientCheckLowRank` (query, kv, both),
`TestBackpropPerElementLowRankCED`, `TestLowRankReducesParameters`,
`TestSaveLoadRoundtripLowRank`, `TestForwardLowRankDecodePath`,
`TestMLAConfigValidation`, `TestMLARotaryDimensionDefault`, `TestMLAParameterShapes`,
`TestMLATrainStepFinite`, `TestBackpropDirectionalGradientCheckMLA`,
`TestBackpropPerElementMLA`, `TestMLATrainOverfitsTiny`,
`TestMLAInferenceMatchesTraining`, `TestMLADecodeMatchesPrefill`,
`TestMLACacheCodecRoundTrip`, `TestMLAKVBytesReduced`,
`TestMLAWithGroupedQueryAttention`, `TestMLAForwardNilCache`,
`TestSaveLoadRoundtripMLA`, `TestEngineMatchesNaiveGenerateMLA`,
`TestMLACacheManagerGenerationMatchesFullPrefill`,
`TestMLAPrefixCacheGenerationMatchesFullPrefill`, `TestDistillWithMLATeacher`.

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

**Persistent multi-entry cache tier.** `inference/cache.go` generalizes the
single-entry cache into `CacheManager`, the in-memory analogue of the paper's
persistent KV cache (§3.2.1): it keeps multiple prefixes, evicts by LRU and TTL,
bounds memory by entry count or allocated bytes, and — when given a directory —
serializes each snapshot to disk via `KVBuffer.MarshalBinary` /
`model.UnmarshalKVBuffer`, keeping only metadata in memory. `Engine.Cache` takes
precedence over `Engine.Prefix`, and `cmd/server` enables a small in-memory tier
by default (`--prefix-cache-size`, `--prefix-cache-dir`, `--prefix-cache-ttl`).
As with `PrefixCache`, a hit requires the stored tokens to be a strict prefix of
the request, so the reused state stays exactly consistent with a full prefill.

Units: `TestKVCacheCodecRoundTrip`, `TestKVCacheCodecPlainRoundTrip`,
`TestKVCacheCodecRejectsInvalidData`, `TestCacheManagerLookupStore`,
`TestCacheManagerLongestPrefix`, `TestCacheManagerLRUEviction`,
`TestCacheManagerTTL`, `TestCacheManagerDiskTier`,
`TestCacheManagerGenerationMatchesFullPrefill`,
`TestCacheManagerDiskGenerationMatchesFullPrefill`.

**SWA pool and bounded replay.** `CacheOptions.StripSWA` drops the raw
sliding-window rows from compressed persistent snapshots, so only the global
compressed KV, cached indexer keys, and the buffered compression tail are stored
(the paper's persistent-cache policy of not retaining SWA KV). `Engine.SWACache`
is a small short-TTL cache of *full* snapshots checked before the stripped tier,
so common hits avoid replay; a hit only in the stripped tier is recovered by
`Transformer.ReplaySWA`, which replays the tail of the cached prefix with the
global state already in place. The replay share (`compressionShare.replay`)
truncates the local branch to the replay segment, exactly as the paper specifies
for a replay starting at position `s`. The replayed local state is approximate by
design, except when the window covers the prefix, where it is exact.

Units: `TestKVCacheCodecStrippedRoundTrip`,
`TestReplaySWARebuildsExactWhenWindowCoversPrefix`,
`TestReplaySWARepopulatesTailWindow`, `TestSWAPoolGenerationMatchesFullPrefill`,
`TestStripSWAReplayRuns`.

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

**DSpark speculative decoding.** `cmd/dspark_train` builds a `model.DSpark`: a
small trunk (3 blocks, context capped at 128 tokens) distilled from a frozen,
uncompressed backbone, plus a low-rank **Markov head** and a **confidence head**
trained on top of the frozen trunk. `DSpark.Draft` is **semi-autoregressive**
(DeepSeek-V4.1 §2.4.3): it runs a *single* trunk forward over the context
followed by `count-1` learned **draft-mask positions**
(`Transformer.ForwardHiddenSuffix` + `dspark.draft_mask.embedding`) and reads
the parallel next-token logits of the last `count` positions, then biases each
drafted token on the previous one through the Markov head and returns a
per-token confidence. The number of positions is constant
(`model.DefaultDraftPositions()` = 5, overridable with `--draft-positions`).
`Engine.DSpark` + `Engine.Speculative` then enable **exact greedy** speculative
decoding in `inference/speculative.go`: each round the drafter proposes up to
`DraftLength` (default 5) tokens, the **confidence scheduler** trims the block
using the estimated **prefix-survival probability** (the product of the
per-position conditional acceptance confidences) against
`ConfidenceThreshold` (default 0.5), the target model verifies the whole block
in a single forward, and the longest matching prefix is accepted. Output is
bit-for-bit identical to greedy decoding regardless of the scheduler (verified
by `TestSpeculativeMatchesGreedy` and `TestSpeculativeWithDSparkMatchesGreedy`).
It is limited to single-row, greedy (`--temperature 0`), uncompressed models and
does not run the tool-call state machine; other requests transparently use the
normal path. `cmd/infer_bench` (`--drafter ... --speculative`) and `cmd/chat_cli`
expose it, and a DSpark checkpoint is detected via `checkpoint.IsDSpark`.

The draft-mask embedding is a checkpointed but fixed input representation: the
Markov and confidence heads are trained under the same placeholder forward they
see at inference, but fully training the mask embedding/trunk for semi-AR
generation is left as future work.

This feature required fixing a pre-existing smear-recurrence inconsistency:
`smearAdd` chained the *post-smear* activation of the previous token while the
sequential decode path and the smear backward pass both assume the *pre-smear*
embedding, so batched (prefill) and token-by-token forwarding diverged whenever
`smearLambda` was non-zero. `smearAdd` now visits rows in descending order and a
multi-token forward against a partially filled cache seeds its first position
from the cached previous embedding (`smearSeed`), making batched and sequential
decoding agree (`TestBatchedForwardMatchesSequentialDecode`).

Units: `TestDrafterConfigBounded`, `TestDraftTokensLengthAndRange`,
`TestDraftTokensEmptyContext`, `TestDraftTokensBoundedByContext`,
`TestBatchedForwardMatchesSequentialDecode`,
`TestBackpropDirectionalGradientCheckSmear`, `TestSpeculativeMatchesGreedy`,
`TestSpeculativeMatchesGreedyWithLongDraft`, `TestLoadedDrafterSpeculation`,
`TestSpeculativeEligibility`, `TestDSparkDraftAndConfidence`,
`TestDraftFirstTokenMatchesTrunk`, `TestDSparkDraftBoundedByContext`,
`TestDSparkTrainHeadsStepFinite`, `TestDSparkRoundTrip`,
`TestSpeculativeWithDSparkMatchesGreedy`, `TestScheduledLength`.

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

### 9.1 Presets (`cmd/base_train`, `cmd/trainer`)

The architecture is selected with a single flag, `--preset`. The recommended
DeepSeek-V4.1-Flash stack is the default, so a training run needs no
feature flags at all.

| Preset | Default | What it enables |
|---|---|---|
| `flash` | ✓ | HCA compression (ratio 4), CSA sparse attention (top-k 8), hierarchical indexer (pool 8), cross-layer reuse (`FRU`), SWA (128), CED, SwiGLU DeepSeekMoE, GQA, head-wise Muon, partial RoPE, Sinkhorn embeddings/head |
| `latent` | | Absorbed MLA + SwiGLU DeepSeekMoE + GQA + partial RoPE + Sinkhorn |
| `dense` | | The historical dense nanochat decoder (full attention, ReLU² MLP, no MoE) |

The remaining flags are the ordinary training controls, not architecture
switches: `--depth`, `--max-seq-len`, `--vocab-size`, `--num-iterations`,
`--device-batch-size`, `--total-batch-size`, `--data-dir`, `--data-format`,
`--model-tag`, `--base-dir` (plus the tokenizer flags on `cmd/trainer`).

The former per-feature flags (`--compression-ratio`, `--sparse-topk`,
`--indexer-*`, `--reuse-pattern`, `--swa-window`, `--ced`, `--head-wise-muon`,
`--sinkhorn-embeddings`, `--query-compression-dim`, `--kv-latent-dim`,
`--mla-*`, `--moe`, `--num-experts`, `--experts-per-token`,
`--expert-hidden-dim`, `--kv-head-ratio`) have been removed: the preset is the
best default, and the exact numbers are derived from `--depth` and the
embedding width.

### 9.2 Programmatic configuration (`model.Config`)

`model.Config` still carries every field, and it is serialized into each
checkpoint, so a loaded model reconstructs its architecture exactly. Library
callers get the same defaults without the CLI:

```go
preset, _ := model.ParsePreset("flash") // "" also selects the default
cfg := model.ConfigForPreset(preset, depth, vocabSize, 64, 128, seqLen, "SSSL")
model := model.NewTransformer(cfg)
groups := model.SetupOptimizer(0.01, 0.1, 0.02, 0.28, 0.5, preset.UsesSinkhorn())
```

`Config.ApplyPreset` only turns *on* disabled options, so an explicit
configuration is never silently broadened. MoE shape (`NumExperts`,
`NumExpertsPerToken`, `ExpertHiddenDim`, `SharedExpertHiddenDim`) defaults to
`max(8, 2·depth)` experts, top-2, with an expert width equal to the embedding
width. `Config.Validate` still enforces the cross-field rules (sparse requires
compression, CED requires compression + SWA, MLA excludes the compression
stack and head-wise Muon, etc.).

### 9.3 Recommended configuration

The recommended long-context configuration **is the default**:

```bash
go run ./cmd/base_train --depth 12 --max-seq-len 8192
```

which is equivalent to the explicit `flash` preset. Use `--preset latent` for
the absorbed-MLA variant (smaller KV and parameter count, at the cost of the
compression/sparse stack) and `--preset dense` for the classic decoder.

---

## 10. Benchmarking and verification

**Reproduce a long-context run.** `benchmark_longctx.sh` trains a tiny model
with a given architecture config and sweeps prefill/decode at one or more
sequence lengths:

```bash
SEQS=4096,16384 BATCH_SIZES=1,16,64 DECODE_TOKENS=16 ./benchmark_longctx.sh

CONFIG="--preset latent" ./benchmark_longctx.sh
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
| MoE / routing / balancing | `model/moe_test.go`, `trainer/moe_test.go`, `model/checkpoint/moe_test.go` |
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
- **Engram conditional memory and Single-Pass mHC.** gonano is a
  single-residual-stream model; these are architectural components of the 552B
  model and do not map onto it. The MoE backbone *is* implemented (§2.7).
- **End-to-end semi-autoregressive DSpark training.** The single-pass parallel
  draft and the survival scheduler are implemented, and the Markov/confidence
  heads are trained under the same placeholder forward as inference, but the
  draft-mask embedding is a fixed checkpointed input rather than being optimized
  jointly with the trunk.
- **EPD disaggregation and the GPU kernel fusions** (Mega-* kernels, FlashMLA):
  single-process CPU serving only.

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
| MoE feed-forward + routing | `model/moe.go`, `model/mlp.go` |
| Indexer kernels | `kernels/simd/indexer.go`, `kernels/scalar/indexer.go` |
| Long-context benchmark | `benchmark_longctx.sh`, `cmd/infer_bench` |
