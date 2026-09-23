# gonano -- Architecture Overview

This document explains how the gonano codebase is organized, how the packages
relate to one another, and what actually happens -- step by step -- when you
train a model or run inference. It is meant to be read top to bottom: each
section builds on the previous one. There are no diagrams; instead, the flow is
described with ordered lists you can follow like a recipe.

The project is a single Go module (`github.com/cookiengineer/gonano`) of
importable, non-`internal` packages. The executables in `cmd/` are thin
wrappers that only parse flags and call the library.

---

## 1. The three layers

Think of the code as three stacked layers, plus a set of stand-alone tools.

1. **The numeric core** -- `kernels` (the backend contract), `kernels/simd`,
   `kernels/scalar`, `tensors`, and `model/layers`. The `kernels.Backend` interface is the
   API contract every other package computes through: elementwise vector ops,
   reductions, matmul, fused softmax/RMSNorm, and flash attention. `kernels/simd`
   is the production AVX-512 backend; `kernels/scalar` is a portable reference.
   `tensors` owns the dense float32/int32 tensor types and the high-level ops
   that delegate to the active backend, and `model/layers` adds the two primitive layer
   types (`Linear`, `Embedding`). This layer has no idea what a transformer or a
   tokenizer is.

2. **The model** -- `model`. This is the actual nanochat GPT transformer: the
   `Config` that defines its shape, the `Transformer` that owns the weights, and
   both the forward pass (`Forward`) and the training forward/backward pair
   (`TrainForward`/`TrainBackward`). It is built out of `model/layers` primitives and
   `tensors` ops, and its attention uses the flash-attention kernel.

3. **The orchestration** -- `trainer`, `inference`, `evaluator`, `data`, `tokenizer`, and -- for
   Mixture-of-Experts Sharding -- `router` and `bank`. These packages drive the model:
   producing data, running training loops, sampling completions, routing a
   prompt to a bank of expert models, and scoring results.

The stand-alone tools are `optimizer` (the optimizers), `model/checkpoint`
(persistence), `executor` (sandboxed code execution for evaluations), and the
internal foundation packages `internal/parallel`, `internal/device`, and
`internal/logging`.

The reason for this split: training, inference, and the numeric core are
decoupled, so each can be imported and understood on its own.

---

## 2. Dependency layers

No package may import a package above it. Reading this list bottom-to-top is
the same order the program bootstraps.

1. `internal/parallel`, `internal/logging`, `internal/device` -- the leaves. A goroutine worker pool, an
   `slog` setup, and CPU capability detection (`simd.VectorBitSize`,
   `runtime.GOMAXPROCS`).

2. `kernels` -- the backend contract. It is dependency-free and declares the
   `Backend` interface (`Elementwise`, `Reductions`, `LinearAlgebra`, `Rows`,
   `Attention`).

3. `kernels/scalar` and `kernels/simd` -- the implementations. `kernels/scalar` is
   pure Go (no `simd` import); `kernels/simd` depends on `internal/parallel` and the
   standard-library `simd` package and is the default backend.

4. `tensors` -- depends on `kernels` and `internal/parallel`. Owns `Tensor` (row-major
   float32 data plus an optional gradient buffer), `Int32s`, and the high-level
   operations that delegate to the active `kernels.Backend`. The default backend
   is `kernels/simd`; `UseKernelBackend` swaps it (the scalar backend is used in
   tests and portable builds).

5. `model/layers` -- depends on `tensors` and `internal/parallel`. `Linear` and `Embedding` wrap the
   tensor matmul/gather operations and add analytic gradients.

6. `model` -- depends on `model/layers` and `tensors`. The transformer and its
   forward/backward. It also imports `optimizer` in one place (`SetupOptimizer`
   returns `optimizer.ParamGroup`s), which keeps parameter grouping next to the
   parameters.

7. `optimizer` -- depends on `tensors`. `MuonAdamW` routes parameter groups to
   `AdamW` or `Muon`.

8. `tokenizer` -- depends on `internal/parallel`. Byte-level BPE training and inference,
   plus chat rendering. No dependency on the model.

9. `data` -- depends on `tokenizer`, `tensors`, and its own `data/parquet`
   sub-package. Turns files into token tensors.

10. `trainer` -- depends on `model`, `optimizer`, `data`, `tensors`. The training
    loops and hyperparameter derivation.

11. `inference` -- depends on `model`, `tokenizer`, `tensors`. The KV-cache engine,
    sampler, and calculator tool. Its `Ensemble` blends several models' logits
    for Mixture-of-Experts Sharding.

12. `router` and `bank` -- `router` depends on `model`, `tensors`, `optimizer`,
    `model/checkpoint`; `bank` depends on `model`, `model/checkpoint`,
    `tokenizer`. The domain meta-router and its on-demand model bank.

13. `evaluator` -- depends on `model`, `tokenizer`, `inference`, `tensors`, plus the
    `evaluator/tasks` sub-package (which depends only on `tokenizer`).

14. `executor` and `model/checkpoint` -- `executor` is a leaf (`os/exec` only);
    `model/checkpoint` depends on `model` and `tensors`.

15. `cmd/*` -- the executables import the library packages they need.

This ordering is enforced implicitly (there are no import cycles); it is also
why the KV cache (`model.KVBuffer`) lives in `model` rather than `inference` -- the
model's forward pass needs to write to it, and `model` may not import `inference`.

---

## 3. Package-by-package tour

### 3.1 `kernels`, `kernels/simd`, `kernels/scalar`

1. `kernels` declares the `Backend` interface. It groups five contracts:
   `Elementwise` (`Add`, `Subtract`, `Multiply`, `Divide`, `Scale`, `AddScaled`,
   `Negate`, `Abs`, `Square`, `ReluSquared`, `SwiGLU`, `Exp`, `Sigmoid`, `Tanh`,
   `Rsqrt`),
   `Reductions` (`Sum`, `Max`, `ArgMax`), `LinearAlgebra` (`MatMul`,
   `MatMulTransposed`, `DotProduct`), `Rows` (`SoftmaxLastDim`,
   `RMSNormLastDim`), and `Attention` (`AttentionForward`,
   `AttentionBackward`).

2. `kernels/simd` is the production backend: vectorized over the
   standard-library `simd` lanes and parallelized across CPU cores via
   `internal/parallel`. Its fused `AttentionForward`/`AttentionBackward` implement flash
   attention: the `[queryLength, keyLength]` score matrix is never materialized
   -- only a `[queryBlockSize, keyBlockSize]` tile and a
   `[queryBlockSize, headDim]` accumulator are live, with the online softmax
   statistics carried forward. The transcendental operations (`Exp`, `Sigmoid`,
   `Tanh`, `Rsqrt`) fall back to `math` because the `simd` package exposes none.

3. `kernels/scalar` implements the same contract in portable Go and is the
   reference used by the scalar-vs-SIMD parity tests. It materializes the score
   matrix, so agreement between the two backends validates the flash algorithm.

4. Only `kernels/simd` imports the standard-library `simd` package; the rest of
   the project computes through the contract. Receiver methods whose bodies
   contain a `simd` intrinsic delegate to package-level `...Core` functions:
   the Go compiler currently rejects a method that type-checks an intrinsic
   whose name matches the enclosing method, so the delegation is required.

### 3.2 `tensors`

1. `Tensor` is `{ Shape []int, Data []float32, Grad []float32 }`; `Int32s` is
   the same idea for token ids and masks. Both are row-major and contiguous.

2. Every operation allocates a result and delegates the arithmetic to
   `KernelBackend()`. The default is `kernels/simd`; tests and portable builds
   can call `UseKernelBackend(scalar.New())`.

3. Operations: `MatMul`/`MatMulTransposed`, `Add`/`Subtract`/`Multiply`/`Divide`,
   `Scale`, `AddScaled`, `Negate`, `Abs`, `Square`, `ReluSquared`, `Exp`,
   `Sigmoid`, `Tanh`, `Rsqrt`, `Softcap`, `SoftmaxLastDim`, `RMSNormLastDim`,
   reductions (`SumAll`, `MaxAll`, `ArgMax`, `SumLastDim`, `MaxLastDim`,
   `ArgMaxLastDim`), and `AttentionForward`/`AttentionBackward`.

4. Loss and gradients: `CrossEntropy`, `CrossEntropyPerPosition`,
   `CrossEntropyGrad` (the `softmax - onehot` gradient), plus the backward
   primitives `MatMulBackward`, `MatMulTransposedBackward`, `RMSNormBackward`,
   `SoftcapBackward`, `SigmoidBackward`, `ReluSquaredBackward`, and
   `ScaleBackward`.

5. `RNG` wraps `math/rand/v2` and provides `SampleMultinomial`.

### 3.3 `model/layers`

1. `Linear{Weight [out,in]}` -- `Forward` is `x @ W^T`, `Backward` accumulates
   `gradW = gradOut^T @ x` and returns `gradIn = gradOut @ W`. No biases
   anywhere in the model.

2. `Embedding{Weight [vocab,dim]}` -- `Forward` gathers rows, `Backward`
   scatters the gradient back into the gathered rows.

3. `init.go` provides `InitNormal`, `InitUniform`, `InitZeros`, and
   `InitValue`.

### 3.4 `model`

1. `Config` is the architecture description: `SequenceLen`, `VocabSize`,
   `NumLayer`, `NumHead`, `NumKVHead`, `EmbedDim`, `WindowPattern`, plus the
   DeepSeek-V4.1 features (compression, sparsity, reuse, CED, MoE, MLA). A
   preset fills the best defaults:
   `ConfigForPreset(preset, depth, vocab, 64, 128, seqLen, "SSSL")` derives the
   whole config from a single depth dial and then applies the preset
   (`flash`/`latent`/`dense`).

2. `NewTransformer(cfg)` allocates every parameter (zero-initialized) and
   precomputes the rotary `cos`/`sin` tables; `InitWeights(rng)` applies the
   nanochat initialization (normal embeddings, uniform matrices, zero
   projections).

3. `NamedParameters()` returns every trainable tensor keyed by a state-dict
   name; this single map is the source of truth used by `InitWeights`,
   `SetupOptimizer`, and `checkpoint`.

4. `Forward(idx, cache)` runs the whole stack: embedding, RMSNorm, the "smear"
   step, the per-layer residual trunk (attention + feed-forward, with
   `resid_lambdas`/`x0_lambdas` and value embeddings), the backout step, final
   norm, `lm_head`, and the logit softcap. The feed-forward is a dense ReLU^2 MLP
   by default, or an always-active SwiGLU shared expert plus a routed
   DeepSeekMoE (`model/moe.go`) when `Config.NumExperts > 0`.

5. `TrainForward`/`TrainBackward` are the training pair. `TrainForward` saves a
   context of intermediate activations; `TrainBackward` walks it in reverse
   and accumulates each parameter's `.Grad`.

6. `KVBuffer` is the inference key/value cache (`model/kvcache.go`).

7. Attention (both `Forward` and `forwardTraining`/`backwardTraining`) calls
   the flash-attention kernel. The training forward saves only the per-query
   log-sum-exp (`[B,Hq,T]`) instead of the full `[B,Hq,T,T]` probability matrix,
   and the backward recomputes probabilities from those statistics.

8. `SetupOptimizer` mirrors nanochat: embeddings, `lm_head`, and scalars go to
   AdamW; the 2-D matrix weights go to Muon, grouped by shape.

### 3.5 `optimizer`

1. `ParamGroup` carries a `Kind` (AdamW or Muon) plus its hyperparameters.

2. `NewMuonAdamW(groups)` allocates per-parameter state (AdamW moments, or
   Muon momentum + factored second moment).

3. `Step()` consumes each parameter's `.Grad` and updates it; `ZeroGrad()`
   clears the gradients. AdamW is a fused elementwise update; Muon runs
   Nesterov momentum, MuonEq row equilibration, Polar-Express orthogonalization,
   Muon+ renormalization, variance reduction, and cautious weight decay.

### 3.6 `tokenizer`

1. `SplitPieces` is a hand-written implementation of the GPT-4 regex split
   pattern (Go's RE2 rejects its possessive quantifiers and lookahead).

2. `TrainBPE` runs byte-level BPE with a max-heap over byte pairs and an
   inverted index so each merge only touches the pieces that contain the pair.

3. `Tokenizer` encodes/decodes, renders conversations
   (`RenderConversation`/`RenderForCompletion`), and serializes to JSON.

### 3.7 `data` and `data/parquet`

1. `data/parquet` is a minimal dependency-free Parquet reader (Thrift compact
   metadata, Snappy, PLAIN and RLE_DICTIONARY encodings) that reads a flat
   `text` BYTE_ARRAY column.

2. `ParquetSource` iterates Parquet shards' row groups and yields document
   batches; `MarkdownSource` does the same for a directory of `.md` files.

3. `NewPretrainLoader` implements BOS-aligned best-fit packing (every row
   starts with `<|bos|>`; documents are packed, the shortest is cropped to
   fill). `NewSFTLoader` does the same but pads instead of cropping and emits a
   loss mask.

4. `hub.go` provides `ListHFParquetShards`, `DownloadFile`, and
   `DownloadWithLock` for pulling datasets off the HuggingFace Hub.

### 3.8 `trainer`

1. `DeriveHyperparams` turns the depth dial into the training horizon, batch
   size, LR scale, and weight decay using the scaling laws.

2. `scheduler.go` holds the LR, Muon momentum, and weight-decay schedules.

3. `Trainer` runs the loop: `TrainStep` (forward + backward, scaled by gradient
   accumulation) and `StepOptimizer` (apply schedules, step, zero grads).

4. `TrainSFT` runs supervised fine-tuning; `PolicyGradientStep`/`RLStep`
   implement the GRPO/REINFORCE policy gradient.

### 3.9 `inference`

1. `Engine.Generate` runs a batch-1 prefill, replicates the KV cache, then
   loops: sample, run the tool-use state machine, and forward the next token
   column.

2. `SampleNextToken` implements argmax, temperature, and top-k sampling.

3. The `Tool` interface and `Registry` execute tool calls; the built-in
   `Calculator` handles arithmetic/`.count()`, and callers can register more
   Go-implemented tools.

4. `Measure` times TTFT and per-step decode latency.

### 3.10 `evaluator`, `evaluator/tasks`, `executor`

1. `evaluator` computes `BitsPerByte`, the DCLM CORE metric (`EvaluateTask`), and
   `ChatCORE` (`CategoricalAccuracy`/`GenerativeAccuracy`), plus a minimal YAML
   parser for `core.yaml`.

2. `evaluator/tasks` defines the `Task` interface and `TaskMixture`, and the
   datasets (MMLU, GSM8K, ARC, HumanEval, SmolTalk).

3. `executor` runs untrusted Python (HumanEval) in a sandboxed subprocess.

### 3.11 `model/checkpoint`

1. `Save`/`Load` handle the native `.gn` format (magic, JSON metadata, raw
   float32 blobs).

2. `ExportGGUF`/`LoadGGUF` convert to and from the GGUF container, tagged with
   the `nanochat` architecture.

3. `LoadModel` rebuilds a `model.Transformer` from a loaded metadata+params
   pair, and `LoadAny` auto-detects `.gn` vs `.gguf`.

### 3.12 `server`

1. `server.Server` wraps a model, tokenizer, `inference.Engine`, and an
   `inference.Registry` of tools, exposing an OpenAI-compatible HTTP API
   (`POST /v1/chat/completions`, `GET /v1/models`).

2. `renderMessages` translates OpenAI `system`/`user`/`assistant`/`tool`
   messages into the gonano token protocol; `generate` runs the engine and
   parses `<|tool_start|>...<|tool_end|>` spans into structured `tool_calls`.

3. Tool calls are executed server-side in Go by the registry, so a single
   request can carry a complete tool-using turn.

4. In **bank mode** (`cmd/server --bank`), `Server.selectGenerator` classifies
   the request (default: the last user turn), acquires the routed experts from
   the `bank`, and returns a blended `inference.Ensemble`; the routed domains
   are reported in the `X-Gonano-Domains` header. Without a bank, the server
   serves the single model as before.

### 3.13 `router`

1. `NgramClassifier` is a hashed token n-gram linear softmax classifier (the
   fast path); `TransformerClassifier` is a `model.Transformer` encoder plus a
   linear head on its pooled hidden state, trained with `Train` (frozen probe)
   or `FineTune` (joint encoder/head, via `model.Transformer.TrainClassification`).

2. `DistillNgram` trains the fast path from the transformer's labels;
   `Router.Predict` runs the fast path and escalates to the transformer when
   the top-1/top-2 margin is below `Threshold`.

3. A `Router` bundles the domain list, the fast path, the optional transformer,
   and the encoder path; it is saved with the shared `model/checkpoint` format
   and reloaded by `LoadRouterAuto`.

### 3.14 `bank`

1. `Manifest` is a JSON descriptor of the domains (id, name, checkpoint, data
   dirs) and the router path.

2. `Bank` loads experts on demand, validates that each model's `VocabSize`
   matches the shared tokenizer, and evicts by LRU within a byte/model-count
   budget (`BankOptions`).

3. `Acquire`/`AcquireMany` return the requested expert models; the caller
   builds `inference.ModelWeight`s and runs the `Ensemble`.

---

## 4. Behind the scenes: one pretraining step

Follow these steps to trace what `cmd/base_train` actually does.

1. `base_train` loads (or falls back to creating) a tokenizer, then calls
   `model.ConfigForPreset(preset, depth, vocab, ...)` (preset defaults to the
   DeepSeek-V4.1 `flash` stack) to derive the model shape and
   `model.NewTransformer` to allocate it.

2. `InitWeights` fills the parameters, and `SetupOptimizer` builds the
   AdamW/Muon parameter groups.

3. A `data.NewPretrainLoader` is built over a `ParquetSource` (or
   `MarkdownSource`), and `trainer.NewTrainer` wraps the model + optimizer.

4. The training loop pulls a batch of token ids `x` and its shifted targets
   `y` from the loader. Each row begins with `<|bos|>` and is packed by the
   best-fit algorithm.

5. `Trainer.TrainStep` runs `Transformer.TrainForward`, which walks the
   transformer and saves activations, then computes `CrossEntropy` against `y`.

6. `tensors.CrossEntropyGrad` produces the `softmax - onehot` gradient, and
   `TrainBackward` flows it in reverse -- through `lm_head`, the final RMSNorm,
   the backout, then each block's MLP and attention, the residual/smear
   connections, and finally back into the embedding table.

7. `StepOptimizer` applies the LR/momentum/weight-decay schedules and calls
   `MuonAdamW.Step`, which updates every parameter in place and clears the
   gradients.

8. Periodically, `checkpoint.Save` writes the parameters to a `.gn` file.

The single source of truth that ties these together is
`Transformer.NamedParameters`: weight names are the contract between
initialization, the optimizer, and the checkpoint.

---

## 5. Behind the scenes: one inference step

Follow these steps to trace `cmd/chat_cli`.

1. `checkpoint.LoadAny` loads the weights (`.gn` or `.gguf`) and
   `checkpoint.LoadModel` rebuilds the `Transformer` from the stored config.

2. `tokenizer.LoadTokenizer` loads the tokenizer; `inference.NewEngine` wraps both.

3. `Engine.Generate` runs the prompt through `Forward` once with a batch-1
   `model.KVBuffer`, which stores one key/value tensor per layer and per
   key/value head.

4. `model.PrefillFrom` clones that cache across the requested number of sample
   rows, and expands the smear state.

5. In the decode loop, `SampleNextToken` picks the next token per row from the
   last logits, the tool-call state machine rewrites any
   `<|tool_start|>...<|tool_end|>` spans into tool outputs (via the registered
   `inference.Tool` set), and the resulting token column is forwarded again
   through `Forward` against the KV cache.

6. The loop ends when every row emits `<|assistant_end|>` or the token budget
   is exhausted; `GenerateBatch` collects the final sequences, which
   `tokenizer.Decode` turns back into text.

The key performance fact: decode re-reads all weights plus the KV cache for a
single token, so it is memory-bandwidth-bound -- which is why `infer_bench`
shows higher tokens-per-second at larger batch sizes.

---

## 6. Artifacts on disk

1. `$GONANO_BASE_DIR/tokenizer/tokenizer.json` -- the BPE vocabulary
   (default base dir is `~/.cache/gonano`).

2. `$GONANO_BASE_DIR/base_checkpoints/<tag>/model_<step>.gn` -- a pretraining
   checkpoint; the same layout is used under `chatsft_checkpoints` and
   `chatrl_checkpoints`.

3. `*.gguf` -- an exported GGUF container, written by `cmd/export`.

4. `$GONANO_BASE_DIR/base_data/*.parquet` -- the downloaded text shards
   (a flat `text` column each).

---

## 7. Where parallelism lives

1. `parallel.Pool` is a process-wide worker pool sized to `GOMAXPROCS`.

2. `kernels/simd` parallelizes matmul, reductions, and rowwise softmax/RMSNorm
   across CPU cores; `data` and `tokenizer` fan out tokenization. Flash
   attention is parallelized by the model over (batch, head) pairs while each
   head streams over key blocks.

3. `inference` gets parallelism for free by batching sample rows: the batched
   matmuls are themselves parallelized.

4. Training uses a single model copy with gradient accumulation; each
   micro-batch's forward/backward is parallelized at the tensor level.

---

## 8. Quick index

| Looking for | Start here |
|---|---|
| The model architecture | `model/config.go`, `model/transformer.go` |
| A single layer | `model/block.go`, `model/attention.go`, `model/mlp.go` |
| Mixture-of-Experts | `model/moe.go`, `model/mlp.go` |
| The backprop math | `model/train.go`, `model/train_transformer.go` |
| The optimizers | `optimizer/optimizer.go` (AdamW), `optimizer/muon.go` (Muon) |
| The kernel contract | `kernels/backend.go` |
| The SIMD kernels | `kernels/simd/matmul.go`, `kernels/simd/norm.go`, `kernels/simd/elementwise.go` |
| Flash attention | `kernels/simd/attention.go`, `kernels/scalar/attention.go` |
| The scalar reference backend | `kernels/scalar/` |
| Tokenizer internals | `tokenizer/splitter.go`, `tokenizer/bpe.go` |
| Data loading | `data/dataloader.go`, `data/dataset.go` |
| The training loop | `trainer/trainer.go`, `trainer/scaling.go` |
| The inference engine | `inference/engine.go`, `inference/sampler.go` |
| The OpenAI API server | `server/server.go`, `server/handler.go`, `cmd/server` |
| Checkpoint/GGUF | `model/checkpoint/checkpoint.go`, `model/checkpoint/gguf.go` |
| Domain meta-router | `router/router.go`, `router/ngram.go`, `router/transformer.go` |
| Domain model bank | `bank/bank.go` |
| Blended (ensemble) generation | `inference/ensemble.go` |
| MoE Sharding design + workflow | `guides/07-moe-sharding.md` |
| Benchmarks | `guides/08-benchmarking.md` |

For a from-scratch walkthrough, continue with
[00-quickstart.md](00-quickstart.md); for training specifics see
[01-training.md](01-training.md).
