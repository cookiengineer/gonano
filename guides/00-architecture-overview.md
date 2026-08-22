# gonano — Architecture Overview

This document explains how the gonano codebase is organized, how the packages
relate to one another, and what actually happens — step by step — when you
train a model or run inference. It is meant to be read top to bottom: each
section builds on the previous one. There are no diagrams; instead, the flow is
described with ordered lists you can follow like a recipe.

The project is a single Go module (`github.com/cookiengineer/gonano`) of
importable, non-`internal` packages. The executables in `cmd/` are thin
wrappers that only parse flags and call the library.

---

## 1. The three layers

Think of the code as three stacked layers, plus a set of stand-alone tools.

1. **The execution substrate** — `tensor` and `nn`. Everything numeric lives
   here: dense float32 tensors, SIMD-accelerated kernels (matmul, softmax,
   RMSNorm, reductions), and the two primitive layer types (`Linear`,
   `Embedding`). This layer has no idea what a transformer or a tokenizer is.

2. **The model** — `model`. This is the actual nanochat GPT transformer: the
   `Config` that defines its shape, the `Transformer` that owns the weights, and
   both the forward pass (`Forward`) and the training forward/backward pair
   (`TrainForward`/`TrainBackward`). It is built out of `nn` primitives and
   `tensor` kernels.

3. **The orchestration** — `train`, `infer`, `eval`, `data`, `tokenizer`. These
   packages drive the model: producing data, running training loops, sampling
   completions, and scoring results.

The stand-alone tools are `optim` (the optimizers), `checkpoint` (persistence),
`exec` (sandboxed code execution for evaluations), and the small foundation
packages `parallel`, `device`, and `logging`.

The reason for this split: training, inference, and the numeric core are
decoupled, so each can be imported and understood on its own.

---

## 2. Dependency layers

No package may import a package above it. Reading this list bottom-to-top is
the same order the program bootstraps.

1. `parallel`, `logging`, `device` — the leaves. A goroutine worker pool, an
   `slog` setup, and CPU capability detection (`simd.VectorBitSize`,
   `runtime.GOMAXPROCS`).

2. `tensor` — depends only on `parallel`. Owns `Tensor` (row-major float32
   data plus an optional gradient buffer) and every SIMD kernel.

3. `nn` — depends on `tensor` and `parallel`. `Linear` and `Embedding` wrap the
   tensor matmul/gather kernels and add analytic gradients.

4. `model` — depends on `nn` and `tensor`. The transformer and its
   forward/backward. It also imports `optim` in one place (`SetupOptimizer`
   returns `optim.ParamGroup`s), which keeps parameter grouping next to the
   parameters.

5. `optim` — depends on `tensor`. `MuonAdamW` routes parameter groups to
   `AdamW` or `Muon`.

6. `tokenizer` — depends on `parallel`. Byte-level BPE training and inference,
   plus chat rendering. No dependency on the model.

7. `data` — depends on `tokenizer`, `tensor`, and its own `data/parquet`
   sub-package. Turns files into token tensors.

8. `train` — depends on `model`, `optim`, `data`, `tensor`. The training loops
   and hyperparameter derivation.

9. `infer` — depends on `model`, `tokenizer`, `tensor`. The KV-cache engine,
   sampler, and calculator tool.

10. `eval` — depends on `model`, `tokenizer`, `infer`, `tensor`, plus the
    `eval/tasks` sub-package (which depends only on `tokenizer`).

11. `exec` and `checkpoint` — `exec` is a leaf (`os/exec` only); `checkpoint`
    depends on `model` and `tensor`.

12. `cmd/*` — the executables import the library packages they need.

This ordering is enforced implicitly (there are no import cycles); it is also
why the KV cache (`model.KVBuffer`) lives in `model` rather than `infer` — the
model's forward pass needs to write to it, and `model` may not import `infer`.

---

## 3. Package-by-package tour

### 3.1 `tensor`

1. `Tensor` is `{ Shape []int, Data []float32, Grad []float32 }`; `Int32s` is
   the same idea for token ids and masks. Both are row-major and contiguous.

2. The compute kernels are `MatMul` (`A @ B`) and `MatMulTransB` (`A @ Bᵀ`),
   both tiled and parallelized over output rows via `parallel`.

3. Elementwise/normalization kernels: `RMSNormLastDim`, `SoftmaxLastDim`,
   `Relu2`, `Sigmoid`, `Tanh`, `Softcap`, and reductions (`SumAll`, `MaxAll`,
   `ArgMaxLastDim`).

4. Loss and gradients: `CrossEntropy`, `CrossEntropyPerPosition`, and
   `CrossEntropyGrad` (the `softmax - onehot` gradient), plus the backward
   primitives `MatMulBackward`, `RMSNormBackward`, `SoftcapBackward`,
   `SigmoidBackward`, and `Relu2Backward`.

5. `RNG` wraps `math/rand/v2` and provides `SampleMultinomial`.

The `simd` package is used only here (and in a tiny bit of `optim`); the rest
of the project never touches `simd` directly.

### 3.2 `nn`

1. `Linear{Weight [out,in]}` — `Forward` is `x @ Wᵀ`, `Backward` accumulates
   `gradW = gradOutᵀ @ x` and returns `gradIn = gradOut @ W`. No biases
   anywhere in the model.

2. `Embedding{Weight [vocab,dim]}` — `Forward` gathers rows, `Backward`
   scatters the gradient back into the gathered rows.

3. `init.go` provides `InitNormal`, `InitUniform`, `InitZeros`, and
   `InitValue`.

### 3.3 `model`

1. `Config` is the architecture description: `SequenceLen`, `VocabSize`,
   `NumLayer`, `NumHead`, `NumKVHead`, `EmbedDim`, `WindowPattern`.
   `ConfigForDepth(depth, vocab, 64, 128, seqLen, "SSSL")` derives a whole
   config from a single depth dial.

2. `NewTransformer(cfg)` allocates every parameter (zero-initialized) and
   precomputes the rotary `cos`/`sin` tables; `InitWeights(rng)` applies the
   nanochat initialization (normal embeddings, uniform matrices, zero
   projections).

3. `NamedParameters()` returns every trainable tensor keyed by a state-dict
   name; this single map is the source of truth used by `InitWeights`,
   `SetupOptimizer`, and `checkpoint`.

4. `Forward(idx, cache)` runs the whole stack: embedding, RMSNorm, the "smear"
   step, the per-layer residual trunk (attention + ReLU² MLP, with
   `resid_lambdas`/`x0_lambdas` and value embeddings), the backout step, final
   norm, `lm_head`, and the logit softcap.

5. `TrainForward`/`TrainBackward` are the training pair. `TrainForward` saves a
   context of intermediate activations; `TrainBackward` walks it in reverse
   and accumulates each parameter's `.Grad`.

6. `KVBuffer` is the inference key/value cache (`model/kvcache.go`).

7. `SetupOptimizer` mirrors nanochat: embeddings, `lm_head`, and scalars go to
   AdamW; the 2-D matrix weights go to Muon, grouped by shape.

### 3.4 `optim`

1. `ParamGroup` carries a `Kind` (AdamW or Muon) plus its hyperparameters.

2. `NewMuonAdamW(groups)` allocates per-parameter state (AdamW moments, or
   Muon momentum + factored second moment).

3. `Step()` consumes each parameter's `.Grad` and updates it; `ZeroGrad()`
   clears the gradients. AdamW is a fused elementwise update; Muon runs
   Nesterov momentum, MuonEq row equilibration, Polar-Express orthogonalization,
   Muon+ renormalization, variance reduction, and cautious weight decay.

### 3.5 `tokenizer`

1. `SplitPieces` is a hand-written implementation of the GPT-4 regex split
   pattern (Go's RE2 rejects its possessive quantifiers and lookahead).

2. `TrainBPE` runs byte-level BPE with a max-heap over byte pairs and an
   inverted index so each merge only touches the pieces that contain the pair.

3. `Tokenizer` encodes/decodes, renders conversations
   (`RenderConversation`/`RenderForCompletion`), and serializes to JSON.

### 3.6 `data` and `data/parquet`

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

### 3.7 `train`

1. `DeriveHyperparams` turns the depth dial into the training horizon, batch
   size, LR scale, and weight decay using the scaling laws.

2. `scheduler.go` holds the LR, Muon momentum, and weight-decay schedules.

3. `Trainer` runs the loop: `TrainStep` (forward + backward, scaled by gradient
   accumulation) and `StepOptimizer` (apply schedules, step, zero grads).

4. `TrainSFT` runs supervised fine-tuning; `PolicyGradientStep`/`RLStep`
   implement the GRPO/REINFORCE policy gradient.

### 3.8 `infer`

1. `Engine.Generate` runs a batch-1 prefill, replicates the KV cache, then
   loops: sample, run the tool-use state machine, and forward the next token
   column.

2. `SampleNextToken` implements argmax, temperature, and top-k sampling.

3. `UseCalculator` is a safe arithmetic/`.count()` evaluator used by the tool
   loop.

4. `Measure` times TTFT and per-step decode latency.

### 3.9 `eval`, `eval/tasks`, `exec`

1. `eval` computes `BitsPerByte`, the DCLM CORE metric (`EvaluateTask`), and
   `ChatCORE` (`CategoricalAccuracy`/`GenerativeAccuracy`), plus a minimal YAML
   parser for `core.yaml`.

2. `eval/tasks` defines the `Task` interface and `TaskMixture`, and the
   datasets (MMLU, GSM8K, ARC, HumanEval, SmolTalk).

3. `exec` runs untrusted Python (HumanEval) in a sandboxed subprocess.

### 3.10 `checkpoint`

1. `Save`/`Load` handle the native `.gn` format (magic, JSON metadata, raw
   float32 blobs).

2. `ExportGGUF`/`LoadGGUF` convert to and from the GGUF container, tagged with
   the `nanochat` architecture.

3. `LoadModel` rebuilds a `model.Transformer` from a loaded metadata+params
   pair, and `LoadAny` auto-detects `.gn` vs `.gguf`.

---

## 4. Behind the scenes: one pretraining step

Follow these steps to trace what `cmd/base_train` actually does.

1. `base_train` loads (or falls back to creating) a tokenizer, then calls
   `model.ConfigForDepth(depth, vocab, ...)` to derive the model shape and
   `model.NewTransformer` to allocate it.

2. `InitWeights` fills the parameters, and `SetupOptimizer` builds the
   AdamW/Muon parameter groups.

3. A `data.NewPretrainLoader` is built over a `ParquetSource` (or
   `MarkdownSource`), and `train.NewTrainer` wraps the model + optimizer.

4. The training loop pulls a batch of token ids `x` and its shifted targets
   `y` from the loader. Each row begins with `<|bos|>` and is packed by the
   best-fit algorithm.

5. `Trainer.TrainStep` runs `Transformer.TrainForward`, which walks the
   transformer and saves activations, then computes `CrossEntropy` against `y`.

6. `tensor.CrossEntropyGrad` produces the `softmax − onehot` gradient, and
   `TrainBackward` flows it in reverse — through `lm_head`, the final RMSNorm,
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

2. `tokenizer.LoadTokenizer` loads the tokenizer; `infer.NewEngine` wraps both.

3. `Engine.Generate` runs the prompt through `Forward` once with a batch-1
   `model.KVBuffer`, which stores one key/value tensor per layer and per
   key/value head.

4. `model.PrefillFrom` clones that cache across the requested number of sample
   rows, and expands the smear state.

5. In the decode loop, `SampleNextToken` picks the next token per row from the
   last logits, the tool-use state machine rewrites any
   `<|python_start|>…<|python_end|>` spans into calculator outputs, and the
   resulting token column is forwarded again through `Forward` against the KV
   cache.

6. The loop ends when every row emits `<|assistant_end|>` or the token budget
   is exhausted; `GenerateBatch` collects the final sequences, which
   `tokenizer.Decode` turns back into text.

The key performance fact: decode re-reads all weights plus the KV cache for a
single token, so it is memory-bandwidth-bound — which is why `infer_bench`
shows higher tokens-per-second at larger batch sizes.

---

## 6. Artifacts on disk

1. `$GONANO_BASE_DIR/tokenizer/tokenizer.json` — the BPE vocabulary
   (default base dir is `~/.cache/gonano`).

2. `$GONANO_BASE_DIR/base_checkpoints/<tag>/model_<step>.gn` — a pretraining
   checkpoint; the same layout is used under `chatsft_checkpoints` and
   `chatrl_checkpoints`.

3. `*.gguf` — an exported GGUF container, written by `cmd/export`.

4. `$GONANO_BASE_DIR/base_data/*.parquet` — the downloaded text shards
   (a flat `text` column each).

---

## 7. Where parallelism lives

1. `parallel.Pool` is a process-wide worker pool sized to `GOMAXPROCS`.

2. `tensor` parallelizes matmul and reductions over output rows and fan out
   tokenization in `data` and `tokenizer`.

3. `infer` gets parallelism for free by batching sample rows: the batched
   matmuls are themselves parallelized.

4. Training uses a single model copy with gradient accumulation; each
   micro-batch's forward/backward is parallelized at the tensor level.

---

## 8. Quick index

| Looking for | Start here |
|---|---|
| The model architecture | `model/config.go`, `model/transformer.go` |
| A single layer | `model/block.go`, `model/attention.go`, `model/mlp.go` |
| The backprop math | `model/train.go`, `model/train_transformer.go` |
| The optimizers | `optim/optimizer.go` (AdamW), `optim/muon.go` (Muon) |
| The SIMD kernels | `tensor/matmul.go`, `tensor/norm.go`, `tensor/elementwise.go` |
| Tokenizer internals | `tokenizer/splitter.go`, `tokenizer/bpe.go` |
| Data loading | `data/dataloader.go`, `data/dataset.go` |
| The training loop | `train/trainer.go`, `train/scaling.go` |
| The inference engine | `infer/engine.go`, `infer/sampler.go` |
| Checkpoint/GGUF | `checkpoint/checkpoint.go`, `checkpoint/gguf.go` |

For a from-scratch walkthrough, continue with
[00-quickstart.md](00-quickstart.md); for training specifics see
[01-training.md](01-training.md).
