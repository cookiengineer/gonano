# gonano

gonano is a pure-Go LLM training and inference harness. It is a reimplementation
of [nanochat](https://github.com/karpathy/nanochat) built on the experimental Go
1.27 [`simd`](https://pkg.go.dev/simd) standard-library package, and it is
optimized for long-context inference and training on CPUs with `AVX-512`.

Everything is **float32**, parallelism is goroutine-per-op, and the model is a
DeepSeek-V4.1-style transformer. On top of that, gonano introduces
**Mixture-of-Experts Sharding**: a way to run a library of independently trained
domain models that are loaded on demand.

---

## Mixture-of-Experts Sharding

Instead of one monolithic model, gonano lets you train and serve a **bank of
expert models**, one per semantic domain:

- **Expert models.** Each domain (`physics`, `math`, `wikipedia`,
  `stackoverflow`, ...) is an independently trained model with its own weights
  and its own datasets. Experts share nothing at the weight level.
- **Meta-router.** A small classifier maps a prompt to a probability
  distribution over domains. It has a fast **n-gram** path (distilled from the
  transformer) that escalates to the transformer only when its confidence is
  low, so routing is cheap.
- **Model bank.** Only the selected experts are loaded into RAM. A byte-budgeted
  LRU keeps the working set bounded and evicts the rest; unrelated experts stay
  on disk.
- **Blended inference.** When the router selects several domains, each selected
  expert runs and their next-token logits are combined into one distribution, so
  a single token is still sampled once.

**Why.** The goal is dedicated, scalable CPU compute per expert model while
using only the RAM that the active experts actually need. A desktop can hold a
large on-disk library of domain experts and pull in only the relevant ones for a
request; adding a domain means training and registering one more expert, not
growing one giant model. Independent experts also mean independent scheduling:
each expert is a full model that can be run on its own share of the CPU.

> **Not the same as DeepSeekMoE.** DeepSeekMoE (implemented here too) routes
> *within* a layer's feed-forward block to fine-grained experts that all live in
> one model. Mixture-of-Experts Sharding routes *between whole models* at the
> bank level. The two compose: a bank expert can itself be a DeepSeekMoE model.

See [guides/07-moe-sharding.md](guides/07-moe-sharding.md) for the full design
and the end-to-end workflow.

---

## The model

gonano's transformer is a decoder-only stack with:

- **Rotary embeddings** (partial RoPE, deep-tail 64 dims), **QK-norm**, and a
  1.2 attention scale.
- **Grouped-query attention** (one KV head per `N` query heads) to cut KV size.
- **Value embeddings** (ResFormer-style) on alternating layers.
- **Untied** token embedding and `lm_head`, no biases anywhere, softcapped
  logits.

The `flash` preset turns on the full DeepSeek-V4.1-Flash long-context stack:

| Concept | What it does |
|:--|:--|
| Causal Encoder-Decoder (CED) | Decoder global KV is projected from the encoder's final hidden state; prefill reuses the encoder. |
| HCA dense KV compression | Merges each block of `ratio` key/value rows into one compressed entry. |
| CSA sparse attention | Each query attends to the top-`k` compressed blocks selected by a lightning indexer. |
| Cross-layer reuse (`FRU`) | `Full`/`Reindex`/`Reuse` layers share compressed KV and indexer work. |
| Hierarchical sparse indexer | Coarse-to-fine block selection keeps deeper indexing constant-cost. |
| Local sliding window (SWA) | Bounded local attention merged into the same softmax as the global branch. |
| DeepSeekMoE | One shared expert plus routed fine-grained experts per block. |
| Grouped-query attention / partial RoPE | Smaller KV cache and DeepSeek-style rotary. |
| MLA | Absorbed low-rank query/KV latent (the `latent` preset). |

Presets: `flash` (default, the DeepSeek-V4.1 long-context + MoE stack),
`latent` (absorbed MLA + MoE), and `dense` (the classic decoder).

---

## Quickstart

The `simd` package is gated behind the `goexperiment.simd` build tag, so
**every** build, test, and run must set it:

```bash
export GOEXPERIMENT=simd
```

Train a tiny model on synthetic data and chat with it:

```bash
go run ./cmd/base_train --depth 2 --max-seq-len 64 --num-iterations 20

go run ./cmd/chat_cli \
  --model ~/.cache/gonano/base_checkpoints/d2/model_000020.gn \
  --prompt "the capital of France is" \
  --max-tokens 24
```

The output is near-gibberish on a 20-step toy model -- the point is that the
whole pipeline works: train, save, load, infer.

Serve an OpenAI-compatible API:

```bash
go run ./cmd/server \
  --model ~/.cache/gonano/base_checkpoints/d2/model_000020.gn \
  --addr :8080

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gonano","messages":[{"role":"user","content":"what is 2+2?"}]}'
```

### Domain bank quickstart

Train one expert per domain, train the meta-router, register the bank, and serve
with routing:

```bash
# 1) Train a small expert per domain (datasets/ is a runnable example corpus).
go run ./cmd/base_train --domain physics --depth 4 --num-iterations 200 \
  --data-dir datasets/physics --data-format markdown
go run ./cmd/base_train --domain math --depth 4 --num-iterations 200 \
  --data-dir datasets/math --data-format markdown

# 2) Train the router (supervised fast path; --fine-tune adapts the encoder).
go run ./cmd/router_train \
  --domain physics=datasets/physics --domain math=datasets/math \
  --data-format markdown --out ~/.cache/gonano/router/router.gn

# 3) Register the bank.
go run ./cmd/bank_init \
  --domain physics=~/.cache/gonano/domains/physics/base_checkpoints/d4/model_000200.gn \
  --domain math=~/.cache/gonano/domains/math/base_checkpoints/d4/model_000200.gn \
  --router ~/.cache/gonano/router/router.gn \
  --out ~/.cache/gonano/bank/bank.json

# 4) Serve with per-turn routing and 2-way blending.
go run ./cmd/server --bank ~/.cache/gonano/bank/bank.json --max-domains 2 --addr :8080
```

The router's choice is returned in the `X-Gonano-Domains` response header, and
requesting a domain by name in the `model` field pins it.

---

## Guides

| Guide | Contents |
|:--|:--|
| [00-quickstart.md](guides/00-quickstart.md) | Copy-pasteable ArchLinux setup and smoke run. |
| [00-architecture-overview.md](guides/00-architecture-overview.md) | How the packages fit together, step by step. |
| [01-training.md](guides/01-training.md) | Data, tokenizer, pretraining, SFT, and per-domain training. |
| [02-export.md](guides/02-export.md) | The `.gn` checkpoint format and GGUF export. |
| [03-deployment.md](guides/03-deployment.md) | Loading weights, the Go API, and the OpenAI server. |
| [04-debugging.md](guides/04-debugging.md) | Symptom-to-file troubleshooting. |
| [05-deepseek-v4.1-optimizations.md](guides/05-deepseek-v4.1-optimizations.md) | The DeepSeek-V4.1 paper-to-code map. |
| [06-numeric-precision.md](guides/06-numeric-precision.md) | The float32 requirement and the low-bit decisions. |
| [07-moe-sharding.md](guides/07-moe-sharding.md) | **Mixture-of-Experts Sharding: design and workflow.** |
| [08-benchmarking.md](guides/08-benchmarking.md) | Reproducing prefill/decode benchmarks and the measured tables. |

---

## Packages

| Package | Responsibility |
|:--|:--|
| `kernels` | The numeric backend contract (`Backend`): elementwise, reductions, matmul, softmax/RMSNorm, flash attention, indexer, MoE. |
| `kernels/simd` | Production SIMD implementation (`AVX-512`/`AVX2`, goroutine-parallel). |
| `kernels/scalar` | Portable reference backend used for parity tests. |
| `tensors` | Dense float32/int32 tensor types plus high-level ops. |
| `model` | The transformer, its training forward/backward, and the DeepSeek-V4.1 features. |
| `model/layers`, `model/checkpoint` | Building blocks; versioned `.gn`/GGUF persistence. |
| `optimizer` | AdamW, Muon (Polar Express), Sinkhorn, and MuonAdamW. |
| `tokenizer` | Byte-level BPE training/inference and chat rendering. |
| `data` | Parquet/Markdown sources and BOS-aligned dataloaders. |
| `trainer` | Scaling laws, schedules, and the pretraining/SFT/RL/distillation loops. |
| `inference` | KV-cache engine, sampler, tools, and the blended `Ensemble`. |
| `router` | The domain meta-router: n-gram fast path + transformer classifier. |
| `bank` | The domain model bank: manifest, on-demand loading, LRU eviction. |
| `server` | OpenAI-compatible HTTP API, single-model or bank mode. |
| `evaluator`, `evaluator/tasks` | BPB, CORE, ChatCORE, and the task datasets. |
| `executor` | Sandboxed Python execution (HumanEval). |
| `internal/parallel`, `internal/device`, `internal/logging` | Worker pool, hardware detection, logging. |

Swap backends with `tensors.UseKernelBackend(scalar.New())`; the default is
`kernels/simd`.

## Testing

Every feature has unit tests, including scalar-vs-SIMD parity for the whole
kernel contract, numerical gradient checks of the backprop, and an end-to-end
domain-bank test that trains from `datasets/` and serves over HTTP:

```bash
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -short ./...   # skips the slow end-to-end test
```

## References

The architecture is inspired by, and links directly to:

- **DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression** --
  <https://arxiv.org/abs/2609.19969> (the long-context attention and KV stack).
- **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts
  Language Models** -- <https://arxiv.org/abs/2401.06066> (in-model routed
  experts).
- **Bag of Tricks for Efficient Text Classification (fastText)** --
  <https://arxiv.org/abs/1607.01759> (the n-gram fast path).
- **SetFit: Efficient Few-Shot Learning Without Prompts** --
  <https://arxiv.org/abs/2209.11055> (encoder plus classification head).
- **Efficient Intent Detection with Dual Sentence Encoders** --
  <https://arxiv.org/abs/2003.04807> (sentence-encoder routers).
- **RouteLLM: Learning to Route LLMs with Preference Data** --
  <https://arxiv.org/abs/2406.18665> (routing between whole models).

## License

MIT
