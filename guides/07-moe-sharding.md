# gonano -- Mixture-of-Experts Sharding

This guide describes **Mixture-of-Experts Sharding (MoE Sharding)**: training a
bank of independent domain models ("expert models"), routing a prompt to the
relevant ones with a small meta-router, loading only those into memory, and
blending their outputs during inference.

It is distinct from **DeepSeekMoE**
([arXiv:2401.06066](https://arxiv.org/abs/2401.06066)), which routes *within* a
single model's feed-forward blocks to fine-grained experts that are all resident
in that one model (see
[05-deepseek-v4.1-optimizations.md](05-deepseek-v4.1-optimizations.md)).
MoE Sharding routes *between whole models*. The two compose: a bank expert may
itself be a DeepSeekMoE model.

> **Setup.** Run everything from the repo root with `GOEXPERIMENT=simd`. See
> [00-quickstart.md](00-quickstart.md).

---

## 1. Why

A single large model must keep all of its weights resident and pays the same
per-token compute regardless of the question. MoE Sharding makes the working set
proportional to the question:

- **Dedicated, scalable CPU compute per expert.** Each expert is a full,
  independent model, so it can be scheduled on its own share of the CPU.
  Adding a capability means training and registering one more expert rather than
  growing one monolithic model.
- **Only the RAM the active experts need.** The bank holds the routed experts
  and evicts the rest; unrelated experts stay on disk. A desktop can keep a
  large on-disk library and pull in only what a request requires.
- **Independent training.** Experts are trained, updated, and versioned
  separately, on their own datasets, with no weight sharing.

The trade-off is explicit and bounded by configuration: blending `m` domains
runs `m` full models per step and holds `m` models resident.

Routing between whole models follows the cost-aware router idea of
[RouteLLM](https://arxiv.org/abs/2406.18665).

---

## 2. Architecture

```
prompt --> meta-router --> domain scores
                              |
                    model bank | (load + evict)
                              v
                    selected expert models --> blended generation --> tokens
```

### 2.1 Expert models

Every domain (`physics`, `math`, `cooking`, `trivia`, `wikipedia`,
`stackoverflow`, ...) is an independently trained checkpoint. Experts share no
weights. They cluster under
`$GONANO_BASE_DIR/domains/<name>/base_checkpoints/<tag>/model_<step>.gn` when
trained with `--domain`.

**Hard invariant.** All experts in a bank must share **one tokenizer and
`VocabSize`**, otherwise their logits cannot be combined. The bank validates
this at load time and rejects mismatches. Train the tokenizer once on a mix of
all domains and reuse it.

### 2.2 Meta-router (`router`)

The router maps a token sequence to a probability distribution over domains.
Its encoder-plus-classification-head design follows sentence-encoder intent
detection ([arXiv:2003.04807](https://arxiv.org/abs/2003.04807)) and
[SetFit](https://arxiv.org/abs/2209.11055); the n-gram fast path follows
[fastText](https://arxiv.org/abs/1607.01759).

- **Transformer classifier.** A `model.Transformer` encoder plus a linear head
  on the mean-pooled final hidden state. `Train` trains the head over a frozen
  encoder (a fast linear probe); `FineTune` backpropagates the classification
  loss into the encoder via `model.Transformer.TrainClassification`.
- **Distilled n-gram fast path.** A hashed unigram/bigram/trigram linear softmax
  classifier. It is trained supervised (the domain directories are the labels)
  by default, or distilled from the transformer teacher with `--distill`.
- **Confidence-gated escalation.** `Router.Predict` runs the fast path and, when
  its top-1/top-2 margin falls below `Threshold`, escalates to the transformer
  for that request.

### 2.3 Model bank (`bank`)

A JSON manifest lists the domain checkpoints. The bank loads on demand and
evicts by LRU within a byte and/or model-count budget. `Acquire` returns a
model pointer (the Go GC keeps it alive even if the bank later evicts its
entry); `AcquireMany` loads a routed set.

### 2.4 Blended inference (`inference.Ensemble`)

Every selected expert keeps its own KV cache. At each step each expert runs a
full forward and the next-token logits are combined as a weighted sum, so one
token is sampled once from the blended distribution. Thinking-budget handling
and tool calls work through the blended path.

### 2.5 Serving (`server`, `cmd/server`)

The OpenAI-compatible server can run in bank mode. Per request it classifies the
prompt, loads the routed experts, and blends. Two routing controls are exposed:

- `--route-scope last-turn` (default) classifies only the latest user message;
  `full-prompt` classifies the whole rendered prompt.
- Requesting a domain by name in the `model` field pins that domain (bypassing
  the classifier).
- The routed domains are returned in the `X-Gonano-Domains` response header.

---

## 3. End-to-end workflow

### 3.1 Prepare datasets

Each domain is a directory of documents. `datasets/` contains a runnable example
corpus for `physics`, `math`, `cooking`, and `trivia`, written to expose
deliberate topic overlaps; see [datasets/README.md](../datasets/README.md).

### 3.2 Train a shared tokenizer

Train one tokenizer on a mix of all domains and place it at
`$GONANO_BASE_DIR/tokenizer/tokenizer.json` (see
[01-training.md](01-training.md)). Every expert must use it.

### 3.3 Train one expert per domain

```bash
go run ./cmd/base_train \
  --domain physics \
  --data-dir ~/data/physics --data-format markdown \
  --depth 12 --max-seq-len 8192 --num-iterations 200

go run ./cmd/base_train \
  --domain math \
  --data-dir ~/data/math --data-format markdown \
  --depth 12 --max-seq-len 8192 --num-iterations 200
```

`--domain` writes to `domains/<name>/base_checkpoints/` and records the domain
in the checkpoint metadata. The same flag records the domain (and defaults the
output directory under `domains/<name>/`) on the post-training/alignment
commands: `chat_sft`, `chat_rl`, `chat_opd`, and `chat_eval`.

### 3.4 Train the meta-router

```bash
go run ./cmd/router_train \
  --domain physics=~/data/physics \
  --domain math=~/data/math \
  --data-format markdown \
  --out ~/.cache/gonano/router/router.gn
```

The command collects labeled examples from each domain directory, trains a small
encoder LM if `--encoder` is not given, trains the head, and writes the bundle.
Useful flags:

| Flag | Meaning |
|:--|:--|
| `--distill` | Train the n-gram fast path from the transformer teacher instead of supervised labels. |
| `--fine-tune` | Fine-tune the encoder jointly with the head (`--fine-tune-epochs`, `--fine-tune-lr`). |
| `--ngram-only` | Train only the fast path. |
| `--max-examples-per-domain` | Examples sampled per domain (default 2000). |
| `--max-seq-len` | Router input truncation (default 128). |

### 3.5 Write the bank manifest

```bash
go run ./cmd/bank_init \
  --domain physics=~/.cache/gonano/domains/physics/base_checkpoints/d12/model_000200.gn \
  --domain math=~/.cache/gonano/domains/math/base_checkpoints/d12/model_000200.gn \
  --data physics=~/data/physics --data math=~/data/math \
  --tokenizer ~/.cache/gonano/tokenizer/tokenizer.json \
  --router ~/.cache/gonano/router/router.gn \
  --out ~/.cache/gonano/bank/bank.json
```

### 3.6 Serve

```bash
go run ./cmd/server \
  --bank ~/.cache/gonano/bank/bank.json \
  --max-domains 2 --route-scope last-turn --addr :8080
```

`--max-domains` caps the blend size; `--min-domain-score` sets the minimum
probability for an extra domain. `--router` overrides the manifest's router.
With no `--bank`, the server is the ordinary single-model server.

---

## 4. Programmatic API

```go
// Router: load a bundle (its encoder is loaded automatically).
domainRouter, err := router.LoadRouterAuto("router.gn")
scores := domainRouter.TopDomains(tokenIDs, 2, 0.0) // top-2 domains

// Bank: load a manifest and acquire the routed experts.
manifest, _ := bank.LoadManifest("bank.json")
domainBank := bank.NewBank(manifest, tok, bank.BankOptions{MaxBytes: 48 << 30})
models, ids, _ := domainBank.AcquireMany([]string{"physics", "math"})

// Ensemble: blend the acquired experts.
weights := make([]inference.ModelWeight, len(models))
for index := range models {
    weights[index] = inference.ModelWeight{Model: models[index], Weight: 0.5, Domain: ids[index]}
}
ensemble := inference.NewEnsemble(weights, tok)
results, _ := ensemble.GenerateBatch(prompt, 1, 64, 0.7, 50, 42)
```

Training lives in `router.TransformerClassifier` (`Train`, `FineTune`,
`DistillNgram`) and `router.TrainNgram`; persistence is shared with the rest of
gonano through `model/checkpoint`.

---

## 5. Memory and CPU behavior

- **Resident set** = the bank's current experts (bounded by `BankOptions`) plus
  the router and tokenizer. Eviction only shrinks the bank's retained set.
- **Per-token cost** scales with the blend size `m`: `m` forwards and `m` KV
  caches. Keep `--max-domains` small on memory-constrained hosts.
- **Cold shard loads** currently read a whole checkpoint file
  (`checkpoint.Load` uses `os.ReadFile`); mmap loading and a shared mixed-domain
  KV cache are future work.
- **Shared vocabulary is mandatory** and enforced at bank load.

---

## 6. Testing

- `router/router_test.go` -- fast-path learning, head training, distillation
  fidelity, save/load, escalation, encoder fine-tuning.
- `bank/bank_test.go` -- manifest round-trip, tokenizer-mismatch rejection,
  acquire, eviction.
- `inference/ensemble_test.go` -- a one-member blend is bit-identical to the
  plain engine; a two-member blend is finite and deterministic.
- `model/gradcheck_test.go` -- `TestBackpropClassificationDirectionalGradientCheck`
  validates the fine-tuning backward path against finite differences.
- `datasets_e2e_test.go` -- trains experts, router, and bank from `datasets/`,
  checks routing (including overlap probes and per-turn re-routing), blends, and
  drives the HTTP server. Run with `go test ./...` (skipped by `-short`).

See [08-benchmarking.md](08-benchmarking.md) for measuring blended throughput.

---

## 7. Findings and caveats

- **Router context matters.** In the `datasets/` fixture, truncating documents to
  48 subword tokens hid distinguishing vocabulary ("entropy" never reached
  training) and the router preferred the overlapping "heat" topic. Raising the
  truncation fixed it; size the router context to cover the discriminative text.
- **Supervised beats unsupervised when labels exist.** Because the domain
  directories are labeled, `cmd/router_train` trains the fast path supervised by
  default and reserves `--distill` for the unlabeled case.
- **Blending is an ensemble, not per-token routing.** Every selected expert runs
  on every step; this is the cost of combining models whose weights cannot be
  merged.
- **Per-turn drift.** Routing on the latest user turn handles a topic change
  between turns, but a single turn that spans domains is served by the blend,
  not by per-token switching.
