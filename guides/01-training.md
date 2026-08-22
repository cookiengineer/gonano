# gonano — Training Guide

This guide walks through training a gonano model end to end, from raw text to a
talking model. It focuses on the two most common data pipelines:

1. **Off-the-shelf web corpus** — download a pretraining dataset (we use
   [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), the
   standard "base English understanding" corpus).
2. **Your own webdata encoded as Markdown** — crawl pages, convert HTML to
   Markdown, and train directly on the `.md` files.

Every command below must run with the `simd` build experiment enabled (see
`README.md`):

```bash
export GOEXPERIMENT=simd
```

> **New to the project / on ArchLinux?** Start with
> [00-quickstart.md](00-quickstart.md) for the copy-pasteable install
> (Go 1.27 + `GOEXPERIMENT=simd` + clone + smoke run). The rest of this guide
> assumes you are in the repo root with `GOEXPERIMENT=simd` exported.

---

## 0. Prerequisites

- **Go 1.27** or newer (the `simd` package is gated behind `goexperiment.simd`).
- Disk space for the dataset (a "sample-10BT" subset is ~10 GB; the full
  FineWeb-Edu corpus is terabytes).
- Optionally, many CPU cores (training parallelizes across all cores
  automatically via the `parallel` package).

Verify the toolchain works:

```bash
GOEXPERIMENT=simd go test ./...
```

---

## 1. The pipeline at a glance

```
┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐
│ raw text /   │ → │  tokenizer    │ → │  pretraining │ → │  finetuning  │
│ markdown     │   │  (tok_train)  │   │ (base_train) │   │ (chat_sft)   │
└──────────────┘   └───────────────┘   └──────────────┘   └──────────────┘
        │                                    │
        │                                    ├── evaluate (base_eval)
        │                                    └── export (export → GGUF)
        └────────────────────────────────────┴── deploy (chat_cli)
```

There is **one complexity dial** — `--depth`, the number of transformer
layers. Everything else (width, heads, batch size, learning rates, training
horizon, weight decay) is derived automatically from the scaling laws in
`train/scaling.go` (`DeriveHyperparams`).

---

## 2. Getting a base English dataset

gonano's pretraining loader (`data.NewPretrainLoader`) reads **Parquet shards
with a `text` column**, or **Markdown files**. Two ways to get data:

### 2a. Download FineWeb-Edu (recommended base corpus)

FineWeb-Edu is the standard, freely-licensed English web corpus used to teach a
model general English. It ships as Parquet shards, each with a `text` column —
exactly what gonano consumes.

Pick a size:

| Config | Approx size | Good for |
|---|---|---|
| `sample-10BT` | ~10 GB | smoke tests, small models |
| `sample-100BT` | ~100 GB | real experiments |
| `default` | ~TB | full pretraining |

Download shards with the built-in downloader:

```bash
go run ./cmd/dataset \
  --repo HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --num 20 \
  --out ~/.cache/gonano/base_data
```

> `--num 20` downloads 20 shards. The loader cycles through whatever shards are
> present, so you can train on a prefix of the corpus and download more later.
> The `dataset` command uses `data.ListHFParquetShards` + `data.DownloadFile`
> (`data/hub.go`).

### 2b. Use your own webdata encoded as Markdown

If you already crawl the web and convert HTML to Markdown, you can skip the
Parquet step entirely. Point `base_train` at a directory of `.md` files with
`--data-format markdown`:

```bash
mkdir -p ~/webdata-md
# one document per file, e.g. your crawler writes one .md per page
cat > ~/webdata-md/article-1.md <<'EOF'
# What is a neural network?
A neural network is a function composed of many simple units, each of which
applies a linear transform followed by a nonlinearity.
EOF
```

Rules for Markdown data:

- One `.md` file = one document (the whole file, including Markdown syntax, is
  fed to the tokenizer as-is).
- Files are discovered recursively and sorted by path.
- A UTF-8 BOM is stripped automatically.
- Long pages should be split by you (e.g. by heading) before training; gonano
  does not split them.

Implementation: `data.NewMarkdownSource` in `data/markdown.go`.

---

## 3. Training the tokenizer

The tokenizer is a byte-level BPE tokenizer in the style of GPT-4
(`tokenizer.TrainBPE` + the GPT-4 split pattern in `tokenizer/splitter.go`).

Train it on the same data you will train the model on:

```bash
# From Parquet shards:
go run ./cmd/tok_train \
  --data-dir ~/.cache/gonano/base_data \
  --vocab-size 32768
```

Or let `trainer.sh` handle the tokenizer for you — it trains one on the data,
loads an existing one, or copies a bundled default (`tokenizer/defaults/`:
`markdown.json` for Markdown, `byte.json` otherwise):

```bash
# Train a tokenizer on Markdown webdata and then train a model:
./trainer.sh ~/webdata-md --format markdown --train-tokenizer --depth 4

# Train a tokenizer on Parquet shards and then train a model:
./trainer.sh ~/.cache/gonano/base_data --format parquet --train-tokenizer --depth 4

# Use the bundled default tokenizer (no training) if none exists yet:
./trainer.sh ~/webdata-md --format markdown --depth 4
```

The underlying primitive is `data.TrainTokenizer` (format-agnostic), which
feeds `tokenizer.SplitPieces` + `tokenizer.TrainBPE`.

Check the result:

```bash
go run ./cmd/tok_eval
```

The tokenizer is saved to `$GONANO_BASE_DIR/tokenizer/tokenizer.json` (default
`~/.cache/gonano/tokenizer/tokenizer.json`).

> **Vocabulary note.** nanochat special tokens are fixed:
> `<|bos|>`, `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`,
> `<|assistant_end|>`, `<|python_start|>`, `<|python_end|>`,
> `<|output_start|>`, `<|output_end|>`.
> They are appended *after* the mergeable vocabulary, so
> `vocab_size = num_merges + 256 + 9`.

---

## 4. Pretraining

Train the base model. The **only required dial is `--depth`**:

```bash
# Quick smoke run (minutes):
go run ./cmd/base_train \
  --depth 4 \
  --max-seq-len 512 \
  --num-iterations 200 \
  --data-dir ~/.cache/gonano/base_data

# On Markdown webdata:
go run ./cmd/base_train \
  --depth 4 \
  --max-seq-len 512 \
  --num-iterations 200 \
  --data-dir ~/webdata-md \
  --data-format markdown

# A larger, more capable run:
go run ./cmd/base_train \
  --depth 20 \
  --max-seq-len 2048 \
  --num-iterations 20000 \
  --data-dir ~/.cache/gonano/base_data
```

Key flags (`cmd/base_train/main.go`):

| Flag | Meaning | Default |
|---|---|---|
| `--depth` | number of layers; sets width/heads automatically | 12 |
| `--max-seq-len` | context length | 512 |
| `--num-iterations` | optimization steps | 50 |
| `--device-batch-size` | sequences per step | 1 |
| `--total-batch-size` | tokens per step (auto from scaling laws) | auto |
| `--data-dir` | Parquet or Markdown directory | synthetic |
| `--data-format` | `parquet` or `markdown` | parquet |
| `--base-dir` | checkpoint/tokenizer directory | `~/.cache/gonano` |
| `--model-tag` | checkpoint subdirectory | `d<depth>` |

What happens under the hood (`train/trainer.go`, `train/scaling.go`):

1. `model.ConfigForDepth(depth, vocab, 64, 128, seqLen, "SSSL")` computes
   `n_embd = depth*64` (rounded up to a multiple of the 128-dim head).
2. `train.DeriveHyperparams` computes the total batch size
   (`B ∝ D^0.383`, Power Laws), the LR scaling (`∝ √B`), and weight decay
   (T-epoch framework).
3. `model.SetupOptimizer` routes embeddings/scalars to AdamW and 2-D matrices
   to Muon (`optim/muon.go`).
4. Each step runs `Trainer.TrainStep` (forward + backward) and
   `Trainer.StepOptimizer` (LR/momentum/weight-decay schedules).

Checkpoints are written to
`$GONANO_BASE_DIR/base_checkpoints/<tag>/model_<step>.gn`.

### Resuming / scaling up

`base_train` currently trains from scratch. To resume, note the checkpoint step
and restart with a higher `--num-iterations` — the scaling laws are
deterministic, so re-running with the same `--depth` reproduces the same
hyperparameters. Full resume-from-step is on the roadmap.

---

## 5. Evaluate the base model

```bash
go run ./cmd/base_eval --model ~/.cache/gonano/base_checkpoints/d4/model_000200.gn
```

This prints samples and (with `--data-dir`) the bits-per-byte metric
(`eval/bpb.go`). The DCLM CORE benchmark (`eval/core.go`) requires downloading
the evaluation bundle — see the debugging guide for how it is invoked.

---

## 6. Finetune for chat (SFT)

Teach the base model conversation special tokens and tool use:

```bash
go run ./cmd/chat_sft \
  --model ~/.cache/gonano/base_checkpoints/d4/model_000200.gn \
  --out ~/.cache/gonano/chatsft_checkpoints/d4/model_000200.gn
```

SFT uses the best-fit packing loader (`data.NewSFTLoader`) with a loss mask so
the model is only supervised on assistant completions. Production SFT loads
SmolTalk + MMLU + GSM8K via the `eval/tasks` package; the `chat_sft` command
uses synthetic conversations for demonstration.

---

## 7. What "good" looks like

- **Loss decreasing** — pretraining loss should fall smoothly; a depth-4 smoke
  run drops from ~5.5 to well below 1 on a tiny corpus.
- **No NaNs** — see the debugging guide if you see `NaN`/`Inf`.
- **Samples become coherent** — after enough data, `chat_cli` produces
  plausible continuations.
- **Throughput scales with batch size** — `infer_bench` shows higher tok/s at
  larger decode batch sizes (goroutine-per-op parallelism).

Next steps: [Export guide](02-export.md) and [Deployment guide](03-deployment.md).
