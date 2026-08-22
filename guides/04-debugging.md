# gonano — Debugging Guide

Where to look when something goes wrong. For every failure mode, this gives the
symptom, the likely cause, and the exact file/function to inspect.

> **Setup.** Run everything from the repo root with `GOEXPERIMENT=simd`. The
> [00-quickstart.md](00-quickstart.md) covers the ArchLinux install.

---

## 1. Quick reference table

| Symptom | Likely cause | Look in |
|---|---|---|
| `build constraints exclude all Go files` | `GOEXPERIMENT=simd` not set | `go.mod`, `README.md` |
| `package simd is not in GOROOT` / simd link errors | Go < 1.27 or experiment off | `go env GOROOT`, `GOEXPERIMENT` |
| `no trained tokenizer, using byte-level tokenizer` | tokenizer not trained / wrong `--base-dir` | `cmd/base_train/main.go`, `tokenizer.LoadTokenizer` |
| `no parquet files found` / `no .md files found` | wrong `--data-dir`, or `--data-format` mismatch | `cmd/base_train/main.go`, `data.ListParquetFiles`, `data.NewMarkdownSource` |
| `parquet: corrupt …` / `unsupported encoding/compression` | shard not the expected schema | `data/parquet/reader.go`, `data/parquet/encoding.go`, `data/parquet/snappy.go` |
| `sequence longer than rotary cache` | prompt or `max-seq-len` exceeds the trained context | `model/transformer.go` (`Forward`) |
| `tensor: index … out of range` in `nn.Embedding.Forward` | token id ≥ vocab size (model/tokenizer mismatch) | `nn/linear.go` (`Embedding.Forward`), `model/config.go` |
| `loss` becomes `NaN` / `Inf` | optimizer or backprop bug, too-high LR | `optim/muon.go`, `optim/optimizer.go`, `model/train.go` |
| Training very slow / high CPU | matmul not tiling well, tiny batch | `tensor/matmul.go`, `parallel/pool.go` |
| `checkpoint: invalid magic` | wrong file (not a `.gn`), truncated download | `checkpoint/checkpoint.go` |
| `checkpoint: truncated …` | file cut short / version mismatch | `checkpoint/checkpoint.go` |
| GGUF rejected by a tool | not a stock arch, or corrupt | `checkpoint/gguf.go` |
| Empty / garbage generation | temperature=0 with bad tokenizer, or model not trained | `infer/engine.go`, `infer/sampler.go`, `tokenizer` |
| Tool call silent | expression unsupported by any registered tool | `infer/tooluse.go` (`Tool`, `Registry`) |
| HumanEval always 0% | `python3` missing on host | `exec/exec.go` (`Available`) |

---

## 2. Build & environment failures

**Symptom:** `go build ./...` → `package simd: build constraints exclude all Go files`.

**Cause:** the `simd` standard-library package is experimental and gated behind
`goexperiment.simd` (see `doc.go` in GOROOT/src/simd). Every command needs it:

```bash
GOEXPERIMENT=simd go build ./... && GOEXPERIMENT=simd go test ./...
```

There is no `go.mod` trick for this — it is a toolchain experiment flag. Add it
to your shell profile, `Makefile`, or CI environment.

**Symptom:** builds work but `go test` panics with a `simd`/`archsimd` stack trace
(e.g. `cannot convert slice with length … to array … length 16`).

**Cause:** the vector width is **runtime-selected** (128/256/512-bit depending
on CPU). gonano's kernels are width-agnostic (`tensor/simdutil.go` reads
`simd.VectorBitSize()`); a fixed-width panic indicates you are calling a raw
`simd.Load…` with a too-short slice outside the `tensor` package. Use
`tensor` kernels or `Load…Part`/`Store…Part`.

---

## 3. Data & tokenizer failures

### 3.1 "no parquet files found" / "no .md files found"

`cmd/base_train/main.go` selects the source from `--data-dir` + `--data-format`:

- `parquet` (default) → `data.ListParquetFiles` looks for `*.parquet` (ignoring
  `*.tmp`). Each file must have a **flat `text` BYTE_ARRAY column**.
- `markdown` → `data.NewMarkdownSource` walks for `*.md` recursively.

If your webdata is Markdown but you passed `--data-dir` without
`--data-format markdown`, you get "no parquet files found". If your shards are
Parquet but you set `--data-format markdown`, you get "no .md files found".

### 3.2 Parquet read errors

The reader is minimal by design (`data/parquet/`). It supports **PLAIN** and
**RLE_DICTIONARY** encodings for **BYTE_ARRAY / INT32 / INT64**, with **SNAPPY**
or no compression. Anything else fails loudly with a specific message:

- `unsupported compression codec N` → `decompress` in `reader.go` (only 0/1).
- `unsupported encoding N` → `readColumnValues` / `ReadColumnInt64`.
- `parquet: corrupt …` → `compact.go` (Thrift metadata) or `snappy.go`.

If you see "unsupported encoding", the shard likely uses a nested/list column
or DELTA encoding — re-export it with a flat `text` column, or feed the data as
Markdown instead.

### 3.3 Tokenizer / vocab mismatch

**Symptom:** `panic: tensor: index … out of range` inside `nn.Embedding.Forward`
during training or inference.

**Cause:** the model's `VocabSize` does not match the tokenizer's. The model was
built with a config whose `vocab_size` differs from `tokenizer.VocabSize()`.
`base_train` sets `--vocab-size` to the tokenizer's size, so this only happens
when you hand-build a model. Check the invariant:

```go
if model.Config.VocabSize != tok.VocabSize() { /* mismatch */ }
```

The reference `build_model` in nanochat asserts exactly this; gonano's
`checkpoint.LoadModel` rebuilds from the stored config, so a `.gn` always
round-trips — but a hand-written loader must match vocab sizes.

---

## 4. Model / sequence-length failures

**Symptom:** `panic: model: sequence longer than rotary cache`.

**Cause:** `model.Transformer.Forward` rejects sequences longer than
`Config.SequenceLen` (the rotary tables are precomputed for
`SequenceLen * 10` positions). This happens when:

- `infer_bench --prompt-tokens` + `--decode-tokens` exceed the trained context,
- a prompt is longer than `--max-seq-len` used at training time,
- the dataloader's `T` (`--max-seq-len`) is larger than the config's.

The fix is always to clamp input length to the model's `SequenceLen`. The
`infer_bench` command already clamps; a custom loader should too.

**Symptom:** `model: training forward requires T > 1` during a custom training
loop.

**Cause:** `model.TrainForward` (backprop path) requires a sequence of length
> 1. You called `TrainStep` with a single-token batch. Use a batch size ≥ 2.

---

## 5. NaN / Inf loss

This is the most serious failure. Work through the layers in order:

1. **Too-high learning rate.** Reduce `--depth`'s implied LRs, or the matrix LR.
   Muon's Polar-Express iteration (`optim/muon.go`) can diverge if the update
   scale is too large.
2. **Backprop bug.** The backprop is verified by numerical gradient checks in
   `model/gradcheck_test.go` (`TestBackpropDirectionalGradientCheck` and
   `TestBackpropPerElementGradientCheck`). Run them:
   ```bash
   GOEXPERIMENT=simd go test ./model -run TestBackprop -v
   ```
3. **Optimizer state.** `optim/optimizer.go` (AdamW) and `optim/muon.go` (Muon)
   accumulate moments in float32; a NaN there propagates forever. Check for a
   division by zero in the RMSNorm/softmax (`tensor/norm.go`) — the epsilon is
   `1e-6`.
4. **Data.** A target id outside `[0, vocab)` or a `-1` that isn't masked
   correctly can inject garbage into `tensor.CrossEntropyGrad`
   (`tensor/backward.go`).

Bisect with a **tiny model**: depth 1–2, a few tokens, `--num-iterations 5`,
and print `loss` every step. If it NaNs immediately, the bug is in the
forward/backward; if it NaNs later, it's the optimizer or learning rate.

---

## 6. Performance problems

- **Matmul dominates** — see `tensor/matmul.go` (`gemmTransB`/`gemmNN`). They
  tile over rows (block size 64) and parallelize across `parallel.Pool`.
  Verify `parallel.Default().Workers() == runtime.GOMAXPROCS(0)`.
- **Tiny batches** — decode is bandwidth-bound; batching helps (see the
  `infer_bench` guide).
- **Too many goroutines** — the `parallel` pool uses `GOMAXPROCS`; don't nest
  `parallel.For` inside another parallel loop.
- **GC pressure** — the training loop allocates activations per step; the
  reference nanochat freezes the GC. gonano's `train/trainer.go` does not yet
  do this (roadmap).

---

## 7. Checkpoint & export failures

- `checkpoint: invalid magic in …` → you passed a file that isn't a `.gn`
  (or a corrupted/truncated file). `checkpoint.Load` checks the 8-byte magic
  `GONANO\x00\x01`.
- `checkpoint: truncated …` → the file is short. Re-download / re-save.
- GGUF rejected → the file is a *nanochat* container, not a stock llama
  architecture. See the export guide's architectural note. The writer itself is
  validated in `checkpoint/gguf_test.go`.

---

## 8. Inference failures

- **Empty/garbage output** — the model may be under-trained, or the tokenizer
  doesn't round-trip. Check `tokenizer` round-trips
  (`go test ./tokenizer`), then verify `engine.GenerateBatch` equals a naive
  `model.Forward` loop (`infer/infer_test.go`, `TestEngineMatchesNaiveGenerate`).
- **Top-k/temperature odd behavior** — `infer/sampler.go` (`sampleRow`,
  `maskTopK`, `kthLargest`). temperature ≤ 0 is argmax; top-k masks everything
  below the k-th logit.
- **Tool use never fires** — `infer/engine.go` looks for `<|tool_start|>` /
  `<|tool_end|>` and dispatches the enclosed text to the `infer.Tool` registry
  (`infer/tooluse.go`), which by default contains only the calculator
  (pure arithmetic or `"str".count("sub")` expressions). Register more tools
  via `Registry.Register` to extend it.
- **HumanEval 0%** — `exec.Available()` reports whether `python3` is on `PATH`;
  if not, `HumanEval.Evaluate` returns false (`eval/tasks/humaneval.go`).

---

## 9. Where the invariants live

For reference when writing custom loaders or debugging:

| Invariant | Enforced in |
|---|---|
| `n_embd % n_head == 0`, `n_kv_head` divides `n_head` | `model.Config.Validate` (`model/config.go`) |
| vocab padded to multiple of 64 | `model.Config.PaddedVocab` |
| tokenizer vocab == model vocab | caller (`checkpoint.LoadModel` relies on the stored config) |
| rotary cache covers `SequenceLen` | `model.NewTransformer` (`rotaryOvercompute = 10`) |
| KV cache head dim == `n_embd/n_head` | `model.KVBuffer` |
| GGUF magic/version/alignment | `checkpoint/gguf.go` |

If you hit a problem not covered here, open an issue with the exact panic
message, the `--depth`/`--max-seq-len`/`--data-format` flags, and the last few
log lines — the `logging` package (`logging/logging.go`) emits leveled
`slog` output that makes this easy to include.
