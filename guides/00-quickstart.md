# gonano — Quick Start (ArchLinux)

A copy-pasteable path from a fresh ArchLinux host to a talking model. Every
command assumes a normal user shell; run them in order. The full guide lives in
[01-training.md](01-training.md) — this file gets you set up and proves the
toolchain works end to end.

---

## 1. Install Go 1.27

gonano depends on the **experimental `simd` standard-library package**, which
only exists in **Go 1.27** (gated behind `goexperiment.simd`). Arch's `go`
package may be older, so check and fall back to the official tarball.

```bash
# Try the distro package first:
sudo pacman -S --needed go
go version
```

If `go version` reports **1.27.x**, skip the next block. Otherwise install the
official release into `/usr/local/go`:

```bash
sudo pacman -S --needed wget
wget https://go.dev/dl/go1.27.0.linux-amd64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.27.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
go version        # must print "go version go1.27.x"
```

> If `1.27.0` has been superseded, substitute the latest `1.27.x` from
> <https://go.dev/dl/> (the archive is `go1.27.x.linux-amd64.tar.gz`).

## 2. Enable the `simd` experiment

This flag must be set for **every** build, test, and run:

```bash
# For the current shell:
export GOEXPERIMENT=simd

# Or persist it so you don't have to remember it again:
echo 'export GOEXPERIMENT=simd' >> ~/.bashrc
```

## 3. Get and build gonano

```bash
git clone https://github.com/cookiengineer/gonano.git
cd gonano
export GOEXPERIMENT=simd
go test ./...          # compiles and runs the whole test suite
```

All later commands assume you are **in the `gonano` repo root** with
`GOEXPERIMENT=simd` exported.

## 4. Smoke run (no dataset, ~1 minute)

Train a tiny model on synthetic text, then chat with it:

```bash
go run ./cmd/base_train --depth 2 --max-seq-len 64 --num-iterations 20

go run ./cmd/chat_cli \
  --model ~/.cache/gonano/base_checkpoints/d2/model_000020.gn \
  --prompt "the capital of France is" \
  --max-tokens 24
```

The output will be near-gibberish (it's a 20-step toy model) — the point is
that **the whole pipeline works**: train → save → load → infer.

## 5. Train on a real base English corpus (FineWeb-Edu)

```bash
# 1) Download 10 shards of the "sample-10BT" subset (~a few GB):
go run ./cmd/dataset \
  --repo HuggingFaceFW/fineweb-edu \
  --config sample-10BT \
  --split train \
  --num 10

# 2) Train the BPE tokenizer on those shards:
go run ./cmd/tok_train \
  --data-dir ~/.cache/gonano/base_data \
  --vocab-size 32768

# 3) Pretrain (the --depth dial sets width/heads/batch/LRs automatically):
go run ./cmd/base_train \
  --depth 4 \
  --max-seq-len 512 \
  --num-iterations 500 \
  --data-dir ~/.cache/gonano/base_data

# 4) Chat:
go run ./cmd/chat_cli \
  --model ~/.cache/gonano/base_checkpoints/d4/model_000500.gn \
  --prompt "why is the sky blue?"
```

## 6. Train on your own webdata encoded as Markdown

If your crawler emits one `.md` file per page, skip Parquet entirely:

```bash
go run ./cmd/base_train \
  --depth 4 \
  --max-seq-len 512 \
  --num-iterations 500 \
  --data-dir ~/webdata-md \
  --data-format markdown
```

## 7. Export to GGUF and run it

```bash
go run ./cmd/export \
  --model ~/.cache/gonano/base_checkpoints/d4/model_000500.gn \
  --out d4.gguf

go run ./cmd/chat_cli --model d4.gguf --prompt "hello there"
```

## Next

- [01 — Training guide](01-training.md) (data, tokenizer, scaling laws)
- [02 — Export guide](02-export.md) (`.gn` and GGUF)
- [03 — Deployment guide](03-deployment.md) (loading + inference, Go API)
- [04 — Debugging guide](04-debugging.md) (symptom → file to look at)
