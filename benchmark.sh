#!/usr/bin/env bash
#
# Reproducible, self-contained inference benchmark.
#
# Every run starts from a fresh randomized /tmp directory, builds the binaries,
# trains a tiny model on synthetic data, and benchmarks it — so there is no
# shared state between runs and the results are reproducible.
#
# Tune the knobs below (or override on the command line), e.g.:
#   DEPTH=8 NUM_ITERATIONS=200 BATCH_SIZES=1,4,32 ./benchmark.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"   # so `go build ./cmd/...` works from any cwd

# ---- configuration (set variables) -----------------------------------------

# Required for the experimental `simd` package.
export GOEXPERIMENT=simd

DEPTH="${DEPTH:-4}"                    # transformer depth (the complexity dial)
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"       # context length for the trained model
NUM_ITERATIONS="${NUM_ITERATIONS:-50}" # training steps
BATCH_SIZES="${BATCH_SIZES:-1,4,16}"   # decode batch sizes to sweep
DECODE_TOKENS="${DECODE_TOKENS:-64}"   # tokens to generate per row

# ---- fresh, randomized work directory --------------------------------------

WORKDIR="$(mktemp -d)"
export GONANO_BASE_DIR="$WORKDIR"      # keeps model/tokenizer inside the temp dir
trap 'rm -rf "$WORKDIR"' EXIT          # clean up on exit

echo "==> work directory: $WORKDIR"

# ---- prerequisites ----------------------------------------------------------

echo "==> building binaries ..."
go build -o "$WORKDIR/base_train" ./cmd/base_train
go build -o "$WORKDIR/infer_bench" ./cmd/infer_bench

echo "==> training a tiny model (depth=$DEPTH, steps=$NUM_ITERATIONS, synthetic data) ..."
"$WORKDIR/base_train" \
  --depth "$DEPTH" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --num-iterations "$NUM_ITERATIONS"

MODEL="$GONANO_BASE_DIR/base_checkpoints/d$DEPTH/model_$(printf '%06d' "$NUM_ITERATIONS").gn"

# ---- benchmark --------------------------------------------------------------

echo "==> benchmarking $MODEL ..."
"$WORKDIR/infer_bench" \
  --model "$MODEL" \
  --batch-sizes "$BATCH_SIZES" \
  --decode-tokens "$DECODE_TOKENS"
