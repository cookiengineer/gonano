#!/usr/bin/env bash
#
# Long-context inference benchmark for the DeepSeek-V4.1-Flash optimization
# stack (dense compression + sparse attention + hierarchical indexer +
# cross-layer reuse + sliding-window attention + CED).
#
# It trains a tiny model with a given architecture config (random-init weights
# are enough for latency/throughput measurement; set STEPS=0 to skip training),
# then benchmarks prefill TTFT and decode throughput at long context.
#
# Usage:
#   ./benchmark_longctx.sh                       # flash preset, seq 4096+16384
#   SEQS="4096,16384" BATCH_SIZES="1,16,64" ./benchmark_longctx.sh
#   CONFIG="--preset dense" ./benchmark_longctx.sh
#   STEPS=50 ./benchmark_longctx.sh              # actually train a few steps

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export GOEXPERIMENT=simd

DEPTH="${DEPTH:-4}"
SEQS="${SEQS:-4096,16384}"
PROMPT_RATIO="${PROMPT_RATIO:-1}"            # prompt = ratio * seq (clamped by context)
BATCH_SIZES="${BATCH_SIZES:-1,16,64}"
DECODE_TOKENS="${DECODE_TOKENS:-16}"
STEPS="${STEPS:-0}"

# Architecture preset under test. Override with the CONFIG environment variable
# (for example CONFIG="--preset dense" or CONFIG="--preset latent").
CONFIG="${CONFIG:---preset flash}"

WORKDIR="$(mktemp -d)"
export GONANO_BASE_DIR="$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

echo "==> work directory: $WORKDIR"
echo "==> config: $CONFIG"

echo "==> building binaries ..."
go build -o "$WORKDIR/base_train" ./cmd/base_train
go build -o "$WORKDIR/infer_bench" ./cmd/infer_bench

echo "==> training tiny models (depth=$DEPTH, steps=$STEPS, synthetic data) ..."
for SEQ in ${SEQS//,/ }; do
  TAG="lc_seq${SEQ}"
  "$WORKDIR/base_train" \
    --depth "$DEPTH" \
    --max-seq-len "$SEQ" \
    --num-iterations "$STEPS" \
    --model-tag "$TAG" \
    $CONFIG >/dev/null
  MODEL="$GONANO_BASE_DIR/base_checkpoints/$TAG/model_$(printf '%06d' "$STEPS").gn"
  PROMPT=$(( SEQ * PROMPT_RATIO ))
  echo "==> benchmarking seq=$SEQ prompt=$PROMPT : $MODEL"
  "$WORKDIR/infer_bench" \
    --model "$MODEL" \
    --prompt-tokens "$PROMPT" \
    --batch-sizes "$BATCH_SIZES" \
    --decode-tokens "$DECODE_TOKENS"
  echo
done
