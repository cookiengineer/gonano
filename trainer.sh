#!/usr/bin/env bash
#
# trainer.sh — a friendly wrapper around `cmd/trainer`.
#
# Usage:
#   trainer.sh <data-dir> [--format parquet|markdown] [options...]
#
#   <data-dir>               directory of .parquet shards or .md files (required)
#   --format parquet|markdown  how to read the data (default: parquet)
#   --train-tokenizer         train a fresh BPE tokenizer on the data
#   --tokenizer <path>        use a specific tokenizer.json
#   --depth N / --max-seq-len N / --num-iterations N / ...  passed to the trainer
#
# If no tokenizer exists yet, it copies the default tokenizer for the chosen
# format (tokenizer/defaults/markdown.json for Markdown, or byte.json for
# Parquet) into the base directory, so training always has a working tokenizer.
#
# Examples:
#   trainer.sh ~/webdata-md --format markdown --depth 4
#   trainer.sh ~/.cache/gonano/base_data --format parquet --train-tokenizer --depth 12

set -euo pipefail

export GOEXPERIMENT=simd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"   # so `go run ./cmd/...` works from any working directory

GONANO_BASE_DIR="${GONANO_BASE_DIR:-$HOME/.cache/gonano}"
export GONANO_BASE_DIR

if [[ $# -lt 1 ]]; then
  echo "usage: trainer.sh <data-dir> [--format parquet|markdown] [--train-tokenizer] [options...]" >&2
  exit 1
fi

DATA_DIR="$1"
shift

FORMAT="parquet"
TRAIN_TOK=0
TOKENIZER=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --format)
      FORMAT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --train-tokenizer)
      TRAIN_TOK=1; ARGS+=("$1"); shift ;;
    --tokenizer)
      TOKENIZER="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done

# Ensure a tokenizer exists: copy the default tokenizer for the chosen format
# (markdown.json for Markdown, byte.json otherwise) if there is no trained one
# and the user did not ask to train one or supply one.
TOK_FILE="$GONANO_BASE_DIR/tokenizer/tokenizer.json"
if [[ "$TRAIN_TOK" -eq 0 && -z "$TOKENIZER" && ! -f "$TOK_FILE" ]]; then
  mkdir -p "$(dirname "$TOK_FILE")"
  case "$FORMAT" in
    markdown) DEFAULT_TOK="$SCRIPT_DIR/tokenizer/defaults/markdown.json" ;;
    *)        DEFAULT_TOK="$SCRIPT_DIR/tokenizer/defaults/byte.json" ;;
  esac
  cp "$DEFAULT_TOK" "$TOK_FILE"
  echo "==> no tokenizer.json found; using default $FORMAT tokenizer (copied to $TOK_FILE)"
fi

echo "==> training (data-dir=$DATA_DIR, format=$FORMAT)"
go run ./cmd/trainer --data-dir "$DATA_DIR" --format "$FORMAT" "${ARGS[@]}"
