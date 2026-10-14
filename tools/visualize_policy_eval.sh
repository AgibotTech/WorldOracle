#!/usr/bin/env bash
# Visualize policy_eval_dataset only (all tasks / episodes, no GT matching).
set -euo pipefail

source ~/.bashrc
cd "$(dirname "$0")/.."
conda activate acwmbench

POLICY_ROOT="${POLICY_ROOT:-/path/to/policy-inputs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
VIEW="${VIEW:-top_head}"

python tools/visualize_policy_eval_dataset.py \
  "$POLICY_ROOT" \
  --dataset-name policy_eval \
  --view "$VIEW" \
  --prewarm \
  --host "$HOST" --port "$PORT"
