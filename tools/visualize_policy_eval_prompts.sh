#!/usr/bin/env bash
# Visualize policy_eval_dataset keyframes + success_criteria (pause / continue).
set -euo pipefail

source ~/.bashrc
cd "$(dirname "$0")/.."
conda activate acwmbench

ROOT="${ROOT:-/path/to/policy-inputs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
VIEW="${VIEW:-top_head}"

python tools/visualize_policy_eval_prompts.py \
  --root "$ROOT" \
  --view "$VIEW" \
  --prewarm \
  --host "$HOST" --port "$PORT"
