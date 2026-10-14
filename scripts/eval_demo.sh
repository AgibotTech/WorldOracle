#!/usr/bin/env bash
# Smoke test: run physical_judge on a small subset of data.
#
# Assumes:
#   - conda env `acwmbench` is created and activated
#   - a VLM backend is configured in configs/base.yaml (cloud API key in
#     OPENAI_API_KEY in the environment, or a local checkpoint under ./ckpt)
#   - data is mounted at the path configured in configs/base.yaml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/base.yaml}"

python evaluate.py \
    --dimension physical_judge \
    --config_path "${CONFIG}" \
    --overwrite
