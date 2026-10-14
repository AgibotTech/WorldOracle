#!/usr/bin/env bash
# Create the `acwmbench` conda env and install dependencies.
#
# Usage:
#   bash setup_env.sh
#
# Independent from the existing EWMBench env. Re-runnable.

set -euo pipefail

ENV_NAME="${ACWM_ENV:-acwmbench}"
PYTHON_VERSION="${ACWM_PYTHON:-3.10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REPO_ROOT}/requirements.txt"

echo "[acwmbench] Repo root: ${REPO_ROOT}"
echo "[acwmbench] Target conda env: ${ENV_NAME} (python ${PYTHON_VERSION})"

if ! command -v conda >/dev/null 2>&1; then
    echo "[acwmbench] ERROR: conda not found. Install miniconda first." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "[acwmbench] Env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[acwmbench] Creating env '${ENV_NAME}' ..."
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

echo "[acwmbench] Upgrading pip ..."
python -m pip install --upgrade pip

echo "[acwmbench] Installing requirements ..."
python -m pip install -r "${REQ_FILE}"

echo "[acwmbench] Done. Activate with: conda activate ${ENV_NAME}"
