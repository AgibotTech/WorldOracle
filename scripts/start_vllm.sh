#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for one of our supported models.
#
# Detached by default so it survives Cursor / SSH / laptop disconnects.
#
# Usage:
#   bash scripts/start_vllm.sh                      # default: qwen3_vl_8b
#   bash scripts/start_vllm.sh qwen3_vl_8b
#   bash scripts/start_vllm.sh qwen3_5_9b
#   bash scripts/start_vllm.sh /abs/path/to/ckpt    # any local ckpt
#   bash scripts/start_vllm.sh qwen3_vl_8b --foreground
#   bash scripts/start_vllm.sh --status
#   bash scripts/start_vllm.sh --stop
#
# Env vars (all optional):
#   ACWM_VLLM_PORT          default 8000
#   ACWM_VLLM_GPU_FRAC      default 0.85
#   ACWM_VLLM_MAX_LEN       default 32768
#   ACWM_VLLM_DTYPE         default bfloat16
#   ACWM_VLLM_ENV           name of conda env that has vllm installed
#                            (default: vllm). The script will `conda activate` it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_ROOT="${REPO_ROOT}/ckpt"
PID_FILE="${REPO_ROOT}/.vllm.pid"
LOG_FILE="${REPO_ROOT}/.vllm.log"

PORT="${ACWM_VLLM_PORT:-8000}"
GPU_FRAC="${ACWM_VLLM_GPU_FRAC:-0.85}"
MAX_LEN="${ACWM_VLLM_MAX_LEN:-32768}"
DTYPE="${ACWM_VLLM_DTYPE:-bfloat16}"
VLLM_ENV="${ACWM_VLLM_ENV:-vllm}"

# ---------------------------------------------------------------------------
# Alias resolution (matches scripts/download_models.sh + acwmbench/models/model_zoo.py)
# ---------------------------------------------------------------------------

_resolve_alias() {
    case "$1" in
        qwen3_vl_8b|qwen3-vl-8b|qwen3vl8b)
            echo "Qwen3-VL-8B-Instruct" ;;
        qwen3_5_9b|qwen3.5-9b|qwen3.5_9b|qwen35-9b)
            echo "Qwen3.5-9B" ;;
        */*) basename "$1" ;;
        *) echo "$1" ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

MODEL_ARG=""
ACTION=""
for tok in "$@"; do
    case "${tok}" in
        --status|status)  ACTION="status" ;;
        --stop|stop)      ACTION="stop" ;;
        --foreground|-f)  ACTION="foreground" ;;
        --*)              echo "[acwmbench-vllm] unknown flag: ${tok}" >&2; exit 2 ;;
        *)                MODEL_ARG="${tok}" ;;
    esac
done

# ---------------------------------------------------------------------------
# status / stop
# ---------------------------------------------------------------------------

if [[ "${ACTION}" == "status" ]]; then
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "[acwmbench-vllm] running, pid=$(cat "${PID_FILE}")"
        echo "[acwmbench-vllm] endpoint: http://localhost:${PORT}/v1"
        if command -v curl >/dev/null 2>&1; then
            echo "[acwmbench-vllm] /v1/models:"
            curl -sS --max-time 3 "http://localhost:${PORT}/v1/models" \
                | python -m json.tool 2>/dev/null || echo "  (server not yet accepting connections)"
        fi
        echo "[acwmbench-vllm] tail of log:"
        tail -n 8 "${LOG_FILE}" 2>/dev/null
    else
        echo "[acwmbench-vllm] not running"
    fi
    exit 0
fi

if [[ "${ACTION}" == "stop" ]]; then
    if [[ -f "${PID_FILE}" ]]; then
        pid="$(cat "${PID_FILE}")"
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[acwmbench-vllm] killing pid=${pid} and children ..."
            pkill -TERM -P "${pid}" 2>/dev/null || true
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 3
            pkill -KILL -P "${pid}" 2>/dev/null || true
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        rm -f "${PID_FILE}"
    fi
    echo "[acwmbench-vllm] stopped."
    exit 0
fi

# ---------------------------------------------------------------------------
# Resolve model directory + served-model-name
# ---------------------------------------------------------------------------

MODEL_ARG="${MODEL_ARG:-qwen3_vl_8b}"
LOCAL_NAME="$(_resolve_alias "${MODEL_ARG}")"
if [[ "${MODEL_ARG}" == /* ]]; then
    MODEL_DIR="${MODEL_ARG}"
    SERVED_NAME="${LOCAL_NAME}"
else
    MODEL_DIR="${CKPT_ROOT}/${LOCAL_NAME}"
    SERVED_NAME="${MODEL_ARG}"
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[acwmbench-vllm] ERROR: model dir not found: ${MODEL_DIR}" >&2
    echo "                 Run scripts/download_models.sh ${MODEL_ARG} first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_run_vllm() {
    echo "[acwmbench-vllm] === $(date) === starting vllm server"
    echo "[acwmbench-vllm] env:            ${VLLM_ENV}"
    echo "[acwmbench-vllm] model:          ${MODEL_DIR}"
    echo "[acwmbench-vllm] served name:    ${SERVED_NAME}"
    echo "[acwmbench-vllm] port:           ${PORT}"
    echo "[acwmbench-vllm] gpu frac:       ${GPU_FRAC}"
    echo "[acwmbench-vllm] max model len:  ${MAX_LEN}"
    echo "[acwmbench-vllm] dtype:          ${DTYPE}"

    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${VLLM_ENV}"

    exec vllm serve "${MODEL_DIR}" \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --dtype "${DTYPE}" \
        --max-model-len "${MAX_LEN}" \
        --gpu-memory-utilization "${GPU_FRAC}" \
        --served-model-name "${SERVED_NAME}" \
        --trust-remote-code \
        --disable-log-requests
}

if [[ "${ACTION}" == "foreground" ]]; then
    _run_vllm
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[acwmbench-vllm] already running, pid=$(cat "${PID_FILE}")"
    echo "[acwmbench-vllm] bash $0 --status    to inspect"
    echo "[acwmbench-vllm] bash $0 --stop      to restart"
    exit 0
fi

: > "${LOG_FILE}"

setsid nohup env \
    ACWM_VLLM_PORT="${PORT}" \
    ACWM_VLLM_GPU_FRAC="${GPU_FRAC}" \
    ACWM_VLLM_MAX_LEN="${MAX_LEN}" \
    ACWM_VLLM_DTYPE="${DTYPE}" \
    ACWM_VLLM_ENV="${VLLM_ENV}" \
    bash "$0" "${MODEL_ARG}" --foreground </dev/null >>"${LOG_FILE}" 2>&1 &
SRV_PID=$!
echo "${SRV_PID}" > "${PID_FILE}"
disown "${SRV_PID}" 2>/dev/null || true

echo "[acwmbench-vllm] started in background"
echo "[acwmbench-vllm]   model:        ${MODEL_DIR}"
echo "[acwmbench-vllm]   served name:  ${SERVED_NAME}"
echo "[acwmbench-vllm]   endpoint:     http://localhost:${PORT}/v1"
echo "[acwmbench-vllm]   pid:          ${SRV_PID}"
echo "[acwmbench-vllm]   log:          ${LOG_FILE}"
echo "[acwmbench-vllm]"
echo "[acwmbench-vllm] commands:"
echo "[acwmbench-vllm]   tail -f ${LOG_FILE}            # follow boot"
echo "[acwmbench-vllm]   bash $0 --status                                  # show /v1/models"
echo "[acwmbench-vllm]   bash $0 --stop                                    # kill server"
