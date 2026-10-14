#!/usr/bin/env bash
# Download a VLM checkpoint for ACWMBench.
#
# Detached by default (nohup + setsid). Per-model pid/log so the two supported
# checkpoints can be downloaded in parallel without clobbering each other.
#
# Usage:
#   bash scripts/download_models.sh                       # default: qwen3_vl_8b
#   bash scripts/download_models.sh qwen3_vl_8b           # Qwen/Qwen3-VL-8B-Instruct
#   bash scripts/download_models.sh qwen3_5_9b            # Qwen/Qwen3.5-9B
#   bash scripts/download_models.sh Qwen/Qwen3-VL-4B-Instruct   # full HF repo id
#   bash scripts/download_models.sh qwen3_vl_8b --foreground    # run in current shell
#   bash scripts/download_models.sh qwen3_vl_8b --status        # show progress
#   bash scripts/download_models.sh qwen3_vl_8b --stop          # kill the job
#   bash scripts/download_models.sh --status-all          # status of every model
#
# Env vars (all optional):
#   ACWM_VLM_REPO   override the resolved repo id
#   ACWM_SOURCE     hf | ms       (default: hf, via hf-mirror.com, Xet disabled)
#   HF_ENDPOINT     default: https://hf-mirror.com

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_ROOT="${REPO_ROOT}/ckpt"
mkdir -p "${CKPT_ROOT}"

# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

_resolve_repo() {
    # $1 = alias or full repo id ("owner/name"). Echoes the canonical repo id.
    local arg="$1"
    case "${arg}" in
        qwen3_vl_8b|qwen3-vl-8b|qwen3vl8b)
            echo "Qwen/Qwen3-VL-8B-Instruct" ;;
        qwen3_5_9b|qwen3.5-9b|qwen3.5_9b|qwen35-9b)
            echo "Qwen/Qwen3.5-9B" ;;
        */*)
            echo "${arg}" ;;
        *)
            echo "ERROR: unknown model alias '${arg}'. Try: qwen3_vl_8b | qwen3_5_9b | <owner>/<name>" >&2
            return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Parse: first non-flag arg is the model; remaining flags handled below.
# ---------------------------------------------------------------------------

MODEL_ARG=""
ACTION=""
for tok in "$@"; do
    case "${tok}" in
        --status|status)         ACTION="status" ;;
        --stop|stop)             ACTION="stop" ;;
        --status-all|status-all) ACTION="status-all" ;;
        --foreground|-f)         ACTION="foreground" ;;
        --*)                     echo "[acwmbench] unknown flag: ${tok}" >&2; exit 2 ;;
        *)                       MODEL_ARG="${tok}" ;;
    esac
done

if [[ "${ACTION}" == "status-all" ]]; then
    echo "[acwmbench] all download jobs under ${CKPT_ROOT}:"
    shopt -s nullglob
    found=0
    for pidf in "${CKPT_ROOT}"/.download.*.pid; do
        found=1
        name="$(basename "${pidf}" .pid)"; name="${name#.download.}"
        if kill -0 "$(cat "${pidf}")" 2>/dev/null; then
            sz="$(du -sh "${CKPT_ROOT}/${name}" 2>/dev/null | awk '{print $1}')"
            echo "  RUNNING  ${name}  pid=$(cat "${pidf}")  size=${sz:-?}"
        else
            sz="$(du -sh "${CKPT_ROOT}/${name}" 2>/dev/null | awk '{print $1}')"
            echo "  stopped  ${name}  size=${sz:-?}  log=${CKPT_ROOT}/.download.${name}.log"
        fi
    done
    [[ "${found}" -eq 0 ]] && echo "  (none)"
    exit 0
fi

# Resolve repo: explicit arg > env var > default (qwen3_vl_8b).
if [[ -n "${MODEL_ARG}" ]]; then
    REPO="$(_resolve_repo "${MODEL_ARG}")"
elif [[ -n "${ACWM_VLM_REPO:-}" ]]; then
    REPO="${ACWM_VLM_REPO}"
else
    REPO="$(_resolve_repo qwen3_vl_8b)"
fi
SOURCE="${ACWM_SOURCE:-hf}"
LOCAL_NAME="$(basename "${REPO}")"
LOCAL_DIR="${CKPT_ROOT}/${LOCAL_NAME}"

# Per-model pid/log so parallel downloads don't collide.
PID_FILE="${CKPT_ROOT}/.download.${LOCAL_NAME}.pid"
LOG_FILE="${CKPT_ROOT}/.download.${LOCAL_NAME}.log"

# ---------------------------------------------------------------------------
# status / stop (per-model)
# ---------------------------------------------------------------------------

_status() {
    echo "[acwmbench] model:    ${REPO}"
    echo "[acwmbench] target:   ${LOCAL_DIR}"
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "[acwmbench] running, pid=$(cat "${PID_FILE}")"
    else
        echo "[acwmbench] not running"
    fi
    if [[ -d "${LOCAL_DIR}" ]]; then
        echo -n "[acwmbench] downloaded so far: "
        du -sh "${LOCAL_DIR}" 2>/dev/null | awk '{print $1}'
    fi
    if [[ -f "${LOG_FILE}" ]]; then
        echo "[acwmbench] last log lines (full at ${LOG_FILE}):"
        tail -n 5 "${LOG_FILE}" 2>/dev/null
    fi
}

_stop() {
    if [[ -f "${PID_FILE}" ]]; then
        local pid; pid="$(cat "${PID_FILE}")"
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[acwmbench] killing pid=${pid} (model=${LOCAL_NAME}) and children ..."
            pkill -TERM -P "${pid}" 2>/dev/null || true
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 2
            pkill -KILL -P "${pid}" 2>/dev/null || true
            kill -KILL "${pid}" 2>/dev/null || true
        fi
        rm -f "${PID_FILE}"
    fi
    echo "[acwmbench] stopped: ${LOCAL_NAME}"
}

case "${ACTION}" in
    status) _status; exit 0 ;;
    stop)   _stop;   exit 0 ;;
esac

# ---------------------------------------------------------------------------
# The actual download (foreground worker)
# ---------------------------------------------------------------------------

_run_download() {
    echo "[acwmbench] === $(date) === starting download"
    echo "[acwmbench] repo:     ${REPO}"
    echo "[acwmbench] source:   ${SOURCE}"
    echo "[acwmbench] target:   ${LOCAL_DIR}"
    mkdir -p "${LOCAL_DIR}"

    python - <<'PY'
import importlib, subprocess, sys
need = []
for pkg, mod in [("modelscope", "modelscope"), ("huggingface_hub", "huggingface_hub")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    print(f"[acwmbench] installing: {need}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", *need])
PY

    if [[ "${SOURCE}" == "hf" ]]; then
        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
        # hf-mirror only proxies the legacy LFS path; force-disable Xet.
        export HF_HUB_DISABLE_XET=1
        echo "[acwmbench] HF_ENDPOINT = ${HF_ENDPOINT}  (Xet disabled)"
        python - <<PY
import os
from huggingface_hub import snapshot_download
print(f"[acwmbench] endpoint = {os.environ.get('HF_ENDPOINT')}")
snapshot_download(
    repo_id="${REPO}",
    local_dir="${LOCAL_DIR}",
    max_workers=8,
)
print("[acwmbench] OK: ${REPO} -> ${LOCAL_DIR}")
PY
    elif [[ "${SOURCE}" == "ms" ]]; then
        python - <<PY
from modelscope import snapshot_download
out = snapshot_download(
    model_id="${REPO}",
    local_dir="${LOCAL_DIR}",
    max_workers=8,
)
print(f"[acwmbench] OK: {out}")
PY
    else
        echo "[acwmbench] ERROR: unknown ACWM_SOURCE='${SOURCE}'. Use 'hf' or 'ms'." >&2
        exit 1
    fi

    echo "[acwmbench] === $(date) === all downloads finished"
    echo "[acwmbench] configs/base.yaml -> physical_judge.vlm.model_path should be:"
    echo "            ${LOCAL_DIR}"
}

# ---------------------------------------------------------------------------
# Foreground vs. detached
# ---------------------------------------------------------------------------

if [[ "${ACTION}" == "foreground" ]]; then
    _run_download
    exit 0
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "[acwmbench] download already running for ${LOCAL_NAME}, pid=$(cat "${PID_FILE}")"
    echo "[acwmbench] tail -f ${LOG_FILE}                              # follow"
    echo "[acwmbench] bash $0 ${MODEL_ARG:-${REPO}} --status           # status"
    echo "[acwmbench] bash $0 ${MODEL_ARG:-${REPO}} --stop             # abort"
    exit 0
fi

: > "${LOG_FILE}"

# Re-exec ourselves in detached mode with --foreground, preserving env+args.
setsid nohup env \
    ACWM_VLM_REPO="${REPO}" \
    ACWM_SOURCE="${SOURCE}" \
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    bash "$0" --foreground </dev/null >>"${LOG_FILE}" 2>&1 &
DL_PID=$!
echo "${DL_PID}" > "${PID_FILE}"
disown "${DL_PID}" 2>/dev/null || true

echo "[acwmbench] download launched in background"
echo "[acwmbench]   model:  ${REPO}"
echo "[acwmbench]   pid:    ${DL_PID}"
echo "[acwmbench]   target: ${LOCAL_DIR}"
echo "[acwmbench]   log:    ${LOG_FILE}"
echo "[acwmbench]"
echo "[acwmbench] commands:"
echo "[acwmbench]   tail -f ${LOG_FILE}                              # follow log"
echo "[acwmbench]   bash $0 ${MODEL_ARG:-${REPO}} --status           # show progress"
echo "[acwmbench]   bash $0 ${MODEL_ARG:-${REPO}} --stop             # abort"
echo "[acwmbench]   bash $0 --status-all                             # list all jobs"
