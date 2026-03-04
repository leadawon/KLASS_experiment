#!/usr/bin/env bash
set -euo pipefail

# Sweeps diffusion steps (forward-pass budget) on Dream for:
#   - GSM8K (lm-eval-harness task: gsm8k_cot)
#   - IFEval (lm-eval-harness task: ifeval)
#
# Usage examples (from /workspace/KLASS):
#   bash scripts/run_all_sweep.sh
#   GPU_ID=2 GSM8K_LIMIT=200 IFEVAL_LIMIT=50 bash scripts/run_all_sweep.sh
#   SAVE_ROOT=./results_sweep RUN_TAG=mytag bash scripts/run_all_sweep.sh
#   STEPS_LIST="64 128 256 512" bash scripts/run_all_sweep.sh
#
# Notes:
# - KLASS knobs are kept fixed to match scripts/run_all.sh (override via env vars if needed).
# - Defaults to GPU 2.
# - Prefers /workspace/venvs/klassvenv if present.
# - Produces per-sweep outputs under: ${SAVE_ROOT}/sweep/${RUN_TAG}/...

PYTHON_BIN=${PYTHON_BIN:-}

# Prefer the shared workspace venv (this repo is under /workspace/KLASS).
if [[ -z "${PYTHON_BIN}" && -x /workspace/venvs/klassvenv/bin/python ]]; then
  PYTHON_BIN=/workspace/venvs/klassvenv/bin/python
fi

# Backward-compatible path (if you keep a venv inside this repo).
if [[ -z "${PYTHON_BIN}" && -x venvs/klassvenv/bin/python ]]; then
  PYTHON_BIN=venvs/klassvenv/bin/python
fi

# Fall back to whatever python is on PATH.
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

GPU_ID=${GPU_ID:-3}
export GPU_ID
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Optional preflight: avoid confusing CUDA OOMs when the chosen GPU is already occupied
SKIP_GPU_CHECK=${SKIP_GPU_CHECK:-0}
MIN_FREE_MEM_MIB=${MIN_FREE_MEM_MIB:-20000}
if [[ "${SKIP_GPU_CHECK}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  free_mem_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" 2>/dev/null | head -n 1 | tr -d ' ' || true)
  if [[ -n "${free_mem_mib}" ]] && [[ "${free_mem_mib}" =~ ^[0-9]+$ ]]; then
    if (( free_mem_mib < MIN_FREE_MEM_MIB )); then
      echo "[ERROR] GPU ${GPU_ID} has only ${free_mem_mib} MiB free (need >= ${MIN_FREE_MEM_MIB} MiB by default)." >&2
      echo "        Another process is likely using the GPU. Pick a different GPU (GPU_ID=0/1/3) or wait." >&2
      echo "        Override with SKIP_GPU_CHECK=1 or lower MIN_FREE_MEM_MIB if you know what you're doing." >&2
      exit 2
    fi
  fi
fi

SAVE_ROOT=${SAVE_ROOT:-./results_harness}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
SAVE_DIR="${SAVE_ROOT}/sweep/${RUN_TAG}"
export SAVE_ROOT RUN_TAG SAVE_DIR

# Bench-specific limits (lm-eval --limit)
GSM8K_LIMIT=${GSM8K_LIMIT:-200}
IFEVAL_LIMIT=${IFEVAL_LIMIT:-50}
BATCH_SIZE=${BATCH_SIZE:-1}               # lm-eval wrapper
export GSM8K_LIMIT IFEVAL_LIMIT BATCH_SIZE

# IFEval: generation length + context length caps
# Fixed IFEval max_new_tokens as requested.
IFEVAL_MAX_NEW_TOKENS=${IFEVAL_MAX_NEW_TOKENS:-768}
IFEVAL_MAX_LENGTH_LIST=${IFEVAL_MAX_LENGTH_LIST:-"4096"}
export IFEVAL_MAX_NEW_TOKENS IFEVAL_MAX_LENGTH_LIST

DREAM_MODEL_PATH=${DREAM_MODEL_PATH:-./models/Dream-v0-Instruct-7B}
export DREAM_MODEL_PATH

# Model dtype override (useful for memory/debug: float16, bfloat16, float32, auto)
DREAM_DTYPE=${DREAM_DTYPE:-auto}
export DREAM_DTYPE

# Optional memory saver (may help IFEval max_new_tokens=1280 fit on 24GB GPUs)
DREAM_LOAD_IN_8BIT=${DREAM_LOAD_IN_8BIT:-0}
export DREAM_LOAD_IN_8BIT

# Generation defaults (lm-eval wrapper)
GSM8K_MAX_NEW_TOKENS=${GSM8K_MAX_NEW_TOKENS:-256}
export GSM8K_MAX_NEW_TOKENS

# Context length caps (lm-eval wrapper)
# Note: For GSM8K prompts, a larger max_length is typically fine; generation length is controlled by *_MAX_NEW_TOKENS.
GSM8K_MAX_LENGTH=${GSM8K_MAX_LENGTH:-4096}
export GSM8K_MAX_LENGTH

# Keep non-KLASS sampling knobs fixed (not part of KLASS idea)
# Defaults match scripts/run_all.sh
DREAM_TEMPERATURE=${DREAM_TEMPERATURE:-0.2}
DREAM_TOP_P=${DREAM_TOP_P:-0.95}
DREAM_ALG=${DREAM_ALG:-klass}
export DREAM_TEMPERATURE DREAM_TOP_P DREAM_ALG

# Restore the run_all.sh KLASS hyperparameters (fixed, not swept)
KLASS_CONF_THRESHOLD=${KLASS_CONF_THRESHOLD:-0.9}
KLASS_KL_THRESHOLD=${KLASS_KL_THRESHOLD:-0.001}
KLASS_HISTORY_LENGTH=${KLASS_HISTORY_LENGTH:-2}
KLASS_UNMASK_STRATEGY=${KLASS_UNMASK_STRATEGY:-all}
export KLASS_CONF_THRESHOLD KLASS_KL_THRESHOLD KLASS_HISTORY_LENGTH KLASS_UNMASK_STRATEGY

# Sweep target: diffusion steps (GSM8K max_new_tokens=256, so keep steps in 64~256 range)
STEPS_LIST=${STEPS_LIST:-"64 128 192 256"}
export STEPS_LIST

# lm-eval wrapper uses its own default steps=128; we always pass steps explicitly per run.

run_py() {
  local cmd=("${PYTHON_BIN}" "$@")
  echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${cmd[*]}"
  "${cmd[@]}"
}

ensure_lm_eval_deps() {
  # Install lm_eval if missing (prefers the vendored harness in /workspace/Dream)
  if ! "${PYTHON_BIN}" -c "import lm_eval" >/dev/null 2>&1; then
    echo "lm_eval not found in venv; installing from /workspace/Dream/lm-evaluation-harness ..."
    "${PYTHON_BIN}" -m pip install -e /workspace/Dream/lm-evaluation-harness
  fi

  # IFEval requires a couple extra deps (lm-eval-harness calls these out as optional extras).
  if ! "${PYTHON_BIN}" -c "import langdetect, immutabledict" >/dev/null 2>&1; then
    echo "Installing IFEval deps (langdetect, immutabledict) ..."
    "${PYTHON_BIN}" -m pip install -q langdetect immutabledict
  fi

  if ! "${PYTHON_BIN}" -c "import tqdm" >/dev/null 2>&1; then
    echo "Installing tqdm (for sweep progress bar) ..."
    "${PYTHON_BIN}" -m pip install -q tqdm
  fi

  if [[ "${DREAM_LOAD_IN_8BIT}" == "1" ]] && ! "${PYTHON_BIN}" -c "import bitsandbytes" >/dev/null 2>&1; then
    echo "Installing bitsandbytes (for 8-bit model loading) ..."
    "${PYTHON_BIN}" -m pip install -q bitsandbytes
  fi
}

run_lmeval_once() {
  local out_dir="$1"
  local task="$2"
  local limit="$3"
  local max_new_tokens="$4"
  local max_length="$5"
  local steps="$6"
  local conf="$7"
  local kl="$8"
  local hist="$9"
  local unmask="${10}"

  mkdir -p "${out_dir}"

  local pretrained
  pretrained=$("${PYTHON_BIN}" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${DREAM_MODEL_PATH}")

  local model_args
  model_args="pretrained=${pretrained},device=cuda:0,batch_size=${BATCH_SIZE},stats_dir=${out_dir},max_new_tokens=${max_new_tokens},max_length=${max_length},steps=${steps},alg=${DREAM_ALG},conf_threshold=${conf},kl_threshold=${kl},kl_history_length=${hist},unmask_strategy=${unmask},temperature=${DREAM_TEMPERATURE},top_p=${DREAM_TOP_P}"

  local cmd=("${PYTHON_BIN}" eval/klass_lm_eval.py
    --model klass_dream
    --model_args "${model_args}"
    --tasks "${task}"
    --output_path "${out_dir}"
    --batch_size "${BATCH_SIZE}"
  )

  cmd+=(--limit "${limit}")

  echo "[RUN] (lm-eval) task=${task} limit=${limit} steps=${steps} max_new_tokens=${max_new_tokens} max_length=${max_length} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${cmd[*]}"
  "${cmd[@]}"
}

run_ifeval_sweep() {
  local sweep_id="$1"
  local steps="$2"
  local max_length="$3"

  ensure_lm_eval_deps

  local out_dir="${SAVE_DIR}/lm_eval/dream/ifeval/${sweep_id}/maxlen${max_length}"
  local log_path="${out_dir}/run.log"

  mkdir -p "${out_dir}"

  set +e
  run_lmeval_once "${out_dir}" "ifeval" "${IFEVAL_LIMIT}" "${IFEVAL_MAX_NEW_TOKENS}" "${max_length}" "${steps}" "${KLASS_CONF_THRESHOLD}" "${KLASS_KL_THRESHOLD}" "${KLASS_HISTORY_LENGTH}" "${KLASS_UNMASK_STRATEGY}" >"${log_path}" 2>&1
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[ERROR] IFEval sweep failed (max_length=${max_length}). Log: ${log_path}" >&2
    return ${rc}
  fi
}

run_gsm8k_sweep() {
  local sweep_id="$1"
  local steps="$2"

  ensure_lm_eval_deps

  local out_dir="${SAVE_DIR}/lm_eval/dream/gsm8k_cot/${sweep_id}"
  local log_path="${out_dir}/run.log"
  mkdir -p "${out_dir}"

  set +e
  run_lmeval_once "${out_dir}" "gsm8k_cot" "${GSM8K_LIMIT}" "${GSM8K_MAX_NEW_TOKENS}" "${GSM8K_MAX_LENGTH}" "${steps}" "${KLASS_CONF_THRESHOLD}" "${KLASS_KL_THRESHOLD}" "${KLASS_HISTORY_LENGTH}" "${KLASS_UNMASK_STRATEGY}" >"${log_path}" 2>&1
  local rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[ERROR] GSM8K(lm-eval) sweep failed. Log: ${log_path}" >&2
    return ${rc}
  fi
}

echo "[INFO] PYTHON_BIN=${PYTHON_BIN}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (GPU_ID=${GPU_ID})"
echo "[INFO] SAVE_DIR=${SAVE_DIR}"
echo "[INFO] DREAM_MODEL_PATH=${DREAM_MODEL_PATH}"
echo "[INFO] GSM8K(lm-eval): task=gsm8k_cot max_new_tokens=${GSM8K_MAX_NEW_TOKENS} max_length=${GSM8K_MAX_LENGTH} limit=${GSM8K_LIMIT} batch_size=${BATCH_SIZE}"
echo "[INFO] IFEval(lm-eval): task=ifeval max_new_tokens=${IFEVAL_MAX_NEW_TOKENS} max_length_list=(${IFEVAL_MAX_LENGTH_LIST}) limit=${IFEVAL_LIMIT} batch_size=${BATCH_SIZE}"
echo "[INFO] Fixed non-KLASS knobs: temperature=${DREAM_TEMPERATURE} top_p=${DREAM_TOP_P}"
echo "[INFO] Fixed KLASS knobs (from run_all.sh): conf=${KLASS_CONF_THRESHOLD} kl=${KLASS_KL_THRESHOLD} hist=${KLASS_HISTORY_LENGTH} unmask=${KLASS_UNMASK_STRATEGY}"
echo "[INFO] Sweeping steps only: ${STEPS_LIST}"

ensure_lm_eval_deps

# Delegate to a Python runner so we can render a tqdm progress bar while keeping
# per-run logs written to run.log.
exec "${PYTHON_BIN}" scripts/run_all_sweep_steps.py
