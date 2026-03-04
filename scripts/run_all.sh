#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash scripts/run_all.sh
#   TEST_SIZE=8 bash scripts/run_all.sh
#
# Notes:
# - Uses only GPU 3 by default.
# - Activates the provided venv if it exists.

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
export CUDA_VISIBLE_DEVICES=${GPU_ID}

SAVE_DIR=${SAVE_DIR:-./results}
TEST_SIZE=${TEST_SIZE:-}

LLADA_MODEL_PATH=${LLADA_MODEL_PATH:-./models/LLaDA-8B-Instruct}
DREAM_MODEL_PATH=${DREAM_MODEL_PATH:-./models/Dream-v0-Instruct-7B}

run_py() {
  local cmd=("${PYTHON_BIN}" "$@")
  if [[ -n "${TEST_SIZE}" ]]; then
    cmd+=(--test_size "${TEST_SIZE}")
  fi
  echo "[RUN] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${cmd[*]}"
  "${cmd[@]}"
}

# -----------------
# LLaDA (4 datasets)
# -----------------
LLADA_GEN_LENGTH=${LLADA_GEN_LENGTH:-256}
LLADA_BLOCK_LENGTH=${LLADA_BLOCK_LENGTH:-64}
LLADA_STEPS=${LLADA_STEPS:-256}
LLADA_ALG=${LLADA_ALG:-klass}
LLADA_UNMASK_STRATEGY=${LLADA_UNMASK_STRATEGY:-all}
LLADA_CONF_THRESHOLD=${LLADA_CONF_THRESHOLD:-0.6}
LLADA_KL_THRESHOLD=${LLADA_KL_THRESHOLD:-0.01}
LLADA_HISTORY_LENGTH=${LLADA_HISTORY_LENGTH:-2}

for ds in gsm8k math humaneval mbpp; do
  run_py ./src/llada_evaluation.py \
    --model_path "${LLADA_MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --gen_length "${LLADA_GEN_LENGTH}" \
    --block_length "${LLADA_BLOCK_LENGTH}" \
    --steps "${LLADA_STEPS}" \
    --conf_threshold "${LLADA_CONF_THRESHOLD}" \
    --kl_threshold "${LLADA_KL_THRESHOLD}" \
    --history_length "${LLADA_HISTORY_LENGTH}" \
    --dataset "${ds}" \
    --alg "${LLADA_ALG}" \
    --unmask_strategy "${LLADA_UNMASK_STRATEGY}" \
    --save_steps
done

# -----------------
# Dream (4 datasets)
# -----------------
DREAM_GEN_LENGTH=${DREAM_GEN_LENGTH:-256}
DREAM_STEPS=${DREAM_STEPS:-256}
DREAM_ALG=${DREAM_ALG:-klass}
DREAM_UNMASK_STRATEGY=${DREAM_UNMASK_STRATEGY:-all}
DREAM_CONF_THRESHOLD=${DREAM_CONF_THRESHOLD:-0.9}
DREAM_KL_THRESHOLD=${DREAM_KL_THRESHOLD:-0.001}
DREAM_HISTORY_LENGTH=${DREAM_HISTORY_LENGTH:-2}
DREAM_TEMPERATURE=${DREAM_TEMPERATURE:-0.2}
DREAM_TOP_P=${DREAM_TOP_P:-0.95}

for ds in gsm8k math humaneval mbpp; do
  run_py ./src/dream_evaluation.py \
    --model_path "${DREAM_MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --gen_length "${DREAM_GEN_LENGTH}" \
    --steps "${DREAM_STEPS}" \
    --conf_threshold "${DREAM_CONF_THRESHOLD}" \
    --kl_threshold "${DREAM_KL_THRESHOLD}" \
    --history_length "${DREAM_HISTORY_LENGTH}" \
    --dataset "${ds}" \
    --unmask_strategy "${DREAM_UNMASK_STRATEGY}" \
    --alg "${DREAM_ALG}" \
    --temperature "${DREAM_TEMPERATURE}" \
    --top_p "${DREAM_TOP_P}" \
    --save_steps
done

echo "Done. Forward pass stats are saved under ${SAVE_DIR}/**/*/forward_stats.json"
