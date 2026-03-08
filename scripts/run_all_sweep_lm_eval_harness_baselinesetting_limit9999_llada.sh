#!/usr/bin/env bash
set -euo pipefail

# Baseline lm-eval-harness sweep with KLASS (LLaDA-Instruct) using requested settings.
# Tasks: gsm8k_cot, minerva_math500, humaneval_instruct, mbpp_instruct, ifeval
# Limits: all=9999
# num_fewshot: 0 for all
# max_new_tokens: gsm8k=256, math=512, humaneval=768, mbpp=1024, ifeval=768 (keep to avoid OOM)

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

GPU_ID=${GPU_ID:-0}
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

SAVE_ROOT=${SAVE_ROOT:-./results_harness_llada}
RUN_TAG=${RUN_TAG:-}
export SAVE_ROOT RUN_TAG

# lm-eval task registry path (adds tasks like minerva_math500/humaneval_instruct/mbpp_instruct)
LM_EVAL_INCLUDE_PATH=${LM_EVAL_INCLUDE_PATH:-workspace/KLASS_experiment/data/tasks}
export LM_EVAL_INCLUDE_PATH

# Bench-specific limits (lm-eval --limit)
LIMIT_ALL=${LIMIT_ALL:-9999}
GSM8K_LIMIT=${GSM8K_LIMIT:-${LIMIT_ALL}}
MATH_LIMIT=${MATH_LIMIT:-${LIMIT_ALL}}
HUMANEVAL_LIMIT=${HUMANEVAL_LIMIT:-${LIMIT_ALL}}
MBPP_LIMIT=${MBPP_LIMIT:-${LIMIT_ALL}}
IFEVAL_LIMIT=${IFEVAL_LIMIT:-${LIMIT_ALL}}
BATCH_SIZE=${BATCH_SIZE:-1}
TASKS=${TASKS:-"gsm8k_cot humaneval_instruct mbpp_instruct ifeval"}
export GSM8K_LIMIT MATH_LIMIT HUMANEVAL_LIMIT MBPP_LIMIT IFEVAL_LIMIT BATCH_SIZE

# real_setting.sh uses HF_ALLOW_CODE_EVAL=1 for humaneval/mbpp
HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-1}
export HF_ALLOW_CODE_EVAL

# max_new_tokens per task (requested)
GSM8K_MAX_NEW_TOKENS=${GSM8K_MAX_NEW_TOKENS:-256}
MATH_MAX_NEW_TOKENS=${MATH_MAX_NEW_TOKENS:-512}
HUMANEVAL_MAX_NEW_TOKENS=${HUMANEVAL_MAX_NEW_TOKENS:-512} # 줄임..... 원래 768이었음
MBPP_MAX_NEW_TOKENS=${MBPP_MAX_NEW_TOKENS:-768} # 줄임..... 원래 1024였음
IFEVAL_MAX_NEW_TOKENS=${IFEVAL_MAX_NEW_TOKENS:-768}
export GSM8K_MAX_NEW_TOKENS MATH_MAX_NEW_TOKENS HUMANEVAL_MAX_NEW_TOKENS MBPP_MAX_NEW_TOKENS IFEVAL_MAX_NEW_TOKENS

# Context length caps
DEFAULT_MAX_LENGTH=${DEFAULT_MAX_LENGTH:-4096}
export DEFAULT_MAX_LENGTH

LLADA_MODEL_PATH=${LLADA_MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}
export LLADA_MODEL_PATH

# Model dtype override (useful for memory/debug: float16, bfloat16, float32, auto)
LLADA_DTYPE=${LLADA_DTYPE:-bfloat16}
export LLADA_DTYPE

# Optional memory saver
LLADA_LOAD_IN_8BIT=${LLADA_LOAD_IN_8BIT:-0}
export LLADA_LOAD_IN_8BIT

# Keep non-KLASS sampling knobs fixed (not part of KLASS idea)
# Baseline values follow Dream/eval_instruct/real_setting.sh
LLADA_TEMPERATURE=${LLADA_TEMPERATURE:-0.1}
LLADA_ALG=${LLADA_ALG:-klass}
export LLADA_TEMPERATURE LLADA_ALG

# KLASS hyperparameters
KLASS_CONF_THRESHOLD=${KLASS_CONF_THRESHOLD:-0.9}
KLASS_KL_THRESHOLD=${KLASS_KL_THRESHOLD:-0.001}
KLASS_HISTORY_LENGTH=${KLASS_HISTORY_LENGTH:-2}
KLASS_UNMASK_STRATEGY=${KLASS_UNMASK_STRATEGY:-all}
export KLASS_CONF_THRESHOLD KLASS_KL_THRESHOLD KLASS_HISTORY_LENGTH KLASS_UNMASK_STRATEGY

# Sweep target: diffusion steps
# If STEPS_LIST is set, use it for all tasks. Otherwise, derive per-task steps from max_new_tokens (k):
# k, k/2, k/3 (if divisible), k/4, k/5 (if divisible), k/8, k/16
STEPS_LIST=${STEPS_LIST:-}
export STEPS_LIST

make_run_tag() {
  local temp_tag="${LLADA_TEMPERATURE//./p}"
  local steps_tag
  if [[ -n "${STEPS_LIST}" ]]; then
    steps_tag="steps${STEPS_LIST// /_}"
  else
    steps_tag="steps_auto"
  fi
  echo "limits_gsm8k${GSM8K_LIMIT}_math${MATH_LIMIT}_humaneval${HUMANEVAL_LIMIT}_mbpp${MBPP_LIMIT}_ifeval${IFEVAL_LIMIT}_maxlen${DEFAULT_MAX_LENGTH}_temp${temp_tag}_${steps_tag}_llada"
}

if [[ -z "${RUN_TAG}" ]]; then
  RUN_TAG=$(make_run_tag)
fi
SAVE_DIR="${SAVE_ROOT}/sweep/${RUN_TAG}"
export SAVE_DIR

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

  if [[ "${LLADA_LOAD_IN_8BIT}" == "1" ]] && ! "${PYTHON_BIN}" -c "import bitsandbytes" >/dev/null 2>&1; then
    echo "Installing bitsandbytes (for 8-bit model loading) ..."
    "${PYTHON_BIN}" -m pip install -q bitsandbytes
  fi
}

require_code_eval_opt_in() {
  if [[ "${HF_ALLOW_CODE_EVAL:-}" != "1" ]]; then
    echo "ERROR: humaneval/mbpp run generated code. Set HF_ALLOW_CODE_EVAL=1 to proceed." >&2
    exit 3
  fi
}

build_steps_list() {
  local k="$1"
  local steps=()
  local seen=()

  add_step() {
    local v="$1"
    if [[ -z "${seen[$v]:-}" ]]; then
      steps+=("$v")
      seen[$v]=1
    fi
  }

  add_step "$k"
  if (( k % 2 == 0 )); then add_step $((k / 2)); fi
  if (( k % 3 == 0 )); then add_step $((k / 3)); fi
  if (( k % 4 == 0 )); then add_step $((k / 4)); fi
  if (( k % 5 == 0 )); then add_step $((k / 5)); fi
  if (( k % 8 == 0 )); then add_step $((k / 8)); fi
  if (( k % 16 == 0 )); then add_step $((k / 16)); fi

  echo "${steps[*]}"
}

get_steps_list() {
  local k="$1"
  if [[ -n "${STEPS_LIST}" ]]; then
    echo "${STEPS_LIST}"
  else
    build_steps_list "${k}"
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

  # Skip if results already exist (lm-eval writes under a model subdir)
  if find "${out_dir}" -type f -name "results_*.json" -print -quit 2>/dev/null | grep -q .; then
    echo "[SKIP] ${out_dir} already has results_*.json"
    return 0
  fi

  # Clean partial outputs to avoid mixing fp stats on rerun
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local pretrained
  pretrained=$("${PYTHON_BIN}" -c 'import os,sys; p=sys.argv[1]; print(os.path.realpath(p) if os.path.exists(p) else p)' "${LLADA_MODEL_PATH}")

  local model_args
  model_args="pretrained=${pretrained},device=cuda:0,batch_size=${BATCH_SIZE},dtype=${LLADA_DTYPE},stats_dir=${out_dir},max_new_tokens=${max_new_tokens},max_length=${max_length},steps=${steps},block_length=${max_new_tokens},alg=${LLADA_ALG},conf_threshold=${conf},kl_threshold=${kl},kl_history_length=${hist},unmask_strategy=${unmask},temperature=${LLADA_TEMPERATURE},load_in_8bit=${LLADA_LOAD_IN_8BIT}"

  local gen_kwargs
  gen_kwargs="max_gen_toks=${max_new_tokens}"

  local cmd=("${PYTHON_BIN}" eval/klass_lm_eval.py
    --model klass_llada
    --model_args "${model_args}"
    --tasks "${task}"
    --include_path "${LM_EVAL_INCLUDE_PATH}"
    --num_fewshot 0
    --gen_kwargs "${gen_kwargs}"
    --output_path "${out_dir}"
    --batch_size "${BATCH_SIZE}"
    --confirm_run_unsafe_code
  )

  cmd+=(--limit "${limit}")

  echo "[RUN] (lm-eval) task=${task} limit=${limit} steps=${steps} max_new_tokens=${max_new_tokens} max_length=${max_length} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${cmd[*]}"
  "${cmd[@]}"
}

task_enabled() {
  local task="$1"
  [[ " ${TASKS} " == *" ${task} "* ]]
}

ensure_lm_eval_deps

# Safety: humaneval/mbpp run generated code via code_eval
if echo " ${TASKS:-gsm8k_cot minerva_math500 humaneval_instruct mbpp_instruct ifeval} " | grep -qiE "(humaneval|mbpp)"; then
  require_code_eval_opt_in
fi

if task_enabled "gsm8k_cot"; then
  for steps in $(get_steps_list "${GSM8K_MAX_NEW_TOKENS}"); do
    run_lmeval_once "${SAVE_DIR}/lm_eval/llada/gsm8k_cot/steps${steps}" "gsm8k_cot" "${GSM8K_LIMIT}" "${GSM8K_MAX_NEW_TOKENS}" "${DEFAULT_MAX_LENGTH}" "${steps}" "${KLASS_CONF_THRESHOLD}" "${KLASS_KL_THRESHOLD}" "${KLASS_HISTORY_LENGTH}" "${KLASS_UNMASK_STRATEGY}"
  done
fi

# for steps in $(get_steps_list "${MATH_MAX_NEW_TOKENS}"); do
#   run_lmeval_once "${SAVE_DIR}/lm_eval/dream/minerva_math500/steps${steps}" "minerva_math500" "${MATH_LIMIT}" "${MATH_MAX_NEW_TOKENS}" "${DEFAULT_MAX_LENGTH}" "${steps}" "${KLASS_CONF_THRESHOLD}" "${KLASS_KL_THRESHOLD}" "${KLASS_HISTORY_LENGTH}" "${KLASS_UNMASK_STRATEGY}"
# done

if task_enabled "humaneval_instruct"; then
  for steps in $(get_steps_list "${HUMANEVAL_MAX_NEW_TOKENS}"); do
    run_lmeval_once "${SAVE_DIR}/lm_eval/llada/humaneval_instruct/steps${steps}" "humaneval_instruct" "${HUMANEVAL_LIMIT}" "${HUMANEVAL_MAX_NEW_TOKENS}" "${DEFAULT_MAX_LENGTH}" "${steps}" "${KLASS_CONF_THRESHOLD}" "${KLASS_KL_THRESHOLD}" "${KLASS_HISTORY_LENGTH}" "${KLASS_UNMASK_STRATEGY}"
  done
fi

if task_enabled "mbpp_instruct"; then
  for steps in $(get_steps_list "${MBPP_MAX_NEW_TOKENS}"); do
    run_lmeval_once "${SAVE_DIR}/lm_eval/llada/mbpp_instruct/steps${steps}" "mbpp_instruct" "${MBPP_LIMIT}" "${MBPP_MAX_NEW_TOKENS}" "${DEFAULT_MAX_LENGTH}" "${steps}" "${KLASS_CONF_THRESHOLD}" "${KLASS_KL_THRESHOLD}" "${KLASS_HISTORY_LENGTH}" "${KLASS_UNMASK_STRATEGY}"
  done
fi

if task_enabled "ifeval"; then
  for steps in $(get_steps_list "${IFEVAL_MAX_NEW_TOKENS}"); do
    run_lmeval_once "${SAVE_DIR}/lm_eval/llada/ifeval/steps${steps}/maxlen${DEFAULT_MAX_LENGTH}" "ifeval" "${IFEVAL_LIMIT}" "${IFEVAL_MAX_NEW_TOKENS}" "${DEFAULT_MAX_LENGTH}" "${steps}" "${KLASS_CONF_THRESHOLD}" "${KLASS_KL_THRESHOLD}" "${KLASS_HISTORY_LENGTH}" "${KLASS_UNMASK_STRATEGY}"
  done
fi

echo "Done. Outputs under ${SAVE_DIR}/lm_eval/llada/**"
