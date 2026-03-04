#!/usr/bin/env bash
set -euo pipefail

# Runs lm-eval-harness using KLASS wrappers on a single GPU.
# Usage examples (from /workspace/KLASS):
#   GPU_ID=3 TASKS=gsm8k_cot LIMIT=2 ./scripts/run_lm_eval.sh dream
#   GPU_ID=3 TASKS=ifeval LIMIT=2 ./scripts/run_lm_eval.sh dream
#   HF_ALLOW_CODE_EVAL=1 GPU_ID=3 TASKS=humaneval,mbpp LIMIT=2 ./scripts/run_lm_eval.sh llada
#   HF_ALLOW_CODE_EVAL=1 GPU_ID=3 TASKS=all LIMIT=2 ./scripts/run_lm_eval.sh dream

MODEL_KIND="${1:-}"
if [[ -z "$MODEL_KIND" ]]; then
  echo "Usage: $0 <dream|llada>"
  exit 2
fi

GPU_ID="${GPU_ID:-3}"
TASKS="${TASKS:-gsm8k_cot}"
LIMIT="${LIMIT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/klassvenv/bin/python}"

# Install lm_eval if missing (prefers the vendored harness in /workspace/Dream)
if ! "$PYTHON_BIN" -c "import lm_eval" >/dev/null 2>&1; then
  echo "lm_eval not found in venv; installing from /workspace/Dream/lm-evaluation-harness ..."
  "$PYTHON_BIN" -m pip install -e /workspace/Dream/lm-evaluation-harness
fi

OUT_DIR="results/lm_eval/${MODEL_KIND}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

# Convenience preset: mimic run_all.sh's generation-style suite (gsm8k/humaneval/mbpp) + ifeval.
# Note: "math" doesn't map cleanly to a single lm-eval task here, so it's not included.
if [[ "${TASKS}" == "all" ]]; then
  TASKS="gsm8k_cot,humaneval,mbpp,ifeval"
fi

# Safety: some tasks (e.g., humaneval/mbpp) execute model-generated code via the code_eval metric.
# Require explicit opt-in.
if echo ",${TASKS}," | grep -qiE ",(humaneval|mbpp),"; then
  if [[ "${HF_ALLOW_CODE_EVAL:-}" != "1" ]]; then
    echo "ERROR: TASKS includes humaneval/mbpp, which runs generated code." >&2
    echo "Set HF_ALLOW_CODE_EVAL=1 (after sandboxing) to proceed." >&2
    exit 3
  fi
fi

# IFEval requires a couple extra deps (lm-eval-harness calls these out as optional extras).
if echo ",${TASKS}," | grep -qiE ",(ifeval|leaderboard_ifeval),"; then
  if ! "$PYTHON_BIN" -c "import langdetect, immutabledict" >/dev/null 2>&1; then
    echo "Installing IFEval deps (langdetect, immutabledict) ..."
    "$PYTHON_BIN" -m pip install -q langdetect immutabledict
  fi
fi

case "$MODEL_KIND" in
  dream)
    MODEL_NAME="klass_dream"
    PRETRAINED="/workspace/KLASS/models/Dream-v0-Instruct-7B"
    MODEL_ARGS="pretrained=${PRETRAINED},device=cuda:0,batch_size=${BATCH_SIZE},stats_dir=${OUT_DIR},max_new_tokens=256,steps=128,alg=klass,conf_threshold=0.9,kl_threshold=0.01,kl_history_length=2,unmask_strategy=all"
    ;;
  llada)
    MODEL_NAME="klass_llada"
    PRETRAINED="/workspace/KLASS/models/LLaDA-8B-Instruct"
    MODEL_ARGS="pretrained=${PRETRAINED},device=cuda:0,batch_size=1,stats_dir=${OUT_DIR},max_new_tokens=256,steps=128,block_length=16,alg=klass,conf_threshold=0.9,kl_threshold=0.01,kl_history_length=2,unmask_strategy=all"
    ;;
  *)
    echo "Unknown model kind: $MODEL_KIND (expected dream or llada)"
    exit 2
    ;;
esac

echo "Output: $OUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
  HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-}" \
  "$PYTHON_BIN" eval/klass_lm_eval.py \
  --model "$MODEL_NAME" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --output_path "$OUT_DIR" \
  --batch_size "$BATCH_SIZE"
