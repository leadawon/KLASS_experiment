# Fairness Alignment Notes (2026-02-26)

This document records script-level fairness adjustments for Dream-original-based comparison runs.

## Baseline policy used
- Primary baseline protocol: `Dream/eval_instruct/eval_original_*.sh`
- Key normalization targets in this pass:
  - Task naming alignment (`humaneval_instruct`, `mbpp_instruct`)
  - Stop-condition consistency for Fast-dLLM (`escape_until` handling)
  - Context-window alignment (`4096`)
  - Diffusion-steps presentation alignment for GYR (`diffusion_steps=max_new_tokens` default upper bound)

## What was changed

### 1) Task name unification to Dream-original instruct tasks
- Updated Fast-dLLM sweep scripts:
  - `Fast-dLLM/scripts/run_fast_dllm_baselinesetting.sh`
  - `Fast-dLLM/scripts/run_fast_dllm_baselinesetting_with_cache.sh`
- Updated Dream_GYR_v12 sweep scripts:
  - `Dream_GYR_v12/eval_instruct/run_prob_gyr_limit200_qkv_sweep_split_base_humaneval.sh`
  - `Dream_GYR_v12/eval_instruct/run_prob_gyr_limit200_qkv_sweep_split_base_mbpp.sh`

Applied changes:
- `humaneval` -> `humaneval_instruct`
- `mbpp` -> `mbpp_instruct`

Reason:
- `humaneval` and `humaneval_instruct` share source data but are different evaluation protocols (prompt/prefix/filter behaviors differ).
- `mbpp` and `mbpp_instruct` are also protocol variants with different prompt-stop contracts.
- Mixing these task names is not a strict 1:1 evaluation.

### 2) Fast-dLLM `escape_until` fairness adjustment
- Updated in both Fast sweep scripts:
  - `escape_until=true` -> `escape_until=false`

Reason:
- With `escape_until=true`, Fast wrapper bypasses lm-eval task-level `until` truncation.
- That changes output boundary behavior versus configurations that respect task `until`, especially for code-generation tasks.
- For fairness, we enforce task-level stop handling consistency by using `escape_until=false`.

### 3) Context window alignment
- Fast scripts: set `max_length=4096` in `model_args`.
- GYR scripts: set `max_prompt_len=4096` in `model_args`.

Reason:
- Prevents hidden differences from shorter prompt truncation defaults.

### 4) Diffusion steps display alignment for GYR
- GYR scripts now default to:
  - `DIFFUSION_STEPS="${DIFFUSION_STEPS:-${MAX_NEW_TOKENS}}"`

Reason:
- GYR can early-stop; therefore diffusion steps act as an upper bound.
- This keeps external setting presentation aligned while preserving method behavior.

## Why prior setting was unfair (critical points)
1. Task-protocol mismatch (`*_instruct` vs non-instruct).
2. Fast-only stop behavior override (`escape_until=true`) causing non-uniform stopping logic.
3. Different context truncation caps causing unequal effective input conditioning.

## Reviewer-safety note
- Only script-level evaluation configuration was changed.
- No model code / algorithm implementation logic was modified.
- Purpose is protocol harmonization, not algorithm tuning.

## Chat-template audit against author-intended scripts

### Fast-dLLM (author-side scripts/docs)
- README points to `dream/eval.md`.
- In those examples/scripts, explicit `--apply_chat_template` is not used.
- Therefore, not enabling `--apply_chat_template` in custom Fast scripts is consistent with the provided author eval path.

### Slow-Fast-Sampling (author-side scripts)
- README quick start points to `slow-fast-sampling/scripts/*` runs.
- Those scripts also do not pass `--apply_chat_template`.
- Therefore, custom scripts without explicit chat-template flag are consistent with that author path.

### KLASS (author-side scripts)
- README points to `scripts/dream_humaneval.sh`, `scripts/dream_mbpp.sh` etc.
- These call `src/dream_evaluation.py`.
- In `src/dream_evaluation.py`, `tokenizer.apply_chat_template(...)` is called directly.
- So KLASS author-side Dream evaluation does apply chat templating internally.

Implication for current custom KLASS harness sweep:
- `KLASS/scripts/run_all_sweep_lm_eval_harness_baselinesetting.sh` uses `eval/klass_lm_eval.py` and does not pass `--apply_chat_template`.
- This custom path is not identical to KLASS's original `src/dream_evaluation.py` chat-template usage.
- This mismatch is documented here for transparency.

## Humaneval vs Humaneval_instruct clarification
- Source dataset is the same family (`openai/openai_humaneval`).
- But the evaluation protocol is different (prompt template, generation prefix, and prediction postprocessing differ).
- Therefore they should be treated as different benchmark variants for fair comparison setup.

