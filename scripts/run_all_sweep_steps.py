#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    task: str
    out_dir: Path
    log_path: Path
    limit: int
    max_new_tokens: int
    max_length: int
    steps: int


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError as e:
        raise SystemExit(f"Invalid int for {name}={val!r}") from e


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError as e:
        raise SystemExit(f"Invalid float for {name}={val!r}") from e


def _env_str(name: str, default: str) -> str:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _parse_int_list(raw: str) -> list[int]:
    items: list[int] = []
    for part in shlex.split(raw):
        try:
            items.append(int(part))
        except ValueError as e:
            raise SystemExit(f"Expected int list, got {raw!r}") from e
    return items


def _file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return False


def _run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def _gpu_free_mem_mib(gpu_id: int) -> int | None:
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None

    txt = (p.stdout or "").strip().splitlines()
    if not txt:
        return None
    try:
        return int(txt[0].strip())
    except ValueError:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    # Import tqdm lazily so we can print a friendlier error.
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        print(
            "tqdm is not installed. Install it in your venv (pip install tqdm) "
            "or re-run scripts/run_all_sweep.sh which auto-installs it.",
            file=sys.stderr,
        )
        return 2

    gpu_id = _env_int("GPU_ID", 2)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    skip_gpu_check = _env_int("SKIP_GPU_CHECK", 0)
    min_free_mem_mib = _env_int("MIN_FREE_MEM_MIB", 20000)
    if skip_gpu_check != 1:
        free_mib = _gpu_free_mem_mib(gpu_id)
        if free_mib is not None and free_mib < min_free_mem_mib:
            print(
                f"[ERROR] GPU {gpu_id} has only {free_mib} MiB free "
                f"(need >= {min_free_mem_mib} MiB by default).",
                file=sys.stderr,
            )
            return 2

    raw_save_dir = os.environ.get("SAVE_DIR")
    if raw_save_dir is not None and raw_save_dir.strip() != "":
        save_dir = Path(raw_save_dir)
    else:
        save_root = Path(_env_str("SAVE_ROOT", "./results_harness"))
        run_tag = _env_str("RUN_TAG", "")
        if not run_tag:
            import time

            run_tag = time.strftime("%Y%m%d_%H%M%S")
        save_dir = save_root / "sweep" / run_tag

    batch_size = _env_int("BATCH_SIZE", 1)

    dream_model_path = _env_str("DREAM_MODEL_PATH", "./models/Dream-v0-Instruct-7B")
    # Resolve pretrained path similarly to the bash wrapper.
    pretrained = str(Path(dream_model_path).resolve())

    dream_dtype = _env_str("DREAM_DTYPE", "auto")

    dream_load_in_8bit = _env_int("DREAM_LOAD_IN_8BIT", 0) == 1

    # Fixed knobs (match scripts/run_all.sh defaults)
    dream_temperature = _env_float("DREAM_TEMPERATURE", 0.2)
    dream_top_p = _env_float("DREAM_TOP_P", 0.95)
    dream_alg = _env_str("DREAM_ALG", "klass")

    klass_conf = _env_float("KLASS_CONF_THRESHOLD", 0.9)
    klass_kl = _env_float("KLASS_KL_THRESHOLD", 0.001)
    klass_hist = _env_int("KLASS_HISTORY_LENGTH", 2)
    klass_unmask = _env_str("KLASS_UNMASK_STRATEGY", "all")

    # Limits
    gsm8k_limit = _env_int("GSM8K_LIMIT", 200)
    ifeval_limit = _env_int("IFEVAL_LIMIT", 50)

    # Generation
    gsm8k_max_new_tokens = _env_int("GSM8K_MAX_NEW_TOKENS", 256)
    ifeval_max_new_tokens = _env_int("IFEVAL_MAX_NEW_TOKENS", 256)

    # Context caps
    gsm8k_max_length = _env_int("GSM8K_MAX_LENGTH", 4096)
    ifeval_max_length_list = _parse_int_list(_env_str("IFEVAL_MAX_LENGTH_LIST", "4096"))

    # Optional: automatically find the largest IFEval max_new_tokens that doesn't OOM.
    # This is useful when users want to push IFEval generations while keeping the rest constant.
    ifeval_autotune = _env_int("IFEVAL_AUTOTUNE", 0) == 1
    ifeval_candidates = _parse_int_list(
        _env_str("IFEVAL_MAX_NEW_TOKENS_CANDIDATES", "1280 1024 768 512 384 256 192 128")
    )

    steps_list = _parse_int_list(_env_str("STEPS_LIST", "64 128 256 512"))

    # Build run list: IFEval first for each steps, then GSM8K.
    runs: list[RunSpec] = []
    def build_cmd(spec: RunSpec) -> list[str]:
        model_args_parts = [
            f"pretrained={pretrained},"
            f"device=cuda:0,"
            f"batch_size={batch_size},"
            f"dtype={dream_dtype},"
            f"stats_dir={spec.out_dir},"
            f"max_new_tokens={spec.max_new_tokens},"
            f"max_length={spec.max_length},"
            f"steps={spec.steps},"
            f"alg={dream_alg},"
            f"conf_threshold={klass_conf},"
            f"kl_threshold={klass_kl},"
            f"kl_history_length={klass_hist},"
            f"unmask_strategy={klass_unmask},"
            f"temperature={dream_temperature},"
            f"top_p={dream_top_p},",
        ]

        if dream_load_in_8bit:
            model_args_parts.append("load_in_8bit=True")

        model_args = "".join(model_args_parts)

        return [
            sys.executable,
            "eval/klass_lm_eval.py",
            "--model",
            "klass_dream",
            "--model_args",
            model_args,
            "--tasks",
            spec.task,
            "--output_path",
            str(spec.out_dir),
            "--batch_size",
            str(batch_size),
            "--limit",
            str(spec.limit),
        ]

    def autotune_ifeval_max_new_tokens() -> int:
        # Probe at the "worst" (largest) settings among the sweep.
        probe_steps = max(steps_list) if steps_list else 128
        probe_max_len = max(ifeval_max_length_list) if ifeval_max_length_list else 4096
        probe_limit = 1

        for cand in ifeval_candidates:
            probe_out = save_dir / "_preflight" / "ifeval_max_new_tokens" / f"steps{probe_steps}" / f"maxlen{probe_max_len}" / f"tok{cand}"
            probe_log = probe_out / "run.log"

            spec = RunSpec(
                task="ifeval",
                out_dir=probe_out,
                log_path=probe_log,
                limit=probe_limit,
                max_new_tokens=cand,
                max_length=probe_max_len,
                steps=probe_steps,
            )

            rc = _run_cmd(build_cmd(spec), probe_log)
            if rc == 0:
                print(f"[INFO] IFEVAL_AUTOTUNE picked max_new_tokens={cand} (probe steps={probe_steps}, max_length={probe_max_len}).")
                return cand

            if _file_contains(probe_log, "CUDA out of memory") or _file_contains(probe_log, "OutOfMemoryError"):
                continue

            # Non-OOM failure: stop early so user sees the real error.
            raise SystemExit(f"IFEVAL_AUTOTUNE failed for non-OOM reason. See log: {probe_log}")

        raise SystemExit(
            "IFEVAL_AUTOTUNE: all candidates OOM. "
            "Lower IFEVAL_MAX_NEW_TOKENS_CANDIDATES or reduce IFEVAL_MAX_LENGTH_LIST / steps." 
        )

    if ifeval_autotune and ifeval_limit > 0:
        ifeval_max_new_tokens = autotune_ifeval_max_new_tokens()

    for steps in steps_list:
        sweep_id = f"steps{steps}"
        if ifeval_limit > 0:
            for max_len in ifeval_max_length_list:
                out_dir = save_dir / "lm_eval" / "dream" / "ifeval" / sweep_id / f"maxlen{max_len}"
                runs.append(
                    RunSpec(
                        task="ifeval",
                        out_dir=out_dir,
                        log_path=out_dir / "run.log",
                        limit=ifeval_limit,
                        max_new_tokens=ifeval_max_new_tokens,
                        max_length=max_len,
                        steps=steps,
                    )
                )

        if gsm8k_limit > 0:
            out_dir = save_dir / "lm_eval" / "dream" / "gsm8k_cot" / sweep_id
            runs.append(
                RunSpec(
                    task="gsm8k_cot",
                    out_dir=out_dir,
                    log_path=out_dir / "run.log",
                    limit=gsm8k_limit,
                    max_new_tokens=gsm8k_max_new_tokens,
                    max_length=gsm8k_max_length,
                    steps=steps,
                )
            )

    # (pretrained already resolved above)

    print(f"[INFO] SAVE_DIR={save_dir}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} (GPU_ID={gpu_id})")
    print(
        f"[INFO] Runs: {len(runs)} total (steps={steps_list}, ifeval_max_length={ifeval_max_length_list}, "
        f"ifeval_max_new_tokens={ifeval_max_new_tokens}, ifeval_autotune={ifeval_autotune})"
    )

    bar = tqdm(runs, desc="sweep", unit="run", dynamic_ncols=True)
    for spec in bar:
        label = f"{spec.task} steps={spec.steps}"
        if spec.task == "ifeval":
            label += f" maxlen={spec.max_length}"
        bar.set_postfix_str(label)

        cmd = build_cmd(spec)
        rc = _run_cmd(cmd, spec.log_path)
        if rc != 0:
            print(f"\n[ERROR] Run failed: {label}", file=sys.stderr)
            print(f"        Log: {spec.log_path}", file=sys.stderr)
            return rc

    print("\nDone. Outputs:")
    print(f"- GSM8K(lm-eval): {save_dir}/lm_eval/dream/gsm8k_cot/**")
    print(f"- IFEval(lm-eval): {save_dir}/lm_eval/dream/ifeval/**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
