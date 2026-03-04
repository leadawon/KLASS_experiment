import os
import sys
import logging
from typing import List, Optional, Type, TypeVar, Union

import torch
import transformers
from tqdm import tqdm

# Make KLASS src importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KLASS_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_KLASS_SRC = os.path.join(_KLASS_ROOT, "src")
if _KLASS_SRC not in sys.path:
    sys.path.insert(0, _KLASS_SRC)

from forward_stats import ForwardPassCounter, write_forward_stats  # noqa: E402
from model.llada_klass import stable_confident_decode  # noqa: E402

from lm_eval.api.instance import Instance  # noqa: E402
from lm_eval.api.model import LM  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.models.utils import get_dtype  # noqa: E402
from lm_eval.__main__ import cli_evaluate  # noqa: E402


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")


def _parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError(f"Cannot parse bool from: {v!r}")


def _left_truncate(enc, max_len: int):
    if max_len is None:
        return enc
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    if input_ids.shape[1] <= max_len:
        return enc
    input_ids = input_ids[:, -max_len:]
    if attention_mask is not None:
        attention_mask = attention_mask[:, -max_len:]
    enc["input_ids"] = input_ids
    if attention_mask is not None:
        enc["attention_mask"] = attention_mask
    return enc


def _apply_until(text: str, until: List[str]) -> str:
    for s in until:
        if not s:
            continue
        text = text.split(s)[0]
    return text


class _BaseKlassLM(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 256,
        max_length: Optional[int] = 2048,
        trust_remote_code: Optional[bool] = True,
        load_in_8bit: Optional[Union[bool, str, int]] = False,
        stats_dir: Optional[str] = None,
        escape_until: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if not isinstance(pretrained, str):
            raise ValueError("This wrapper expects pretrained to be a local path or HF repo id string.")

        if isinstance(batch_size, str):
            batch_size = int(batch_size)
        self.batch_size_per_gpu = int(batch_size)

        self._device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(max_new_tokens)
        self.max_length = int(max_length)
        self.escape_until = bool(escape_until)

        use_8bit = _parse_bool(load_in_8bit)

        # NOTE: For bitsandbytes/8-bit models, calling .to(device) is not supported.
        # We instead rely on device_map to put the full model onto cuda:0.
        if use_8bit:
            try:
                bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                raise RuntimeError(
                    "Requested load_in_8bit=True but BitsAndBytesConfig is not available. "
                    "Install bitsandbytes and ensure your transformers version supports it."
                ) from e

            self.model = (
                transformers.AutoModel.from_pretrained(
                    pretrained,
                    torch_dtype=get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                    quantization_config=bnb_config,
                    device_map={"": 0},
                )
                .eval()
            )
        else:
            self.model = (
                transformers.AutoModel.from_pretrained(
                    pretrained,
                    torch_dtype=get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                )
                .eval()
                .to(self._device)
            )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)

        self._counter = ForwardPassCounter().attach(self.model)
        self._total_requests = 0

        self.stats_dir = stats_dir
        self._stats_model_name = os.path.basename(pretrained.rstrip("/"))

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        # Leverage lm-eval's helper
        from lm_eval.utils import simple_parse_args_string

        args = simple_parse_args_string(arg_string)
        if additional_config:
            args.update(additional_config)
        return cls(**args)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "KLASS lm-eval wrapper currently implements generate_until only. "
            "Use generation-based tasks (e.g., gsm8k_cot, humaneval, mbpp)."
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "KLASS lm-eval wrapper currently implements generate_until only."
        )

    def _write_stats(self, task_hint: str = "unknown") -> None:
        if not self.stats_dir:
            return

        extra = {
            "lm_eval": True,
            "task_hint": task_hint,
            "device": str(self.device),
            "max_new_tokens_default": self.max_new_tokens,
            "max_length": self.max_length,
        }

        write_forward_stats(
            self.stats_dir,
            model=self._stats_model_name,
            dataset=task_hint,
            total_items=self._total_requests,
            num_samples=1,
            total_forward_passes=self._counter.total_forward_passes,
            extra=extra,
            filename="forward_stats.json",
        )


@register_model("klass_dream")
class KlassDream(_BaseKlassLM):
    def __init__(
        self,
        pretrained: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 256,
        max_length: Optional[int] = 2048,
        trust_remote_code: Optional[bool] = True,
        stats_dir: Optional[str] = None,
        escape_until: Optional[bool] = False,
        steps: int = 128,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        alg: str = "klass",
        unmask_strategy: str = "all",
        conf_threshold: float = 0.9,
        kl_threshold: float = 0.01,
        kl_history_length: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            stats_dir=stats_dir,
            escape_until=escape_until,
            **kwargs,
        )

        self.steps = int(steps)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.unmask_strategy = unmask_strategy
        self.conf_threshold = float(conf_threshold)
        self.kl_threshold = float(kl_threshold)
        self.kl_history_length = int(kl_history_length)

    @torch.no_grad()
    def _generate_batch(self, prompts: List[str], max_new_tokens: int):
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        )

        max_prompt_len = max(self.max_length - max_new_tokens, 1)
        enc = _left_truncate(enc, max_prompt_len)

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        out = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            unmask_strategy=self.unmask_strategy,
            alg_temp=0.0,
            conf_threshold=self.conf_threshold,
            kl_threshold=self.kl_threshold,
            kl_history_length=self.kl_history_length,
            save_steps=False,
        )

        # Decode only the newly generated portion.
        responses = []
        for prompt_ids, seq in zip(input_ids, out.sequences):
            gen_ids = seq[len(prompt_ids) :]
            text = self.tokenizer.decode(gen_ids.tolist())
            if self.tokenizer.eos_token:
                text = text.split(self.tokenizer.eos_token)[0]
            responses.append(text)
        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        res: List[str] = []

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="KLASS(Dream) generate_until",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])

            # lm-eval supplies per-request max_gen_toks; we use the smallest in batch
            requested = min([int(g.get("max_gen_toks", self.max_new_tokens)) for g in gen_args])
            max_new_tokens = min(requested, self.max_new_tokens)
            if requested > self.max_new_tokens:
                eval_logger.warning(
                    f"Capping requested max_gen_toks={requested} to max_new_tokens={self.max_new_tokens} to avoid OOM."
                )
            until = gen_args[0].get("until", []) if gen_args else []

            responses = self._generate_batch(list(contexts), max_new_tokens=max_new_tokens)

            if not self.escape_until:
                responses = [_apply_until(r, until) for r in responses]

            res.extend(responses)
            self._total_requests += len(batch_requests)
            self._write_stats(task_hint="generate_until")

            pbar.update(len(batch_requests))

        return res


@register_model("klass_llada")
class KlassLLaDA(_BaseKlassLM):
    def __init__(
        self,
        pretrained: str,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 256,
        max_length: Optional[int] = 2048,
        trust_remote_code: Optional[bool] = True,
        stats_dir: Optional[str] = None,
        escape_until: Optional[bool] = False,
        steps: int = 128,
        block_length: int = 16,
        temperature: float = 0.0,
        alg: str = "klass",
        unmask_strategy: str = "all",
        conf_threshold: float = 0.9,
        kl_threshold: float = 0.01,
        kl_history_length: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            stats_dir=stats_dir,
            escape_until=escape_until,
            **kwargs,
        )

        if self.batch_size_per_gpu != 1:
            eval_logger.warning("KlassLLaDA currently runs with batch_size=1 (stable_confident_decode is single-sample).")
            self.batch_size_per_gpu = 1

        self.steps = int(steps)
        self.block_length = int(block_length)
        self.temperature = float(temperature)
        self.alg = alg
        self.unmask_strategy = unmask_strategy
        self.conf_threshold = float(conf_threshold)
        self.kl_threshold = float(kl_threshold)
        self.kl_history_length = int(kl_history_length)

    @torch.no_grad()
    def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        max_prompt_len = max(self.max_length - max_new_tokens, 1)
        enc = _left_truncate(enc, max_prompt_len)

        input_ids = enc["input_ids"].to(self.device)

        if max_new_tokens % self.block_length != 0:
            raise ValueError(
                f"For LLaDA, max_new_tokens ({max_new_tokens}) must be divisible by block_length ({self.block_length})."
            )

        seq, _used_steps = stable_confident_decode(
            self.model,
            self.tokenizer,
            input_ids_original=input_ids,
            gen_length=max_new_tokens,
            steps=self.steps,
            block_length=self.block_length,
            temperature=self.temperature,
            conf_threshold=self.conf_threshold,
            kl_threshold=self.kl_threshold,
            kl_history_length=self.kl_history_length,
            alg=self.alg,
            unmask_strategy=self.unmask_strategy,
        )

        gen_ids = seq[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(gen_ids.tolist())
        if self.tokenizer.eos_token:
            text = text.split(self.tokenizer.eos_token)[0]
        return text

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        res: List[str] = []

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="KLASS(LLaDA) generate_until",
        )

        for req in requests:
            context, gen_args = req.arguments
            requested = int(gen_args.get("max_gen_toks", self.max_new_tokens))
            max_new_tokens = min(requested, self.max_new_tokens)
            if requested > self.max_new_tokens:
                eval_logger.warning(
                    f"Capping requested max_gen_toks={requested} to max_new_tokens={self.max_new_tokens} to avoid OOM."
                )
            until = gen_args.get("until", [])

            text = self._generate_one(context, max_new_tokens=max_new_tokens)
            if not self.escape_until:
                text = _apply_until(text, until)

            res.append(text)
            self._total_requests += 1
            self._write_stats(task_hint="generate_until")
            pbar.update(1)

        return res


if __name__ == "__main__":
    cli_evaluate()
