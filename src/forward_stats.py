import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ForwardStats:
    model: str
    dataset: str
    total_items: int
    num_samples: int
    total_forward_passes: int
    avg_forward_passes_per_item: float
    avg_forward_passes_per_sample: float
    timestamp_utc: str
    extra: Dict[str, Any]


class ForwardPassCounter:
    def __init__(self) -> None:
        self.total_forward_passes = 0
        self._handle = None

    def reset(self) -> None:
        self.total_forward_passes = 0

    def attach(self, model) -> "ForwardPassCounter":
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

        def _pre_hook(_module, _inputs):
            self.total_forward_passes += 1

        self._handle = model.register_forward_pre_hook(_pre_hook)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_forward_stats(
    save_dir: str,
    *,
    model: str,
    dataset: str,
    total_items: int,
    num_samples: int,
    total_forward_passes: int,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "forward_stats.json",
    append_jsonl_path: Optional[str] = None,
) -> ForwardStats:
    os.makedirs(save_dir, exist_ok=True)

    total_items = int(total_items)
    num_samples = int(num_samples)
    total_forward_passes = int(total_forward_passes)

    avg_per_item = (total_forward_passes / total_items) if total_items > 0 else 0.0
    denom = total_items * num_samples
    avg_per_sample = (total_forward_passes / denom) if denom > 0 else 0.0

    stats = ForwardStats(
        model=model,
        dataset=dataset,
        total_items=total_items,
        num_samples=num_samples,
        total_forward_passes=total_forward_passes,
        avg_forward_passes_per_item=avg_per_item,
        avg_forward_passes_per_sample=avg_per_sample,
        timestamp_utc=_now_utc_iso(),
        extra=extra or {},
    )

    out_path = os.path.join(save_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(stats), f, indent=2)

    if append_jsonl_path:
        os.makedirs(os.path.dirname(append_jsonl_path), exist_ok=True)
        with open(append_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(stats), ensure_ascii=False) + "\n")

    return stats
