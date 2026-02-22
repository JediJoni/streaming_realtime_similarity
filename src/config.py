from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    reference: str
    stream: str
    topk: int = 5
    max_events: int = 20
    sleep: float = 0.0
    out: str = "outputs/scores.parquet"
    log: str = "outputs/logs/run.log"
    min_score: float = 0.0  # used to flag "no match"


def load_json_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_config(cli_args: dict[str, Any], file_cfg: dict[str, Any] | None) -> RunConfig:
    merged = dict(cli_args)
    if file_cfg:
        merged.update({k: v for k, v in file_cfg.items() if v is not None})

    # required fields
    if not merged.get("reference") or not merged.get("stream"):
        raise ValueError("Config must include 'reference' and 'stream'.")

    return RunConfig(
        reference=str(merged["reference"]),
        stream=str(merged["stream"]),
        topk=int(merged.get("topk", 5)),
        max_events=int(merged.get("max_events", 20)),
        sleep=float(merged.get("sleep", 0.0)),
        out=str(merged.get("out", "outputs/scores.parquet")),
        log=str(merged.get("log", "outputs/logs/run.log")),
        min_score=float(merged.get("min_score", 0.0)),
    )