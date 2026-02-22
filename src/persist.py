from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from src.config import RunConfig


def init_run_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("realtime_similarity")
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers in notebooks / repeated runs
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="w")
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def write_run_config_json(cfg: RunConfig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_scores_parquet(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    for r in records:
        r["timestamp_utc"] = ts

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)