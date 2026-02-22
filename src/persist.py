from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


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


def write_scores_parquet(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    for r in records:
        r["timestamp_utc"] = ts

    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)