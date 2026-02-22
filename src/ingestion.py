from __future__ import annotations

import time
from pathlib import Path
import pandas as pd


def stream_rows(csv_path: Path, max_events: int = 20, sleep_s: float = 0.0):
    """
    Simulated stream: yields dict rows from a CSV.

    Expected columns: event_id, text
    """
    df = pd.read_csv(csv_path)

    required = {"event_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stream CSV missing columns: {sorted(missing)}")

    n = min(len(df), max_events)
    for i in range(n):
        row = df.iloc[i].to_dict()

        # guard against NaN
        row["event_id"] = "" if pd.isna(row["event_id"]) else row["event_id"]
        row["text"] = "" if pd.isna(row["text"]) else row["text"]

        yield row

        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)