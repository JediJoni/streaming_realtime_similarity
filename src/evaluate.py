# src/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def evaluate_scores(path: Path) -> dict:
    df = pd.read_parquet(path)

    required = {"event_id", "top1_score", "is_match"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in scores parquet: {sorted(missing)}")

    n = len(df)
    match_rate = float(df["is_match"].mean()) if n else 0.0
    zero_rate = float((df["top1_score"] == 0.0).mean()) if n else 0.0

    summary = df["top1_score"].describe().to_dict()

    return {
        "rows": n,
        "match_rate": match_rate,
        "zero_top1_rate": zero_rate,
        "top1_score_summary": summary,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate scoring outputs (basic run summary).")
    p.add_argument(
        "--scores",
        type=str,
        default="outputs/scores.parquet",
        help="Path to outputs/scores.parquet",
    )
    args = p.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores parquet not found: {scores_path}")

    report = evaluate_scores(scores_path)

    print(f"rows: {report['rows']}")
    print(f"match_rate: {report['match_rate']:.3f}")
    print(f"zero_top1_rate: {report['zero_top1_rate']:.3f}")
    print("top1_score_summary:")
    for k, v in report["top1_score_summary"].items():
        # describe() returns numpy types sometimes; print safely
        try:
            v = float(v)
            print(f"  {k}: {v:.6f}")
        except Exception:
            print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())