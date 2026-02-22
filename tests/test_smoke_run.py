from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_cli_smoke_run_writes_artifacts(tmp_path: Path) -> None:
    """
    Runs the CLI against the repo's sample data and asserts artifacts are produced.
    Uses a temp output directory to avoid polluting outputs/ in repeated test runs.
    """
    repo_root = Path(__file__).resolve().parents[1]
    reference = repo_root / "data" / "reference.csv"
    stream = repo_root / "data" / "stream_sample.csv"

    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "scores.parquet"
    out_log = out_dir / "run.log"

    cmd = [
        sys.executable, "-m", "src.cli", "run",
        "--reference", str(reference),
        "--stream", str(stream),
        "--topk", "3",
        "--max-events", "4",
        "--min-score", "0.05",
        "--out", str(out_parquet),
        "--log", str(out_log),
    ]

    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    assert out_parquet.exists()
    assert out_log.exists()

    df = pd.read_parquet(out_parquet)
    assert len(df) == 4
    assert "top1_score" in df.columns
    assert "is_match" in df.columns