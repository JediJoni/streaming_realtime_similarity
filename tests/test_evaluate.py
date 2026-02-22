from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def test_evaluate_runs_after_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    reference = repo_root / "data" / "reference.csv"
    stream = repo_root / "data" / "stream_sample.csv"

    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "scores.parquet"
    out_log = out_dir / "run.log"

    # 1) Produce scores via CLI
    cmd_run = [
        sys.executable, "-m", "src.cli", "run",
        "--reference", str(reference),
        "--stream", str(stream),
        "--topk", "3",
        "--max-events", "4",
        "--min-score", "0.05",
        "--out", str(out_parquet),
        "--log", str(out_log),
    ]
    proc = subprocess.run(cmd_run, cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # 2) Evaluate
    cmd_eval = [sys.executable, "-m", "src.evaluate", "--scores", str(out_parquet)]
    proc2 = subprocess.run(cmd_eval, cwd=repo_root, capture_output=True, text=True)
    assert proc2.returncode == 0, proc2.stderr
    assert "match_rate" in proc2.stdout
    assert "zero_top1_rate" in proc2.stdout