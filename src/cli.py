from __future__ import annotations

import argparse
from pathlib import Path

from src.ingestion import stream_rows
from src.vectorize import TfidfVectorizerWrapper
from src.scoring import topk_cosine
from src.persist import init_run_logger, write_scores_parquet, write_run_config_json
from src.config import load_json_config, merge_config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="streaming_realtime_similarity",
        description="Realtime-ish similarity scoring (TF-IDF + cosine) with artifact outputs.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a smoke scoring pass over a reference corpus + stream.")
    run.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    run.add_argument("--reference", type=str, default=None, help="CSV with columns: item_id,text")
    run.add_argument("--stream", type=str, default=None, help="CSV with columns: event_id,text")
    run.add_argument("--topk", type=int, default=5, help="Top-k similar items to return per event.")
    run.add_argument("--max-events", type=int, default=20, help="Max streamed events to process.")
    run.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between events.")
    run.add_argument("--out", type=str, default="outputs/scores.parquet", help="Output parquet path.")
    run.add_argument("--log", type=str, default="outputs/logs/run.log", help="Log file path.")
    run.add_argument("--min-score", type=float, default=0.0, help="Min top1 score to count as a match.")
    return p


def cmd_run(args: argparse.Namespace) -> int:
    # 1) Load config file (optional), then merge with CLI args (CLI overrides file)
    file_cfg = load_json_config(args.config) if args.config else None
    cli_cfg = {
        "reference": args.reference,
        "stream": args.stream,
        "topk": args.topk,
        "max_events": args.max_events,
        "sleep": args.sleep,
        "out": args.out,
        "log": args.log,
        "min_score": args.min_score,
    }
    cfg = merge_config(cli_cfg, file_cfg)

    reference_path = Path(cfg.reference)
    stream_path = Path(cfg.stream)
    out_path = Path(cfg.out)
    log_path = Path(cfg.log)

    logger = init_run_logger(log_path)
    logger.info("Starting run")
    logger.info(
        "reference=%s stream=%s topk=%s max_events=%s sleep=%s min_score=%s out=%s",
        reference_path, stream_path, cfg.topk, cfg.max_events, cfg.sleep, cfg.min_score, out_path
    )

    # Save resolved run config as an artifact
    write_run_config_json(cfg, out_path.parent / "run_config.json")

    # 2) Fit vectorizer on reference corpus
    vec = TfidfVectorizerWrapper()
    ref_df, ref_matrix = vec.fit_reference_csv(reference_path)
    ref_ids = ref_df["item_id"].tolist()
    logger.info("Loaded reference rows=%d", len(ref_df))

    # 3) Stream events and score
    results = []
    processed = 0

    for event in stream_rows(stream_path, max_events=cfg.max_events, sleep_s=cfg.sleep):
        processed += 1
        event_id = str(event["event_id"])
        text = str(event["text"])

        q = vec.transform_query(text)
        ids, scores = topk_cosine(q, ref_matrix, ref_ids, k=cfg.topk)

        top1_score = float(scores[0]) if scores else 0.0
        is_match = bool(top1_score >= cfg.min_score)

        results.append({
            "event_id": event_id,
            "text": text,
            "topk_ids": ids,
            "topk_scores": scores,
            "top1_score": top1_score,
            "is_match": is_match,
        })

        logger.info(
            "Scored event %s -> top1=%s score=%.4f match=%s",
            event_id, (ids[0] if ids else None), top1_score, is_match
        )

    # 4) Persist
    write_scores_parquet(results, out_path)
    logger.info("Wrote %d scored events to %s", processed, out_path)
    logger.info("Done")

    print(f"OK: wrote {processed} events -> {out_path}")
    print(f"OK: log -> {log_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())