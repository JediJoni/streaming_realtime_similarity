from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.ingestion import stream_rows
from src.vectorize import TfidfVectorizerWrapper
from src.scoring import topk_cosine
from src.persist import init_run_logger, write_scores_parquet


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="streaming_realtime_similarity",
        description="Realtime-ish similarity scoring (TF-IDF + cosine) with artifact outputs.",
    )
    sub = p.add_subparsers(dest="command")
    # Don't rely on required=True for portability; enforce manually.
    run = sub.add_parser("run", help="Run a smoke scoring pass over a reference corpus + stream.")
    run.add_argument("--reference", type=str, required=True, help="CSV with columns: item_id,text")
    run.add_argument("--stream", type=str, required=True, help="CSV with columns: event_id,text")
    run.add_argument("--topk", type=int, default=5, help="Top-k similar items per event.")
    run.add_argument("--max-events", type=int, default=20, help="Max streamed events to process.")
    run.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between events.")
    run.add_argument("--out", type=str, default="outputs/scores.parquet", help="Output parquet path.")
    run.add_argument("--log", type=str, default="outputs/logs/run.log", help="Log file path.")
    return p


def cmd_run(args: argparse.Namespace) -> int:
    reference_path = Path(args.reference)
    stream_path = Path(args.stream)
    out_path = Path(args.out)
    log_path = Path(args.log)

    # Early, loud checks
    if not reference_path.exists():
        print(f"ERROR: reference file not found: {reference_path}", file=sys.stderr)
        return 2
    if not stream_path.exists():
        print(f"ERROR: stream file not found: {stream_path}", file=sys.stderr)
        return 2

    logger = init_run_logger(log_path)
    logger.info("Starting run")
    logger.info("reference=%s stream=%s topk=%s max_events=%s sleep=%s out=%s",
                reference_path, stream_path, args.topk, args.max_events, args.sleep, out_path)

    vec = TfidfVectorizerWrapper()
    ref_df, ref_matrix = vec.fit_reference_csv(reference_path)
    logger.info("Loaded reference rows=%d", len(ref_df))

    results = []
    processed = 0

    for event in stream_rows(stream_path, max_events=args.max_events, sleep_s=args.sleep):
        processed += 1
        event_id = str(event.get("event_id", ""))
        text = str(event.get("text", ""))

        q = vec.transform_query(text)
        ids, scores = topk_cosine(q, ref_matrix, ref_df["item_id"].tolist(), k=args.topk)

        results.append({
            "event_id": event_id,
            "text": text,
            "topk_ids": ids,
            "topk_scores": scores,
        })

        logger.info("Scored event %s -> top1=%s score=%.4f",
                    event_id, ids[0] if ids else None, scores[0] if scores else float("nan"))

    write_scores_parquet(results, out_path)
    logger.info("Wrote %d scored events to %s", processed, out_path)

    # Also print a short confirmation to stdout (useful for smoke tests)
    print(f"OK: wrote {processed} events -> {out_path}", flush=True)
    print(f"OK: log -> {log_path}", flush=True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    if args.command == "run":
        return cmd_run(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())