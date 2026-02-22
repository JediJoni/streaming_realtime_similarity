# streaming_realtime_similarity

A minimal **realtime-ish similarity scoring service** that streams incoming text events and returns the **top-k most similar items** from a reference corpus using **TF-IDF + cosine similarity**.

This project complements **streaming_similarity** by shifting from a batch/offline pipeline to an **online scoring** framing:
- ingestion (simulated stream)
- vectorization
- scoring
- logging + persisted artifacts
- config-driven runs

---

## What it produces (artifacts)

On each run, the CLI writes:

- `outputs/scores.parquet`  
  One row per processed event, with:
  - `event_id`, `text`
  - `topk_ids`, `topk_scores`
  - `top1_score` (float)
  - `is_match` (bool, derived from `top1_score >= min_score`)
  - `timestamp_utc`

- `outputs/run_config.json`  
  The **resolved configuration** used for the run (after merging `--config` + CLI overrides).

- `outputs/logs/run.log`  
  Basic run metadata + per-event progress (including match flag).

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .