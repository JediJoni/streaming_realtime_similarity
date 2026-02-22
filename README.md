# streaming_realtime_similarity

A minimal **realtime-ish similarity scoring service** that streams incoming text events and returns **top-k most similar items** from a reference corpus using **TF-IDF + cosine similarity**.

This project complements **streaming_similarity** by shifting from a batch/offline pipeline to an **online scoring** framing:
- ingestion (simulated stream)
- vectorization
- scoring
- logging + persisted artifacts

---

## What it produces (artifacts)

On a run, the CLI writes:

- `outputs/scores.parquet`  
  Columns: `event_id`, `text`, `topk_ids`, `topk_scores`, `timestamp_utc`

- `outputs/logs/run.log`  
  Basic run metadata + per-event progress

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .