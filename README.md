# streaming_realtime_similarity

A minimal **realtime-ish similarity scoring service** that streams incoming text events and returns the **top-k most similar items** from a reference corpus using **TF-IDF + cosine similarity**.

This project complements **streaming_similarity** by shifting from a batch/offline pipeline to an **online scoring** framing:

- ingestion (simulated stream)
- vectorization
- scoring
- logging + persisted artifacts
- config-driven runs
- lightweight evaluation signals

---

## Design Goals

- **Fast, deterministic baseline**: TF-IDF + cosine provides a clear, inspectable reference point.
- **Artifact discipline**: every run writes outputs, logs, and resolved configuration.
- **Explicit failure modes**: low-confidence matches are surfaced via `top1_score` and `is_match`.
- **CLI-first usability**: runnable end-to-end without notebooks.
- **Minimal but structured**: small system with clear module boundaries.

### Non-goals (by design)

- Not a deployed production service (no Kafka, no API server).
- Not semantic retrieval (no embeddings).
- Not optimized for large-scale indexing.

This repository establishes a clean baseline to extend later.

---

## What It Produces (Artifacts)

On each run, the CLI writes:

### `outputs/scores.parquet`

One row per processed event.

#### Schema

| column         | type         | description |
|---------------|--------------|-------------|
| event_id       | string       | Streamed event identifier |
| text           | string       | Event text |
| topk_ids       | list[str]    | Top-k matched reference item IDs |
| topk_scores    | list[float]  | Cosine similarities aligned to `topk_ids` |
| top1_score     | float        | Maximum similarity score |
| is_match       | bool         | `top1_score >= min_score` |
| timestamp_utc  | string       | ISO-8601 UTC timestamp |

---

### `outputs/run_config.json`

The fully resolved configuration used for the run (after merging `--config` + CLI overrides).

---

### `outputs/logs/run.log`

Structured log file containing:

- run parameters
- reference corpus size
- per-event scoring summary
- artifact write confirmation

---

## How Scoring Works

1. Fit a TF-IDF vectorizer on the **reference corpus**.
2. Transform each streamed event into the same vector space.
3. Compute cosine similarity between the event vector and all reference vectors.
4. Select the top-k matches.
5. Mark `is_match` if `top1_score >= min_score`.

This is a transparent baseline suitable for inspection and debugging.

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Run via CLI arguments

```bash
python -m src.cli run \
  --reference data/reference.csv \
  --stream data/stream_sample.csv \
  --topk 3 \
  --max-events 4 \
  --min-score 0.05
```

### 3) Run via config file

```bash
python -m src.cli run --config configs/dev.json
```

Override config values (CLI takes precedence):

```bash
python -m src.cli run --config configs/dev.json --topk 5 --min-score 0.1
```

---

## Evaluation (Lightweight)

A minimal evaluation script summarizes a run:

```bash
python -m src.evaluate --scores outputs/scores.parquet
```

It reports:

* `match_rate`: fraction of events where `top1_score >= min_score`
* `zero_top1_rate`: fraction of events with no vocabulary overlap (TF-IDF limitation)
* distribution summary of `top1_score`

---

## Reproducibility

Each run produces:

* `scores.parquet`
* `run_config.json`
* `run.log`

This makes runs auditable and allows parameter changes to be compared explicitly.

---

## Known Limitations

### 1. Vocabulary Dependence

TF-IDF relies on token overlap between query and reference corpus.

If no meaningful overlap exists, cosine similarity may be **0.0 across all items**.

Mitigation:

* `top1_score` makes this explicit.
* `is_match` thresholds low-confidence results.

---

### 2. Small Reference Corpora

Very small reference corpora may produce unstable or trivial matches.

Mitigation:

* This repository is a baseline.
* Scaling the reference corpus improves robustness.

---

### 3. No Semantic Understanding

Synonyms and paraphrases are not captured.

Mitigation:

* Future extension could introduce embedding-based retrieval as a comparison baseline.

---

## Inputs

### Reference CSV

Required columns:

* `item_id`
* `text`

### Stream CSV

Required columns:

* `event_id`
* `text`

---

## Tests

Run tests:

```bash
pip install -U pytest
pytest -q
```

Test coverage includes:

* CLI smoke run produces artifacts and expected columns
* Evaluation script runs successfully after a scoring run

---

## Project Structure

```
src/
  cli.py
  config.py
  ingestion.py
  vectorize.py
  scoring.py
  persist.py
  evaluate.py

tests/
  test_smoke_run.py
  test_evaluate.py

configs/
data/
outputs/
```

---

## Roadmap (Minimal & Disciplined)

* Add threshold strategy variants (fixed vs percentile).
* Add simple evaluation script reporting match-rate and score distribution.
* Optionally compare against embedding-based retrieval in a separate module.