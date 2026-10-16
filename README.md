# Adaptive ML Platform

Serve ML models, watch for drift, and roll back automatically.

A model-agnostic serving platform built with FastAPI and React. It runs three deliberately different models (tabular fraud detection, satellite image classification, and AI-generated-text detection) behind one backend, monitors the data each model receives, and when drift is sustained it rolls traffic back to the last stable version and records an incident.

## Why this exists

Training a model is the easy part. The harder part is what happens after deployment: how requests are batched, how new versions are rolled out, how you notice that live inputs no longer look like the training data, and what the system does about it. This project builds that layer end to end and tests it with injected drift on three different data types.

The models themselves are intentionally simple and use well-known public datasets, so the serving and reliability engineering is the thing being evaluated.

## Demo results

For each model, drifted traffic was injected into a running system. In every run the monitor flagged three consecutive breaching windows and the platform rolled back to the previous version on its own, with one incident recorded.

| Model | Injection | Signal (threshold) | Peak score | Injection to rollback | Result |
|---|---|---|---|---|---|
| fraud | Large `Amount` values and 8 PCA features shifted by +6 | PSI (0.25) | 4.56 | 27.6 s | `auto_rollback` to v1 |
| satellite | Inverted, high-contrast, noisy EuroSAT images | Cosine distance (0.15) | 0.399 | 61.3 s | `auto_rollback` to v1 |
| ai_text | Current-generation LLM text vs. a detector trained on 2022-era ChatGPT text | Cosine distance (0.08) | 0.104 | 138.0 s | `auto_rollback` to v1 |

Read these numbers with some care:

- Rollback times depend on the demo scripts' request rates, the window size, and the 15-second recompute interval. They are not latency guarantees.
- The ai_text threshold of 0.08 was set from the plateau (about 0.10) measured in an earlier run of this same injection, and the peak clears it narrowly. That run shows the mechanism works at that threshold. It does not independently validate the threshold.
- `v2` of each model is a byte-identical copy of `v1`, created by `scripts/seed_v2_versions.py` so the rollback has a previous version to return to. It is not a retrained model.

## Highlights

- **One serving abstraction, three modalities.** A common `ModelVersion` interface (`predict`, `predict_batch`, `embed`) lets tabular, vision, and NLP models share the same registry, batcher, router, and drift monitor.
- **Async micro-batching with fault isolation.** Requests queue and flush as one batched forward pass, by size or by time window. If a batch fails, the batcher retries each item alone so one bad input does not fail its neighbours.
- **Sticky canary routing.** Consistent hashing assigns users to model versions, so a user keeps seeing the same version during a rollout.
- **Observability built in.** Every request is logged with latency, version, confidence, and errors. `/metrics` serves Prometheus text format from in-memory counters, and an async writer persists the audit log to SQLite off the request path.
- **Statistical drift detection.** Population Stability Index (PSI) with Laplace smoothing for tabular features, and cosine distance against a reference embedding centroid for the image and text models.
- **Closed-loop response.** After three consecutive breaching windows the system rolls the stable version back (or pulls a drifting canary) and writes an incident record.
- **React dashboard.** Per-model health, latency percentiles and error rate over time, drift trend with threshold and breach markers, an incident timeline, and a registry view.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │               FastAPI app                │
                    │  /predict/{fraud,satellite,ai_text}       │
                    │  /metrics  /drift  /stats  /incidents     │
                    └───────────────┬───────────────────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
          │   Router     │   │ MicroBatcher │   │   MetricsStore    │
          │ sticky hash, │   │ per-model    │   │ sync counters +   │
          │ canary/stable│   │ batch/window │   │ async SQLite log  │
          └──────┬───────┘   └──────┬───────┘   └─────────┬─────────┘
                 │                  │                      │
                 └────────┬─────────┘                      ▼
                          ▼                         ┌──────────────┐
                  ┌───────────────┐                 │ DriftMonitor │
                  │ ModelRegistry │◄────────────────┤ PSI / cosine │
                  │ manifest.json │                 └──────┬───────┘
                  │ hot-swap load │                        ▼
                  └───────────────┘                ┌────────────────────┐
                                                    │ IncidentResponder  │
                                                    │ rollback / canary  │
                                                    │ kill + incident log│
                                                    └────────────────────┘
```

The React dashboard talks only to the FastAPI read endpoints. It has no knowledge of how state is stored.

## Models

| Model | Modality | Dataset | Architecture |
|---|---|---|---|
| `fraud` | Tabular | Kaggle Credit Card Fraud Detection | LightGBM with `scale_pos_weight` for the ~0.17% positive class |
| `satellite` | Image | EuroSAT (`tanganke/eurosat` on Hugging Face) | Fine-tuned MobileNetV3-Small, 10 land-use classes |
| `ai_text` | Text | HC3 (human vs. ChatGPT answers) | Fine-tuned DistilBERT, binary human/AI |

Training notebooks are in `notebooks/`. Validation metrics for each model are saved in `models/<name>/metrics_v1.json`.

The AI-text detector reaches 99.47% accuracy, 99.16% F1, and 99.99% ROC-AUC on a held-out HC3 split, grouped by question to avoid leakage. That reflects how separable HC3 is, not general AI-text-detection ability. The drift demo above is the more honest test.

## How drift detection works

| Model | Signal | Threshold | Notes |
|---|---|---|---|
| `fraud` | Max per-feature PSI vs. a reference sample | 0.25 | Quantile bins from the reference sample, Laplace smoothing |
| `satellite` | Cosine distance between live and reference embedding centroids | 0.15 | MobileNetV3 pooled features |
| `ai_text` | Cosine distance between live and reference embedding centroids | 0.08 | DistilBERT CLS embeddings |

- A rolling window of recent samples (default 200) is compared against a reference captured at training time.
- No score is computed until the window holds at least 150 samples. PSI is unreliable on small windows (see Engineering notes).
- The monitor recomputes every 15 seconds. Action fires only after 3 consecutive breaching windows, so one noisy window does not trigger a rollback.
- Drift on a canary pulls the canary. Drift on the stable version rolls back to the previous registered version. Every intervention is written to an `incidents` table.

The ai_text threshold started as an untested guess of 0.15. Text from a current-generation LLM, run through a detector trained on 2022-era ChatGPT text, plateaued around 0.10 (0.0999 with a 15-sample pool, 0.1046 with a 30-sample pool across 10 prompt types), so the threshold was lowered to 0.08.

## Evaluating the drift detector

`scripts/compare_drift_detectors.py` compares this project's PSI detector against scipy's two-sample KS test and Evidently on the fraud model, under controlled shifts of increasing severity. For each condition it reports each detector's detection rate, the false-positive rate with no shift, and the fraud model's real PR-AUC drop. It shifts both the most and least important features, to test whether a drift alarm tracks actual performance loss.

```powershell
pip install evidently scipy
python -m scripts.compare_drift_detectors
```

Results are written to `docs/drift_detector_comparison.csv`.

## Performance

### Micro-batching (real registered models, 300 requests, concurrency 30)

| Model | Unbatched | Batched | Speedup | p99 unbatched / batched |
|---|---|---|---|---|
| fraud | 118.69 req/s | 1605.51 req/s | 13.53x | 398.95 ms / 23.42 ms |
| satellite | 84.69 req/s | 202.08 req/s | 2.39x | 466.60 ms / 269.63 ms |
| ai_text | 64.15 req/s | 99.98 req/s | 1.56x | 560.04 ms / 339.44 ms |

The fraud speedup is mostly not compute amortization, since a single LightGBM row prediction is sub-millisecond. Collapsing many concurrent per-item thread-pool submissions into a few batched ones removes GIL and thread-pool contention. The neural models gain from batched tensor operations, partly offset by per-item image and text preprocessing. For satellite, `ToTensor()` dominated preprocessing, not resizing.

An early mock-based benchmark showed batching as slightly slower for fraud (0.95x) because the mock's cost model did not reflect the real model. All numbers above come from the real registry.

### Thread-pool sizing (5 averaged runs plus warm-up, 12 logical cores)

| Model | Best `max_workers` | Mean throughput |
|---|---|---|
| fraud | 8 | 109.99 req/s (±4.18) |
| satellite | 4 | 105.28 req/s (±16.74) |
| ai_text | 8 | 72.66 req/s (±7.16) |

A single-run sweep had picked `max_workers=16` for ai_text. Averaging revealed the real optimum at 8. A hypothesis that PyTorch intra-op thread oversubscription was the bottleneck was tested directly (`torch.set_num_threads` 1, 2, 4, 10) and ruled out: throughput stayed flat at 44 to 48 req/s.

## Dashboard

React, TypeScript, Vite, Tailwind CSS v4, shadcn/ui, Recharts, and TanStack Query (5-second polling).

- **Model panels:** active version, canary state, p50/p95/p99, request count, latency sparkline. A panel's border changes colour when that model is in canary or drift breach.
- **Latency and error rate:** binned p50/p95/p99 lines with an error-rate overlay, per model.
- **Drift and incident timeline:** drift scores for every version over time, the threshold line, breach markers, and a log of interventions.
- **Registry table:** versions, active and canary versions, request and error counts.

## Quickstart

Developed on Windows with PowerShell and Python 3.11.

### 1. Environment

```powershell
git clone <your-repo-url>
cd adaptive-ml-platform
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Core dependencies: `fastapi`, `uvicorn[standard]`, `pydantic`, `scikit-learn`, `lightgbm`, `torch`, `torchvision`, `transformers`, `datasets`, `pandas`, `numpy`, `scipy`, `pyarrow`, `pillow`, `httpx`, `python-dotenv`, `openai`, `certifi`, `pytest`.

### 2. Data and models

1. Download the Kaggle Credit Card Fraud dataset to `data/raw/creditcard.csv`.
2. Download HC3's `all.jsonl` from the Hugging Face dataset page to `data/hc3_all.jsonl`. The `Hello-SimpleAI/HC3` repo uses a legacy loading script that recent versions of `datasets` no longer support, so load the raw JSONL directly.
3. Run the notebooks in `notebooks/` in order. They write model weights, metrics, and drift reference artifacts into `models/`.

### 3. Registry and routing files

`serving/registry/manifest.json` has one entry per model (shown for `fraud`):

```json
{
  "fraud": {
    "active_version": "v1",
    "versions": {
      "v1": {
        "path": "models/fraud",
        "metrics_file": "models/fraud/metrics_v1.json",
        "reference_file": "models/fraud/reference_sample_v1.parquet"
      }
    }
  }
}
```

`serving/routing_config.json`:

```json
{
  "fraud": {"stable": "v1", "canary": null, "canary_percent": 0},
  "satellite": {"stable": "v1", "canary": null, "canary_percent": 0},
  "ai_text": {"stable": "v1", "canary": null, "canary_percent": 0}
}
```

Then register the demo `v2` versions:

```powershell
python -m scripts.seed_v2_versions
```

### 4. Run the API

```powershell
uvicorn serving.app:app
```

Do not use `--reload` while running the demos. A rollback rewrites `manifest.json` and `routing_config.json`, and uvicorn's file watcher will restart the server mid-request.

### 5. Run the dashboard

```powershell
cd dashboard
npm install
npm run dev
```

Set `VITE_API_BASE_URL=http://localhost:8000` in `dashboard/.env`.

### 6. Run the drift demos

```powershell
python -m scripts.generate_traffic      # optional: baseline traffic for all three models
python -m scripts.run_full_demo         # runs all three demos; starts and stops the API itself
```

Add `--interactive` to pause before each model. Stop any running API first, since `run_full_demo` manages the server itself.

To run one demo by hand, reset state, restart the API, then run the script. State lives in memory, so editing the JSON files has no effect until the server restarts:

```powershell
python -m scripts.reset_for_demo
uvicorn serving.app:app
python -m scripts.demo_ood_fraud        # or demo_ood_satellite / demo_ood_ai_text
python -m scripts.inspect_incidents
```

The AI-text demo reads `GROQ_API_KEY` from a `.env` file to generate current-generation text. Without it, it falls back to a few hand-written samples, which is weaker evidence.

### 7. Tests

```powershell
python -m tests.test_registry
python -m tests.test_parity_and_batcher
python -m tests.test_telemetry
python -m tests.test_drift
python -m pytest tests/test_incident_response.py
```

## API reference

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict/fraud` | JSON with `Time`, `Amount`, `V1`..`V28` |
| POST | `/predict/satellite` | Multipart image upload (`file`) |
| POST | `/predict/ai_text` | JSON `{"text": "..."}` |
| GET | `/metrics` | Prometheus text format |
| GET | `/registry` | Active version, versions, canary state per model |
| GET | `/stats/{model}` | Request, error, and latency percentiles per version |
| GET | `/stats/{model}/timeseries` | Recent per-request latencies |
| GET | `/stats/{model}/percentile_history` | Binned p50/p95/p99 and error rate |
| GET | `/drift/{model}` | Latest drift score and breach state |
| GET | `/drift/{model}/history` | Drift score history |
| GET | `/incidents` | Intervention log (`model_id`, `since`, `limit` filters) |

Pass an `X-User-Id` header to `/predict/*` for sticky canary routing.

## Repository layout

```
adaptive-ml-platform/
├── serving/          FastAPI app, batcher, router, metrics store, incident response
│   └── registry/     ModelVersion interface, per-modality adapters, manifest
├── drift/            Reference loading, PSI and embedding metrics, DriftMonitor
├── models/           Trained artifacts and drift references (fraud, satellite, ai_text)
├── notebooks/        Training and evaluation notebooks
├── dashboard/        React + TypeScript frontend
├── scripts/          Traffic generation, version seeding, demos, detector comparison
├── tests/            Unit tests and benchmarks
├── docs/             design.md, demo logs and results, screenshots
└── data/             Local datasets (gitignored)
```

Design rationale and benchmark methodology are in [`docs/design.md`](docs/design.md).

## Engineering notes

Problems found and fixed while building this, each caught by measurement rather than assumption:

- **Train/validation leakage (ai_text).** HC3 has a human and an AI answer per question, and a plain per-answer split put the same question on both sides. Fixed with `GroupShuffleSplit` grouped by question.
- **PSI false alarms.** In-distribution traffic scored PSI above 1.3. Per-feature inspection showed a smooth gradient across all features, ruling out double-scaling and column-misalignment bugs. The cause was empty histogram bins on small windows combined with a fixed epsilon. Fixed with Laplace smoothing and a 150-sample minimum window.
- **Misleading benchmarks.** A mock-based benchmark gave the opposite conclusion from the real-model benchmark, and single-run executor sweeps produced a noisy curve and a wrong optimum for ai_text. Both were fixed by benchmarking real models with averaged runs and a warm-up.
- **Re-alerting on stale data.** The monitor kept re-scoring a frozen buffer and logging thousands of "consecutive" breaches after a rollback. Fixed by requiring fresh samples per window and clearing the evidence once an action is taken.
- **State in memory vs. files on disk.** The router, registry, and drift buffers load at startup, so editing the JSON files while the server runs does nothing until a restart. Several confusing demo results traced back to this.
- **Dev-server restarts.** uvicorn `--reload` watched the manifest and routing files that a rollback writes, killing the server at the moment of the rollback.
- **Environment issues.** Groq's free tier allows 30 requests per minute, so current-generation text is pre-generated into a pool instead of requested per call. The HC3 loading script is no longer supported by recent `datasets` versions.

## Limitations

- **No circuit breaker yet.** If drift persists after a rollback and there is no earlier version, the monitor logs a `breach_no_fallback` incident every few windows instead of switching to a degraded response.
- State is per-process (batchers, executors, in-memory metrics). Running multiple uvicorn workers would need shared state such as Redis.
- The SQLite audit log lags live traffic by up to the flush interval. The in-memory counters behind `/metrics` are exact.
- Rollback picks the previous version by sorted version name, not from a true history stack. It is correct only if versions are promoted in order.
- Drift references are static per version. There is no automated retraining or reference-refresh pipeline.
- Drift detection is based on inputs only. Without ground-truth labels it cannot say whether a drift alarm means accuracy actually dropped.
- The AI-text detector's headline accuracy is dataset-specific.

## Troubleshooting

- **`PermissionError: [WinError 5]` when a rollback writes `manifest.json`.** A sync client such as OneDrive is briefly locking the file. Keep the repo outside synced folders, or pause syncing while running the demos.
- **Server restarts mid-demo.** Don't run uvicorn with `--reload` during demos.
- **Dashboard shows "Couldn't reach the serving API."** The API isn't running or `VITE_API_BASE_URL` is wrong.

## Roadmap

- Circuit breaker with a degraded response when no fallback version exists
- Champion/challenger comparison between versions using delayed ground-truth labels
- Error-rate and latency SLA triggers alongside drift triggers
- Rollback history stack
- Free-tier deployment
- CI running the test suite on every push

## Data and acknowledgements

- Credit Card Fraud Detection dataset (Worldline and the ULB Machine Learning Group, via Kaggle)
- EuroSAT (Helber et al.)
- HC3, the Human ChatGPT Comparison Corpus (Guo et al.), CC-BY-SA 4.0

See each dataset's page for full terms. Add a `LICENSE` file for the code (MIT is a common choice).
