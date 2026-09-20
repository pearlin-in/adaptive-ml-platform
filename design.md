# Design Document: Adaptive ML Serving & Monitoring Platform

## 1. Overview

This project is a model-agnostic serving platform that treats three structurally unrelated
production ML problems — tabular fraud detection, satellite image classification, and
AI-generated-text detection — as interchangeable payloads behind one FastAPI backend. The
goal is not to advance the state of the art on any individual model; it is to demonstrate the
operational layer that sits *around* models in production: versioning, micro-batching,
canary routing, observability, statistical drift detection, and an automated response to
that drift. The three models were deliberately chosen to be heterogeneous in modality and
failure mode, so that the serving abstraction is proven to be genuinely general rather than
tuned to one input shape.

**Non-goal:** state-of-the-art model accuracy. Each model uses a well-understood, publicly
available dataset and a small, fast-to-train architecture on purpose — model novelty is not
what this project is evaluating. The infrastructure is.

## 2. Models

| Model | Modality | Dataset | Architecture | Notes |
|---|---|---|---|---|
| `fraud` | Tabular | Kaggle Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`) | LightGBM, `scale_pos_weight` for class imbalance | PR-AUC used as the primary metric over accuracy given ~0.17% positive class |
| `satellite` | Image | EuroSAT via Hugging Face (`tanganke/eurosat`) | Fine-tuned MobileNetV3-Small, 10-class land-use | Balanced classes; macro-F1 tracked alongside accuracy |
| `ai_text` | Text | HC3 (`Hello-SimpleAI/HC3`, config `all`) | Fine-tuned DistilBERT-base-uncased, binary (human/AI) | See §6.3 on split-leakage and generalization caveats |

All three were trained free of cost (Colab free tier / local CPU), consistent with the
project's zero-budget constraint.

## 3. System Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              FastAPI app                 │
                    │  /predict/{fraud,satellite,ai_text}       │
                    │  /metrics   /drift/{model}   /incidents   │
                    └───────────────┬───────────────────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
          ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
          │   Router     │   │ MicroBatcher │   │   MetricsStore    │
          │ (A/B/canary, │   │ (per-model   │   │ (sync counters +  │
          │  sticky hash)│   │  batch/window│   │  async SQLite log)│
          └──────┬───────┘   │  config, per-│   └─────────┬─────────┘
                 │           │  item fault  │             │
                 │           │  isolation)  │             ▼
                 │           └──────┬───────┘      ┌──────────────┐
                 │                  │               │ DriftMonitor │
                 └────────┬─────────┘               │ (PSI / cosine│
                          ▼                          │  drift, per- │
                  ┌───────────────┐                  │  model refs) │
                  │ ModelRegistry │◄─────────────────┴──────┬───────┘
                  │ (manifest.json,                          │
                  │  hot-swap load,                          ▼
                  │  atomic writes)                  ┌───────────────────┐
                  └───────────────┘                  │ IncidentResponder  │
                                                       │ (auto-rollback /   │
                                                       │  canary-kill)      │
                                                       └────────────────────┘
```

## 4. Component Design

### 4.1 `ModelVersion` interface & Registry
Every model implements a common abstract interface — `predict`, `predict_batch`, `embed` —
so the serving layer never branches on modality. Each concrete implementation
(`FraudModel`, `SatelliteModel`, `AITextModel`) owns its own preprocessing; the registry
only knows how to load and cache instances by `(model_id, version)`.

The registry's `manifest.json` is the system's single source of truth for which version is
active per model. Writes to it (`set_active`, `register_version`) use a temp-file-plus-
`os.replace` pattern for atomicity — this matters specifically because the manifest is what
`IncidentResponder` mutates during an automated rollback, and a corrupted manifest during an
actual incident would be the worst possible failure mode for a reliability-focused system.

Class-name/label-map lookups fail loudly (`raise ValueError`) rather than silently falling
back to a default — an earlier version silently substituted default EuroSAT class names on a
missing manifest key, which is the opposite of what an observability-first platform should do.

### 4.2 Serving core & micro-batching
`MicroBatcher` collects individual `submit()` calls into a queue and flushes them as one
`predict_batch()` call when either `max_batch_size` is reached or `max_latency_ms` elapses
since the oldest pending item, whichever comes first — implemented with `asyncio.Queue` +
`asyncio.Future`, with the actual model call offloaded to a thread-pool executor so blocking
NumPy/PyTorch/sklearn calls never block the event loop.

Batch config and executor pool size are tuned **per model**, not globally — see §7. A single
bad item in a batch does not fail the whole batch: `_process_batch` falls back to per-item
`predict()` calls on batch failure, isolating the fault to the specific offending input.

### 4.3 A/B / canary routing
`Router` assigns traffic to model versions via consistent hashing (`sha256(model_id:key) %
100`) against a configurable canary percentage, so routing is sticky per user/session rather
than randomized per request — matching how production canary rollouts are actually done.
Routing config, like the manifest, is written atomically.

### 4.4 Observability
`MetricsStore` maintains two parallel views of the same data deliberately: **synchronous
in-memory counters and latency windows** (so `/metrics` never blocks on I/O and is always
exactly current) and an **asynchronously flushed SQLite prediction log** (batched writes off
the event loop via a background drain task, eventually consistent up to
`flush_interval_ms`). `/metrics` is rendered in Prometheus text exposition format.

Latency is measured at the endpoint boundary, including batching queue wait time — this is
deliberate: it's the latency the caller actually experiences, and it's what makes the
batching-tradeoff benchmark in §7 meaningful.

### 4.5 Drift detection
Reference statistics are captured once at training time (Phase 1) and stored per model
version: a validation-set sample with quantile bin edges for the tabular model, and a
penultimate-layer embedding centroid for the two neural models.

- **Tabular (`fraud`)**: Population Stability Index (PSI) computed per feature against
  reference quantile bins; the maximum per-feature PSI is the model's drift score.
- **Neural (`satellite`, `ai_text`)**: cosine distance between the reference embedding
  centroid and the centroid of a recent live-traffic window.

PSI uses Laplace (add-one) smoothing rather than a fixed epsilon floor — see §6.4 for why the
original epsilon-based version produced false alarms on small windows. A minimum window size
of 150 samples is enforced before any drift score is computed at all, to keep per-bin sample
counts large enough for PSI to be statistically meaningful.

An alert (and the automated response in §4.6) fires only after `CONSECUTIVE_BREACHES_TO_ALERT`
(3) consecutive drift windows exceed threshold — a single noisy window does not trigger action.

### 4.6 Automated incident response — the project's core differentiator
`IncidentResponder` subscribes to `DriftMonitor`'s breach callback. Two distinct policies:

- **Drift on a canary version** → the canary is pulled (traffic reverts fully to stable);
  cheap, low-blast-radius response since the canary was never serving all traffic.
- **Drift on the stable/active version** → automated rollback to the previous registered
  version, with every intervention logged to an `incidents` table (reason, metric value,
  which version was restored).

This closes the loop from *detecting* a problem to *acting* on it — the distinction between a
monitoring dashboard and an operable reliability system.

**Known limitation, stated deliberately rather than hidden:** rollback currently selects the
lexically/numerically previous version string (`v1`, `v2`, …), not a true "what was actually
stable before this" history stack. This is correct as long as versions are promoted in order,
but would need an explicit rollback-history stack to handle out-of-order promotions correctly
in a real system. Documented here rather than over-engineered for a portfolio project's scope.

## 5. Phase-by-Phase Build Summary

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffold, design doc | Done |
| 1 | Train & export all 3 models with reference artifacts | Done |
| 2 | Model registry, hot-swap loading | Done |
| 3 | FastAPI + micro-batching core | Done |
| 4 | A/B / canary routing | Done |
| 5 | Structured logging, `/metrics` | Done |
| 6 | Drift detection engine | Done |
| 7 | Automated incident response | Done |
| 8 | Dashboard | Planned next |
| 9 | OOD injection demo | Planned |
| 10 | Free-tier deployment | Planned |
| 11 | Documentation & demo video | Planned |

## 6. Engineering Journal — Bugs Found, Diagnosed, and Fixed

Documenting these explicitly because the debugging process, not any single component, is the
strongest evidence of engineering ability in this project.

**6.1 Missing training observability (Phase 1, satellite).** Initial training loop logged
only training loss, with no per-epoch validation pass — meaning convergence and overfitting
were unobservable in a project whose entire premise is observability. Fixed by adding a
validation pass (loss, accuracy) after every epoch, with the history persisted alongside the
model artifact.

**6.2 Unstratified split risk.** Initial train/val splits used plain `random_split`; fixed to
stratified/grouped splits (`stratify_by_column` for satellite; see §6.3 for the more serious
text-model case) to guarantee representative class distributions.

**6.3 Train/validation leakage (Phase 1, ai_text).** HC3 provides a human and an AI answer per
question. Splitting at the individual-answer level allowed the same question's human and AI
answers to land on opposite sides of the split — meaning the model could see a question's
topic/vocabulary during training and be "evaluated" on a near-duplicate context. Fixed with
`GroupShuffleSplit` grouped by question ID, so no question appears in both splits.

Even after this fix, the model scored 99.47% accuracy / 99.16% F1 / 99.99% ROC-AUC in-dataset
— treated as an expected artifact of HC3's strong stylistic separability (2022-era ChatGPT
answers have consistent, formulaic tells), not evidence of general AI-text-detection ability.
The credible test is deferred to Phase 9: evaluating against text from a current-generation
LLM the model never saw in training, where a measurable confidence/accuracy drop is expected
and will be reported as the honest result, not treated as a failure.

**6.4 PSI false alarms on the drift engine (Phase 6), two independent root causes.**
- First hypothesis (double-scaling the `Amount`/`Time` features) and second hypothesis
  (positional column misalignment from `embed()` returning a bare NumPy array instead of a
  labeled `pd.Series`) were both ruled out by inspecting the per-feature PSI breakdown, which
  showed a smooth gradient across all 30 features rather than an isolated or scrambled spike.
- Root cause: small-sample noise. With `MIN_WINDOW_SIZE=30` and `n_bins=10`, empty histogram
  bins were common, and the fixed-epsilon floor (`1e-4`) turned single empty bins into large
  spurious PSI contributions. Fixed by switching to Laplace (add-one) smoothing, which scales
  correctly with sample size, and raising `MIN_WINDOW_SIZE` to 150. Post-fix sanity check:
  in-distribution PSI = 0.1051 (below the 0.25 alert threshold); synthetic out-of-distribution
  shift correctly triggered a breach at PSI = 4.2467.

**6.5 Import/wiring errors caught by manual testing, not by me.** A `MicroBatcher` import
pointed at the wrong module (`metrics_store` instead of `batcher`) — caught by running the
app rather than by code review, reinforcing that integration testing matters even when unit
logic looks correct in isolation.

**6.6 Auto-rollback verified by direct test, not just code review.** Two versions registered
for one model, a synthetic breach forced past the consecutive-breach threshold, and the
router's active/stable version confirmed to flip with a corresponding row in the `incidents`
table — both tests passing before the feature was considered done.

## 7. Performance Engineering

### 7.1 Batching benefit — measured against the real registered models, not mocks
An initial mock-based benchmark for the fraud model gave a misleading result (batching
*0.95x*, i.e. slightly worse) because the mock's `time.sleep()`-based cost model didn't
reflect either the real model's compute profile or its actual per-call overhead. Re-run
against the real, registered models via `ModelRegistry`:

| Model | Unbatched throughput | Unbatched p99 | Batched throughput | Batched p99 | Speedup |
|---|---|---|---|---|---|
| fraud | 118.69 req/s | 398.95 ms | 1605.51 req/s | 23.42 ms | **13.53x** |
| satellite | 84.69 req/s | 466.60 ms | 202.08 req/s | 269.63 ms | **2.39x** |
| ai_text | 64.15 req/s | 560.04 ms | 99.98 req/s | 339.44 ms | **1.56x** (after fixing the batching window — see §7.3) |

**Interpretation, stated precisely rather than generically:** fraud's large speedup is not
primarily about amortizing model compute (a single LightGBM row prediction is sub-millisecond)
— it's a reduction in thread-pool/GIL contention overhead from collapsing many concurrent
per-item executor submissions into far fewer batched submissions. Satellite and ai_text's more
modest speedups reflect genuine compute amortization from batched tensor operations, partially
offset by unbatchable per-item preprocessing (see §7.2) and CPU-bound matrix multiplication
scaling closer to linearly with batch size than GPU inference would.

### 7.2 Satellite preprocessing profile
Per-item preprocessing was suspected (correctly) to be a bottleneck independent of the model's
forward pass:

| Sub-step | Time (16-image batch) |
|---|---|
| `convert('RGB')` | 0.144 ms |
| `resize(64, 64)` | 0.358 ms (near no-op; EuroSAT images are already 64×64) |
| `to_tensor()` | 5.096 ms (dominant cost) |
| `stack` + vectorized `normalize` | 4.023 ms |

`ToTensor()` being the dominant cost, not `resize`, redirected optimization effort correctly:
normalization was moved out of the per-item loop into one vectorized call on the stacked
tensor.

### 7.3 Thread-pool executor tuning (per model, 5-run averaged, `os.cpu_count()=12`)

An initial single-run sweep showed a non-monotonic, noisy curve for the fraud model — later
confirmed as measurement noise, not signal, after re-running with 5 averaged trials and a
discarded warm-up run per configuration:

| Model | Optimal `max_workers` | Mean throughput | Notes |
|---|---|---|---|
| fraud | 8 | 109.99 req/s (±4.18) | `max_workers=128` showed the highest variance (±14.13) of any config — instability, not just lower throughput |
| satellite | 4 | 105.28 req/s (±16.74) | Statistically indistinguishable from `max_workers=2`; both clearly ahead of 8+ |
| ai_text | 8 | 72.66 req/s (±7.16) | A single-run sweep had incorrectly picked `max_workers=16`; averaging revealed the true optimum |

**A disproven hypothesis, reported because it was tested, not assumed:** PyTorch intra-op
thread-count (`torch.set_num_threads`) was hypothesized as a source of thread oversubscription
under concurrent load. Direct measurement (`torch_threads` swept 1/2/4/10) showed flat
throughput (44–48 req/s) across all settings for both neural models — ruled out as a factor at
this model/scale.

### 7.4 Batch size × window grid search

| Model | Best config (by throughput) | Result | Decision |
|---|---|---|---|
| fraud | `batch=32, window=10ms` | 2022.93 req/s, p99 17.18 ms | Adopted — clear, monotonic winner |
| satellite | `batch=32, window=10ms` | 362.62 req/s, p99 124.81 ms | Adopted |
| ai_text | `batch=32, window=100ms` | 21.57 req/s, p99 1522 ms | **Not adopted as-is** — see below |

For `ai_text`, throughput varied by less than 2x across the entire grid while p99 latency
stayed uniformly high (1.5–3s) at every configuration — a signal that request concurrency
itself, not batch size, was the binding constraint (batch sizes above the number of
concurrently in-flight requests can't be filled regardless of the configured maximum). Rather
than adopting the raw throughput-maximizing config, `batch=8, window=10ms` was selected
deliberately as a latency-conscious choice appropriate to a smaller realistic batch target.

### 7.5 Final locked configuration
```python
EXECUTORS = {
    "fraud": ThreadPoolExecutor(max_workers=8),
    "satellite": ThreadPoolExecutor(max_workers=4),
    "ai_text": ThreadPoolExecutor(max_workers=8),
}

BATCH_CONFIG = {
    "fraud":     {"max_latency_ms": 10, "max_batch_size": 32},
    "satellite": {"max_latency_ms": 10, "max_batch_size": 32},
    "ai_text":   {"max_latency_ms": 10, "max_batch_size": 8},
}
```

## 8. Known Limitations & Tradeoffs

Stated explicitly rather than glossed over, consistent with the project's observability-first
philosophy:

- **Single-process scope.** Batchers, executors, and in-memory metrics are per-process; a
  multi-worker (multiple uvicorn workers) deployment would need shared state (e.g. Redis) for
  batching and metrics to behave correctly across processes. Out of scope for this project.
- **Eventually-consistent audit log.** The SQLite prediction log lags live traffic by up to
  `flush_interval_ms`; the in-memory counters backing `/metrics` are always exact. This
  tradeoff is deliberate (never block the request path on disk I/O) and is stated here rather
  than left implicit.
- **Rollback versioning is order-based, not history-based** (§4.6).
- **Drift references are static per version** — a new model version requires manually
  recomputing its reference sample/embeddings; there is no automated reference-refresh
  pipeline.
- **AI-text detector's benchmark accuracy is dataset-specific**, not a general claim of
  AI-text-detection capability (§6.3).

## 9. Testing Strategy

- **Unit/parity tests**: single-item `predict()` vs. `predict_batch([x])[0]` equivalence for
  all three models, run before trusting the batched path in any benchmark.
- **Registry tests**: load + predict + embed for all three modalities, asserting expected
  output shapes (fraud: 30-dim feature vector; satellite: 576-dim MobileNetV3 pooled features;
  ai_text: 768-dim DistilBERT CLS embedding).
- **Drift sanity tests**: reference data fed back through the monitor should score near zero;
  synthetically shifted data should breach threshold. Both directions verified (§6.4).
- **Incident-response tests**: forced consecutive breaches verified to flip the active version
  in the router and produce a corresponding row in the `incidents` table.
- **Benchmark methodology**: all performance numbers reported here are averaged over multiple
  runs with a discarded warm-up run, after an earlier single-run methodology was shown to
  produce misleading results (§7.3).
