from contextlib import asynccontextmanager
import hashlib
import io
import json
import sqlite3
import time
import uuid
import asyncio

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from fastapi.responses import PlainTextResponse
from PIL import Image
from pydantic import BaseModel

from drift.monitor import DriftMonitor, MIN_WINDOW_SIZE
from serving.batcher import MicroBatcher
from serving.metrics_store import MetricsStore
from serving.registry.registry import ModelRegistry
from serving.routing import Router
from serving.incident_response import IncidentResponder

# Initialize core services
registry = ModelRegistry()
metrics_store = MetricsStore()
router = Router()
drift_monitor = DriftMonitor(db_path="serving/predictions.db")
incident_responder = IncidentResponder(router=router, registry=registry, db_path="serving/predictions.db")
drift_monitor.on_breach(incident_responder.handle_breach)
batchers: dict[tuple[str, str], MicroBatcher] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background telemetry store
    await metrics_store.start()

    # Load manifest and register drift reference datasets
    with open("models/manifest.json", "r") as f:
        manifest = json.load(f)

    for model_id in ("fraud", "satellite", "ai_text"):
        if model_id in manifest:
            active_v = manifest[model_id]["active_version"]
            ref_file = manifest[model_id]["versions"][active_v]["reference_file"]
            drift_monitor.register_reference(model_id, active_v, ref_file)

    await drift_monitor.start()

    yield  # Application runs here

    # Shutdown: Stop monitors and batch workers
    await drift_monitor.stop()
    await metrics_store.stop()
    for batcher in batchers.values():
        await batcher.stop()


app = FastAPI(title="Adaptive ML Serving Engine", lifespan=lifespan)


# --- Input Models ---
class FraudInput(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float


class TextInput(BaseModel):
    text: str

#Per-model batcher configuration tailored to model compute profile
BATCH_CONFIG = {
    "fraud":     {"max_latency_ms": 10.0,  "max_batch_size": 32},
    "satellite": {"max_latency_ms": 20.0,  "max_batch_size": 16},
    "ai_text":   {"max_latency_ms": 150.0, "max_batch_size": 8},
}

# --- Helpers ---
async def get_batcher(model_id: str, version: str) -> MicroBatcher:
    key = (model_id, version)
    if key not in batchers:
        cfg = BATCH_CONFIG[model_id]
        b = MicroBatcher(registry.get(model_id, version), **cfg)
        await b.start()
        batchers[key] = b
    return batchers[key]


def summarize_bytes(data: bytes) -> dict:
    return {
        "hash": hashlib.sha256(data).hexdigest()[:16],
        "size_bytes": len(data),
    }


# --- Endpoints ---
@app.post("/predict/fraud")
async def predict_fraud(
    payload: FraudInput, x_user_id: str | None = Header(default=None)
):
    request_id = str(uuid.uuid4())
    version = router.route("fraud", x_user_id or request_id)
    model = registry.get("fraud", version)
    start = time.perf_counter()
    result, confidence, error = None, None, None

    payload_dict = (
        payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    )
    raw_bytes = str(payload_dict).encode("utf-8")

    try:
        batcher = await get_batcher("fraud", version)
        result, confidence = await batcher.submit(payload_dict)

        # Offload embedding/feature vector extraction for drift tracking
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, model.embed, payload_dict)
        drift_monitor.record_sample("fraud", version, vector)

        return {
            "request_id": request_id,
            "version_served": version,
            "result": result,
            "confidence": confidence,
        }
    except Exception as e:
        error = str(e)
        raise e
    finally:
        metrics_store.log_prediction(
            request_id=request_id,
            model_id="fraud",
            version=version,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            input_summary=summarize_bytes(raw_bytes),
            output=result,
            confidence=confidence,
            error=error,
        )


@app.post("/predict/satellite")
async def predict_satellite(
    file: UploadFile = File(...), x_user_id: str | None = Header(default=None)
):
    request_id = str(uuid.uuid4())
    version = router.route("satellite", x_user_id or request_id)
    model = registry.get("satellite", version)
    start = time.perf_counter()
    result, confidence, error = None, None, None
    image_bytes = await file.read()

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        batcher = await get_batcher("satellite", version)
        result, confidence = await batcher.submit(pil_img)

        # Extract image embedding asynchronously for drift monitoring
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, model.embed, pil_img)
        drift_monitor.record_sample("satellite", version, vector)

        return {
            "request_id": request_id,
            "version_served": version,
            "result": result,
            "confidence": confidence,
        }
    except Exception as e:
        error = str(e)
        raise HTTPException(
            status_code=422, detail=f"Invalid image file payload: {e}"
        )from e
    finally:
        metrics_store.log_prediction(
            request_id=request_id,
            model_id="satellite",
            version=version,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            input_summary=summarize_bytes(image_bytes),
            output=result,
            confidence=confidence,
            error=error,
        )


@app.post("/predict/ai_text")
async def predict_text(
    payload: TextInput, x_user_id: str | None = Header(default=None)
):
    request_id = str(uuid.uuid4())
    version = router.route("ai_text", x_user_id or request_id)
    model = registry.get("ai_text", version)
    start = time.perf_counter()
    result, confidence, error = None, None, None

    raw_bytes = payload.text.encode("utf-8")
    try:
        batcher = await get_batcher("ai_text", version)
        result, confidence = await batcher.submit(payload.text)

        # Extract text embedding asynchronously for drift monitoring
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, model.embed, payload.text)
        drift_monitor.record_sample("ai_text", version, vector)

        return {
            "request_id": request_id,
            "version_served": version,
            "result": result,
            "confidence": confidence,
        }
    except Exception as e:
        error = str(e)
        raise e
    finally:
        metrics_store.log_prediction(
            request_id=request_id,
            model_id="ai_text",
            version=version,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            input_summary=summarize_bytes(raw_bytes),
            output=result,
            confidence=confidence,
            error=error,
        )


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(metrics_store.render_prometheus())


@app.get("/drift/{model_id}")
async def get_drift(model_id: str, version: str = "v1"):
    latest = drift_monitor.get_latest(model_id, version)
    if latest is None:
        return {
            "status": "insufficient_data",
            "min_window_size": MIN_WINDOW_SIZE,
            "current_samples": len(drift_monitor.buffers[(model_id, version)]),
        }
    return latest
@app.get("/drift/{model_id}/history")
async def get_drift_history(model_id: str, version: str = "v1", limit: int = 100):
    conn = sqlite3.connect("serving/predictions.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, metric_value as score, breached_threshold as breached "
        "FROM drift_scores WHERE model_id = ? AND version = ? "
        "ORDER BY timestamp DESC LIMIT ?",
        (model_id, version, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]  # chronological for charting

@app.get("/stats/{model_id}/timeseries")
async def get_latency_timeseries(model_id: str, version: str = "v1", limit: int = 100):
    conn = sqlite3.connect("serving/predictions.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, latency_ms FROM predictions "
        "WHERE model_id = ? AND version = ? AND error IS NULL "
        "ORDER BY timestamp DESC LIMIT ?",
        (model_id, version, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]

@app.get("/incidents")
async def get_incidents(model_id: str | None = None, limit: int = 50):
    conn = sqlite3.connect("serving/predictions.db")
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM incidents"
    params = ()
    if model_id:
        query += " WHERE model_id = ?"
        params = (model_id,)
    query += " ORDER BY timestamp DESC LIMIT ?"
    rows = conn.execute(query, params + (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ============================================================================
# DASHBOARD TELEMETRY & REGISTRY ENDPOINTS (JSON)
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Vite frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
    
@app.get("/registry")
async def get_registry():
    """
    Dashboard-facing view of active model versions, available versions, 
    and canary routing state.
    """

    result = {}
    for model_id, manifest_entry in registry.manifest.items():
        routing_entry = router.config.get(model_id, {})
        result[model_id] = {
            "active_version": manifest_entry["active_version"],
            "available_versions": list(manifest_entry["versions"].keys()),
            "canary": routing_entry.get("canary"),
            "canary_percent": routing_entry.get("canary_percent", 0),
        }
    return result


@app.get("/stats/{model_id}")
async def get_stats(model_id: str, version: Optional[str] = Query(default=None)):
    """
    Returns real-time request counts, error counts, and latency percentiles 
    (P50, P95, P99) directly from MetricsStore for React dashboard polling.
    """
    if model_id not in registry.manifest:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")

    versions = [version] if version else list(registry.manifest[model_id]["versions"].keys())
    out = {}
    
    for v in versions:
        key = (model_id, v)
        out[v] = {
            "requests_total": metrics_store.request_counts.get(key, 0),
            "errors_total": metrics_store.error_counts.get(key, 0),
            **metrics_store.percentiles(model_id, v),
        }
    return out

@app.get("/stats/{model_id}/percentile_history")
async def get_percentile_history(model_id: str, version: str = "v1", n_buckets: int = 20, limit: int = 1000):
    conn = sqlite3.connect("serving/predictions.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT timestamp, latency_ms, error FROM predictions "
        "WHERE model_id = ? AND version = ? ORDER BY timestamp DESC LIMIT ?",
        (model_id, version, limit),
    ).fetchall()
    conn.close()

    rows = list(reversed(rows))
    if not rows:
        return []

    bucket_size = max(1, len(rows) // n_buckets)
    buckets = []
    for i in range(0, len(rows), bucket_size):
        chunk = rows[i : i + bucket_size]
        latencies = sorted(r["latency_ms"] for r in chunk if r["error"] is None)
        errors = sum(1 for r in chunk if r["error"] is not None)
        if not latencies:
            continue
        def pct(p):
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return latencies[idx]
        buckets.append({
            "timestamp": chunk[-1]["timestamp"],
            "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "error_rate": errors / len(chunk),
            "count": len(chunk),
        })
    return buckets