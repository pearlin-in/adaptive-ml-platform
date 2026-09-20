from contextlib import asynccontextmanager
import hashlib
import io
import time
import uuid

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image
from pydantic import BaseModel

# Core platform modules
from serving.batcher import MicroBatcher
from serving.metrics_store import MetricsStore
from serving.registry.registry import ModelRegistry
from serving.routing import Router

# State instances
registry = ModelRegistry()
metrics_store = MetricsStore()
router = Router()

# Active batchers cache: (model_id, version) -> MicroBatcher
batchers: dict[tuple[str, str], MicroBatcher] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start async background telemetry queue
    await metrics_store.start()

    yield  # Application processes requests here

    # Shutdown: Stop metrics drainer and all active worker batchers cleanly
    await metrics_store.stop()
    for batcher in batchers.values():
        await batcher.stop()


app = FastAPI(title="Adaptive ML Serving Engine", lifespan=lifespan)


# --- Input Payloads ---
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


# --- Helper Functions ---
async def get_batcher(model_id: str, version: str) -> MicroBatcher:
    """Retrieves or dynamically instantiates and starts a MicroBatcher worker."""
    key = (model_id, version)
    if key not in batchers:
        model = registry.get(model_id, version)
        b = MicroBatcher(model, max_latency_ms=30, max_batch_size=16)
        await b.start()
        batchers[key] = b
    return batchers[key]


def summarize_bytes(data: bytes) -> dict:
    """Generates an anonymized hash + payload size summary for non-PII logging."""
    return {
        "hash": hashlib.sha256(data).hexdigest()[:16],
        "size_bytes": len(data)
    }


# --- Endpoint Handlers ---
@app.post("/predict/fraud")
async def predict_fraud(payload: FraudInput, x_user_id: str | None = Header(default=None)):
    request_id = str(uuid.uuid4())
    version = router.route("fraud", x_user_id or request_id)
    start = time.perf_counter()
    result, confidence, error = None, None, None

    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    raw_bytes = str(payload_dict).encode("utf-8")

    try:
        batcher = await get_batcher("fraud", version)
        result, confidence = await batcher.submit(payload_dict)
        return {
            "request_id": request_id,
            "version_served": version,
            "result": result,
            "confidence": confidence
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
            error=error
        )


@app.post("/predict/satellite")
async def predict_satellite(file: UploadFile = File(...), x_user_id: str | None = Header(default=None)):
    request_id = str(uuid.uuid4())
    version = router.route("satellite", x_user_id or request_id)
    start = time.perf_counter()
    result, confidence, error = None, None, None

    image_bytes = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        batcher = await get_batcher("satellite", version)
        result, confidence = await batcher.submit(pil_img)
        return {
            "request_id": request_id,
            "version_served": version,
            "result": result,
            "confidence": confidence
        }
    except Exception as e:
        error = str(e)
        raise e
    finally:
        metrics_store.log_prediction(
            request_id=request_id,
            model_id="satellite",
            version=version,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            input_summary=summarize_bytes(image_bytes),
            output=result,
            confidence=confidence,
            error=error
        )


@app.post("/predict/ai_text")
async def predict_text(payload: TextInput, x_user_id: str | None = Header(default=None)):
    request_id = str(uuid.uuid4())
    version = router.route("ai_text", x_user_id or request_id)
    start = time.perf_counter()
    result, confidence, error = None, None, None

    raw_bytes = payload.text.encode("utf-8")
    try:
        batcher = await get_batcher("ai_text", version)
        result, confidence = await batcher.submit(payload.text)
        return {
            "request_id": request_id,
            "version_served": version,
            "result": result,
            "confidence": confidence
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
            error=error
        )


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(metrics_store.render_prometheus())