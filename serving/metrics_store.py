import asyncio
from collections import defaultdict, deque
import hashlib
import json
import statistics
import sqlite3
import time

DB_PATH = "serving/predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    version TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    input_summary TEXT,
    output TEXT,
    confidence REAL,
    error TEXT,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_model_version_ts ON predictions(model_id, version, timestamp);
"""


class MetricsStore:
    def __init__(self, db_path=DB_PATH, flush_interval_ms=200, flush_batch_size=50):
        self.db_path = db_path
        self.flush_interval_ms = flush_interval_ms
        self.flush_batch_size = flush_batch_size
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task = None

        # In-memory metrics updated synchronously — zero DB blocking on /metrics calls
        self.request_counts = defaultdict(int)                       # (model, version) -> count
        self.error_counts = defaultdict(int)                          # (model, version) -> count
        self.latency_windows = defaultdict(lambda: deque(maxlen=500))  # (model, version) -> rolling latencies

        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    async def start(self):
        self._task = asyncio.create_task(self._drain_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def log_prediction(self, *, request_id, model_id, version, latency_ms,
                        input_summary, output, confidence, error=None):
        key = (model_id, version)
        self.request_counts[key] += 1
        if error:
            self.error_counts[key] += 1
        else:
            self.latency_windows[key].append(latency_ms)

        self.queue.put_nowait({
            "request_id": request_id,
            "model_id": model_id,
            "version": version,
            "latency_ms": latency_ms,
            "input_summary": json.dumps(input_summary),
            "output": json.dumps(output) if output is not None else None,
            "confidence": confidence,
            "error": error,
            "timestamp": time.time(),
        })

    async def _drain_loop(self):
        loop = asyncio.get_event_loop()
        buffer = []
        while True:
            try:
                record = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=self.flush_interval_ms / 1000.0
                )
                buffer.append(record)
                if len(buffer) >= self.flush_batch_size:
                    await loop.run_in_executor(None, self._flush, buffer)
                    buffer = []
            except asyncio.TimeoutError:
                if buffer:
                    await loop.run_in_executor(None, self._flush, buffer)
                    buffer = []

    def _flush(self, records):
        # Runs inside thread executor pool — prevents blocking the FastAPI async event loop
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            """INSERT INTO predictions
               (request_id, model_id, version, latency_ms, input_summary, output, confidence, error, timestamp)
               VALUES (:request_id, :model_id, :version, :latency_ms, :input_summary, :output, :confidence, :error, :timestamp)""",
            records,
        )
        conn.commit()
        conn.close()

    def percentiles(self, model_id, version):
        data = sorted(self.latency_windows[(model_id, version)])
        if not data:
            return {"p50": None, "p95": None, "p99": None}
        def pct(p):
            return data[min(int(len(data) * p), len(data) - 1)]
        return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}

    def render_prometheus(self):
        keys = list(self.request_counts.keys())
        pct_cache = {k: self.percentiles(*k) for k in keys}
        
        lines = [
            "# HELP prediction_requests_total Total prediction requests",
            "# TYPE prediction_requests_total counter"
        ]
        for (m, v), c in self.request_counts.items():
            lines.append(f'prediction_requests_total{{model="{m}",version="{v}"}} {c}')

        lines.extend([
            "# HELP prediction_errors_total Total prediction errors",
            "# TYPE prediction_errors_total counter"
        ])
        for (m, v), c in self.error_counts.items():
            lines.append(f'prediction_errors_total{{model="{m}",version="{v}"}} {c}')

        for name in ("p50", "p95", "p99"):
            lines.append(f"# HELP prediction_latency_ms_{name} {name} latency in ms (rolling window)")
            lines.append(f"# TYPE prediction_latency_ms_{name} gauge")
            for (m, v) in keys:
                val = pct_cache[(m, v)][name]
                if val is not None:
                    lines.append(f'prediction_latency_ms_{name}{{model="{m}",version="{v}"}} {val:.2f}')

        return "\n".join(lines) + "\n"