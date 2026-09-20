import asyncio
import sqlite3
import time
import logging
from collections import defaultdict, deque
import pandas as pd
import numpy as np

from drift.reference import ReferenceData
from drift.metrics import compute_psi, compute_embedding_drift

logger = logging.getLogger(__name__)

DRIFT_SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    version TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    breached_threshold INTEGER NOT NULL,
    window_size INTEGER NOT NULL,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drift_model_version_ts ON drift_scores(model_id, version, timestamp);
"""

# PSI > 0.25 is the conventional "significant shift" cutoff. Cosine-distance
# thresholds are a starting guess — tune them once Phase 9's OOD demo gives
# you real numbers to calibrate against.
THRESHOLDS = {
    "fraud": {"metric": "psi_max_feature", "threshold": 0.25},
    "satellite": {"metric": "cosine_distance", "threshold": 0.15},
    "ai_text": {"metric": "cosine_distance", "threshold": 0.15},
}

MIN_WINDOW_SIZE = 150                
CONSECUTIVE_BREACHES_TO_ALERT = 3     # Phase 7's rollback subscribes to this


class DriftMonitor:
    def __init__(self, db_path: str, recompute_interval_s: float = 15.0, window_size: int = 200):
        self.db_path = db_path
        self.recompute_interval_s = recompute_interval_s
        self.window_size = window_size

        self.references: dict[tuple[str, str], ReferenceData] = {}
        self.buffers: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.consecutive_breaches: dict[tuple[str, str], int] = defaultdict(int)
        self.latest_scores: dict[tuple[str, str], dict] = {}
        self._on_breach_callbacks = []
        self._task: asyncio.Task | None = None
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(DRIFT_SCHEMA)
        conn.commit()
        conn.close()

    def register_reference(self, model_id: str, version: str, reference_file: str):
        self.references[(model_id, version)] = ReferenceData(model_id, reference_file)

    def on_breach(self, callback):
        """Phase 7 calls this to register its rollback handler:
        callback(model_id, version, score_dict)."""
        self._on_breach_callbacks.append(callback)

    def record_sample(self, model_id: str, version: str, vector: np.ndarray):
        """Call after every prediction with model.embed(raw_input)."""
        self.buffers[(model_id, version)].append(vector)

    async def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while True:
            await asyncio.sleep(self.recompute_interval_s)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._recompute_all)

    def _recompute_all(self):
        for key, buffer in list(self.buffers.items()):
            if len(buffer) < MIN_WINDOW_SIZE:
                continue
            model_id, version = key
            ref = self.references.get(key)
            if ref is None:
                logger.warning("No reference registered for %s — skipping", key)
                continue
            try:
                self._compute_and_store(model_id, version, ref, list(buffer))
            except Exception:
                logger.exception("Drift computation failed for %s", key)

    def _compute_and_store(self, model_id, version, ref: ReferenceData, samples: list[np.ndarray]):
        current = np.stack(samples)
        threshold_cfg = THRESHOLDS[model_id]

        if ref.kind == "tabular":
            current_df = pd.DataFrame(samples)
            current_df = current_df.reindex(columns=ref.reference_df.columns)
            psi_per_feature = {
                col: compute_psi(
                    ref.reference_df[col].values,
                    current_df[col].values,
                    ref.bin_edges[col],
                )
                for col in ref.reference_df.columns
            }
            score, metric_name, extra = (
                max(psi_per_feature.values()),
                "psi_max_feature",
                psi_per_feature,
            )
        else:
            result = compute_embedding_drift(ref.centroid_unit, current)
            score, metric_name, extra = result["cosine_distance"], "cosine_distance", result

        breached = score > threshold_cfg["threshold"]
        key = (model_id, version)
        self.consecutive_breaches[key] = self.consecutive_breaches[key] + 1 if breached else 0

        self.latest_scores[key] = {
            "metric_name": metric_name, "score": score, "breached": breached,
            "consecutive_breaches": self.consecutive_breaches[key],
            "detail": extra, "timestamp": time.time(),
        }

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO drift_scores (model_id, version, metric_name, metric_value, "
            "breached_threshold, window_size, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model_id, version, metric_name, score, int(breached), len(samples), time.time()),
        )
        conn.commit()
        conn.close()

        if self.consecutive_breaches[key] >= CONSECUTIVE_BREACHES_TO_ALERT:
            for cb in self._on_breach_callbacks:
                cb(model_id, version, self.latest_scores[key])

    def get_latest(self, model_id: str, version: str) -> dict | None:
        return self.latest_scores.get((model_id, version))