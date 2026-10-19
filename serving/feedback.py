"""
serving/feedback.py

Ground-truth labels and per-version comparison for champion/challenger evaluation.

Callers POST a true label for a request_id (usually delayed, after the fact). compare_versions()
joins those labels with the prediction log and reports, per model version: request and error
counts, latency percentiles, accuracy, and for the fraud model precision, recall and average
precision (from the logged fraud_prob).
"""
import json
import sqlite3
import time
from collections import defaultdict
from typing import Optional

import numpy as np

MIN_LABELED_FOR_VERDICT = 30
VERDICT_MARGIN = 0.05

FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    timestamp REAL NOT NULL
);
"""


def init_feedback_table(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(FEEDBACK_SCHEMA)
    conn.commit()
    conn.close()


def record_feedback(db_path: str, request_id: str, label) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO feedback (request_id, label, timestamp) VALUES (?, ?, ?)",
        (request_id, json.dumps(label), time.time()),
    )
    conn.commit()
    conn.close()


def _percentile(sorted_values: list, p: float) -> Optional[float]:
    if not sorted_values:
        return None
    return float(sorted_values[min(int(len(sorted_values) * p), len(sorted_values) - 1)])


def compare_versions(db_path: str, model_id: str, since: Optional[float] = None) -> dict:
    """Per-version metrics for one model, optionally only for predictions after `since`."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    params: list = [model_id]
    time_clause = ""
    if since is not None:
        time_clause = " AND p.timestamp > ?"
        params.append(since)
    rows = conn.execute(
        "SELECT p.version, p.latency_ms, p.error, p.output, f.label "
        "FROM predictions p LEFT JOIN feedback f ON p.request_id = f.request_id "
        f"WHERE p.model_id = ?{time_clause}",
        params,
    ).fetchall()
    conn.close()

    by_version = defaultdict(list)
    for r in rows:
        by_version[r["version"]].append(r)

    result = {}
    for version, items in by_version.items():
        ok = [r for r in items if r["error"] is None]
        latencies = sorted(r["latency_ms"] for r in ok)
        entry = {
            "requests": len(items),
            "errors": len(items) - len(ok),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "labeled": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "average_precision": None,
        }

        y_true, y_pred, scores = [], [], []
        for r in ok:
            if r["label"] is None or r["output"] is None:
                continue
            out = json.loads(r["output"])
            y_true.append(json.loads(r["label"]))
            y_pred.append(out.get("label"))
            scores.append(out.get("fraud_prob"))

        entry["labeled"] = len(y_true)
        if y_true:
            entry["accuracy"] = float(np.mean([p == t for p, t in zip(y_pred, y_true)]))

            if all(s is not None for s in scores):  # binary scored model (fraud)
                y = np.array(y_true)
                pred = np.array(y_pred)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                entry["precision"] = tp / (tp + fp) if (tp + fp) else None
                entry["recall"] = tp / (tp + fn) if (tp + fn) else None
                if len(set(y_true)) == 2:
                    from sklearn.metrics import average_precision_score
                    entry["average_precision"] = float(average_precision_score(y, np.array(scores)))
        result[version] = entry
    return result


def build_verdict(versions: dict, champion: Optional[str], challenger: Optional[str], metric: str) -> dict:
    """A heuristic recommendation. It is not a statistical test and says so."""
    if not challenger:
        return {"recommendation": "no_canary_active"}
    base = {
        "champion": champion,
        "challenger": challenger,
        "metric": metric,
        "min_labeled": MIN_LABELED_FOR_VERDICT,
        "note": "Heuristic on small samples, not a statistical test.",
    }
    champ, chall = versions.get(champion), versions.get(challenger)
    if not champ or not chall:
        return {**base, "recommendation": "insufficient_data"}
    enough = champ["labeled"] >= MIN_LABELED_FOR_VERDICT and chall["labeled"] >= MIN_LABELED_FOR_VERDICT
    if not enough or champ.get(metric) is None or chall.get(metric) is None:
        return {**base, "recommendation": "insufficient_data",
                "labeled": {champion: champ["labeled"], challenger: chall["labeled"]}}
    delta = chall[metric] - champ[metric]
    if delta > VERDICT_MARGIN:
        recommendation = "promote_challenger"
    elif delta < -VERDICT_MARGIN:
        recommendation = "keep_champion"
    else:
        recommendation = "inconclusive"
    return {**base, "recommendation": recommendation, "champion_value": champ[metric],
            "challenger_value": chall[metric], "delta": delta}