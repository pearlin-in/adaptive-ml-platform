"""
serving/incident_response.py

Subscribes to DriftMonitor events and acts on them:

  - drift on a canary                      -> pull the canary
  - drift on the stable version            -> roll back to the previous version
  - drift on the stable version, and there -> open the circuit breaker (one incident per
    is no earlier version to return to        transition, not one per breaching window)
  - healthy windows while half-open        -> close the breaker

Without a breaker (breaker=None) the old behaviour is kept: a "breach_no_fallback" incident.
"""
import logging
import sqlite3
import time
from typing import Optional

from serving.circuit_breaker import BreakerState, CircuitBreaker

logger = logging.getLogger(__name__)

INCIDENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    version TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    previous_stable TEXT,
    timestamp REAL NOT NULL
);
"""


class IncidentResponder:
    def __init__(self, router, registry, db_path: str, breaker: Optional[CircuitBreaker] = None):
        self.router = router
        self.registry = registry
        self.db_path = db_path
        self.breaker = breaker
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript(INCIDENT_SCHEMA)
        conn.commit()
        conn.close()

    def _log_incident(self, model_id, version, action, reason, score: dict, previous_stable):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO incidents (model_id, version, action, reason, metric_name, "
            "metric_value, previous_stable, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (model_id, version, action, reason, score.get("metric_name"),
             score.get("score"), previous_stable, time.time()),
        )
        conn.commit()
        conn.close()

    def handle_breach(self, model_id: str, version: str, score: dict):
        """Registered with drift_monitor.on_breach(...). Runs in the monitor's executor thread."""
        cfg = self.router.config[model_id]
        current_stable = cfg["stable"]
        current_canary = cfg.get("canary")
        windows = f"{score['metric_name']} breached for {score['consecutive_breaches']} consecutive windows"

        if version == current_canary:
            logger.warning("DRIFT BREACH on %s canary %s: pulling canary", model_id, version)
            cfg["canary"] = None
            cfg["canary_percent"] = 0
            self.router._save()
            self._log_incident(model_id, version, "canary_killed", windows, score, current_stable)
            return

        if version != current_stable:
            logger.info("Ignoring drift on %s %s: not stable or canary (already retired)", model_id, version)
            return

        previous_version = self._find_previous_version(model_id, version)
        if previous_version is not None:
            logger.error(
                "DRIFT BREACH: %s stable %s (%s=%.4f): auto-rolling back to %s",
                model_id, version, score["metric_name"], score["score"], previous_version,
            )
            self.router.set_active_stable(model_id, previous_version)
            self.registry.set_active(model_id, previous_version)
            self._log_incident(model_id, version, "auto_rollback", windows, score, previous_version)
            return

        # Nothing to roll back to.
        if self.breaker is None:
            logger.error("DRIFT BREACH on %s stable %s with no previous version to roll back to", model_id, version)
            self._log_incident(model_id, version, "breach_no_fallback",
                               f"{score['metric_name']} breached, no prior version available",
                               score, current_stable)
            return

        previous_state = self.breaker.trip(model_id, version, windows)
        if previous_state is None:
            return  # already open: do not log a duplicate incident
        action = "breaker_reopened" if previous_state == BreakerState.HALF_OPEN else "breaker_opened"
        logger.error("DRIFT BREACH on %s stable %s with no earlier version: %s", model_id, version, action)
        self._log_incident(model_id, version, action,
                           f"{windows}; no earlier version to roll back to, circuit opened",
                           score, current_stable)

    def handle_window(self, model_id: str, version: str, score: dict):
        """Registered with drift_monitor.on_window(...). Closes a half-open breaker after healthy windows."""
        if self.breaker is None:
            return
        if version != self.router.config[model_id]["stable"]:
            return
        closed = self.breaker.record_window(model_id, healthy=not score["breached"])
        if closed:
            logger.warning("Circuit closed for %s after healthy drift windows", model_id)
            self._log_incident(model_id, version, "breaker_closed",
                               f"{score['metric_name']} back under threshold; circuit closed",
                               score, version)

    def _find_previous_version(self, model_id: str, current_version: str) -> Optional[str]:
        versions = sorted(self.registry.manifest[model_id]["versions"].keys())
        if current_version not in versions:
            return None
        idx = versions.index(current_version)
        return versions[idx - 1] if idx > 0 else None