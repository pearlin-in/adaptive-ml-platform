import time
import logging
import sqlite3

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
    """
    Subscribes to DriftMonitor breach events and takes real action:
    rolls traffic off the drifting version, back to the last known-good one.
    """

    def __init__(self, router, registry, db_path: str):
        self.router = router
        self.registry = registry
        self.db_path = db_path
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
        """
        This is the callback registered with drift_monitor.on_breach(...).
        Called synchronously from DriftMonitor's executor thread — keep it fast,
        no blocking I/O beyond the sqlite write above.
        """
        cfg = self.router.config[model_id]
        current_stable = cfg["stable"]

        if version != current_stable:
            # Drift on a canary — cheapest response: just kill the canary, no rollback needed.
            logger.warning(
                "DRIFT BREACH: %s canary v%s (%s=%.4f) — pulling canary traffic, stable v%s unaffected",
                model_id, version, score["metric_name"], score["score"], current_stable,
            )
            self.router.config[model_id]["canary"] = None
            self.router.config[model_id]["canary_percent"] = 0
            self.router._save()
            self._log_incident(model_id, version, "canary_killed",
                                f"{score['metric_name']} breached for {score['consecutive_breaches']} consecutive windows",
                                score, current_stable)
            return

        # Drift on the stable/active version — this is the real incident.
        previous_version = self._find_previous_version(model_id, version)
        if previous_version is None:
            logger.error(
                "DRIFT BREACH on %s stable v%s with no previous version to roll back to — "
                "serving degraded traffic, manual intervention required",
                model_id, version,
            )
            self._log_incident(model_id, version, "breach_no_fallback",
                                f"{score['metric_name']} breached, no prior version available",
                                score, current_stable)
            return

        logger.error(
            "DRIFT BREACH: %s stable v%s (%s=%.4f) — auto-rolling back to v%s",
            model_id, version, score["metric_name"], score["score"], previous_version,
        )
        self.router.set_active_stable(model_id, previous_version)
        self.registry.set_active(model_id, previous_version)
        self._log_incident(model_id, version, "auto_rollback",
                            f"{score['metric_name']} breached for {score['consecutive_breaches']} consecutive windows",
                            score, previous_version)

    def _find_previous_version(self, model_id: str, current_version: str) -> str | None:
        versions = sorted(self.registry.manifest[model_id]["versions"].keys())
        if current_version not in versions:
            return None
        idx = versions.index(current_version)
        return versions[idx - 1] if idx > 0 else None