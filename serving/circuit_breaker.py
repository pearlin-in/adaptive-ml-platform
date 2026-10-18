"""
serving/circuit_breaker.py

Per-model circuit breaker.

When drift persists on the stable version and there is no earlier version to roll back to,
the breaker OPENs and the API rejects requests with a degraded 503 instead of serving
suspect predictions. After a cooldown it goes HALF_OPEN: traffic flows again so drift can be
re-measured. Enough consecutive healthy drift windows close it; a new breach re-opens it.

State is held in memory, so a server restart closes every breaker.
"""
import threading
import time
from enum import Enum
from typing import Callable, Optional


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        cooldown_s: float = 120.0,
        healthy_windows_to_close: int = 2,
        clock: Callable[[], float] = time.time,
    ):
        self.cooldown_s = cooldown_s
        self.healthy_windows_to_close = healthy_windows_to_close
        self._clock = clock
        self._lock = threading.Lock()
        self._models: dict[str, dict] = {}

    def _entry(self, model_id: str) -> dict:
        return self._models.setdefault(model_id, {
            "state": BreakerState.CLOSED,
            "opened_at": None,
            "reason": None,
            "version": None,
            "healthy_windows": 0,
        })

    def _refresh(self, entry: dict) -> None:
        """Move OPEN to HALF_OPEN once the cooldown has elapsed. Caller holds the lock."""
        if entry["state"] == BreakerState.OPEN and self._clock() - entry["opened_at"] >= self.cooldown_s:
            entry["state"] = BreakerState.HALF_OPEN
            entry["healthy_windows"] = 0

    def state(self, model_id: str) -> BreakerState:
        with self._lock:
            entry = self._entry(model_id)
            self._refresh(entry)
            return entry["state"]

    def allow(self, model_id: str) -> bool:
        """True when requests should be served (closed, or half-open while probing)."""
        return self.state(model_id) != BreakerState.OPEN

    def trip(self, model_id: str, version: str, reason: str) -> Optional[BreakerState]:
        """
        Open the breaker. Returns the previous state when a transition happened, or None
        if it was already open (so callers can avoid logging duplicate incidents).
        """
        with self._lock:
            entry = self._entry(model_id)
            self._refresh(entry)
            previous = entry["state"]
            if previous == BreakerState.OPEN:
                return None
            entry.update(
                state=BreakerState.OPEN,
                opened_at=self._clock(),
                reason=reason,
                version=version,
                healthy_windows=0,
            )
            return previous

    def record_window(self, model_id: str, healthy: bool) -> bool:
        """Report one drift window result. Returns True if this call closed the breaker."""
        with self._lock:
            entry = self._entry(model_id)
            self._refresh(entry)
            if entry["state"] != BreakerState.HALF_OPEN:
                return False
            if not healthy:
                entry["healthy_windows"] = 0
                return False
            entry["healthy_windows"] += 1
            if entry["healthy_windows"] >= self.healthy_windows_to_close:
                entry.update(
                    state=BreakerState.CLOSED,
                    opened_at=None,
                    reason=None,
                    version=None,
                    healthy_windows=0,
                )
                return True
            return False

    def reset(self, model_id: Optional[str] = None) -> None:
        with self._lock:
            if model_id is None:
                self._models.clear()
            else:
                self._models.pop(model_id, None)

    def status(self, model_id: str) -> dict:
        with self._lock:
            entry = self._entry(model_id)
            self._refresh(entry)
            retry_after = None
            if entry["state"] == BreakerState.OPEN:
                retry_after = max(0.0, self.cooldown_s - (self._clock() - entry["opened_at"]))
            return {
                "state": entry["state"].value,
                "reason": entry["reason"],
                "version": entry["version"],
                "opened_at": entry["opened_at"],
                "retry_after_s": retry_after,
                "healthy_windows": entry["healthy_windows"],
            }