# tests/test_incident_response.py
import os
import sqlite3
import pytest
from unittest.mock import MagicMock

from serving.incident_response import IncidentResponder


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_predictions.db")
    yield db_file
    if os.path.exists(db_file):
        os.remove(db_file)


def test_canary_drift_kills_canary_only(temp_db):
    # Mock router & registry
    mock_router = MagicMock()
    mock_router.config = {
        "fraud": {
            "stable": "v1",
            "canary": "v2",
            "canary_percent": 20,
        }
    }
    mock_registry = MagicMock()

    responder = IncidentResponder(router=mock_router, registry=mock_registry, db_path=temp_db)

    fake_score = {
        "metric_name": "psi_max_feature",
        "score": 0.42,
        "consecutive_breaches": 3,
    }

    # Breach occurs on canary version (v2)
    responder.handle_breach("fraud", "v2", fake_score)

    # Verify canary was zeroed out and stable version was untouched
    assert mock_router.config["fraud"]["canary"] is None
    assert mock_router.config["fraud"]["canary_percent"] == 0
    assert mock_router.config["fraud"]["stable"] == "v1"
    mock_router._save.assert_called_once()

    # Verify incident table recorded canary_killed
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM incidents").fetchone()
    conn.close()

    assert row is not None
    assert row["action"] == "canary_killed"
    assert row["version"] == "v2"
    assert row["previous_stable"] == "v1"
    print("\n✅ Canary drift breach test passed: Canary stripped cleanly.")


def test_stable_drift_triggers_auto_rollback(temp_db):
    # Mock router & registry with v2 as stable
    mock_router = MagicMock()
    mock_router.config = {
        "fraud": {
            "stable": "v2",
            "canary": None,
            "canary_percent": 0,
        }
    }
    
    mock_registry = MagicMock()
    mock_registry.manifest = {
        "fraud": {
            "versions": {
                "v1": {"path": "models/fraud/v1.pkl"},
                "v2": {"path": "models/fraud/v2.pkl"},
            }
        }
    }

    responder = IncidentResponder(router=mock_router, registry=mock_registry, db_path=temp_db)

    fake_score = {
        "metric_name": "psi_max_feature",
        "score": 0.38,
        "consecutive_breaches": 3,
    }

    # Breach occurs on active stable version (v2)
    responder.handle_breach("fraud", "v2", fake_score)

    # Verify router & registry rolled back to v1
    mock_router.set_active_stable.assert_called_once_with("fraud", "v1")
    mock_registry.set_active.assert_called_once_with("fraud", "v1")

    # Verify incident recorded auto_rollback
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM incidents").fetchone()
    conn.close()

    assert row is not None
    assert row["action"] == "auto_rollback"
    assert row["version"] == "v2"
    assert row["previous_stable"] == "v1"
    print("✅ Stable drift breach test passed: Rollback triggered to v1.")


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_incident_response.py"])