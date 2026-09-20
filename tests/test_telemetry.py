import os
import sqlite3
import time
from fastapi.testclient import TestClient
from serving.app import app, metrics_store

def test_telemetry_and_metrics_logging():
    with TestClient(app) as client:
        # 1. Send sample Fraud prediction
        fraud_payload = {
            "Time": 0.0, "Amount": 100.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        res = client.post("/predict/fraud", json=fraud_payload)
        assert res.status_code == 200, res.text
        
        # 2. Check /metrics endpoint immediately
        metrics_res = client.get("/metrics")
        assert metrics_res.status_code == 200
        assert "prediction_requests_total" in metrics_res.text
        assert 'model="fraud"' in metrics_res.text

        # 3. Allow async drain loop to flush records to SQLite
        time.sleep(0.3)

        # 4. Query DB directly to verify persistence
        conn = sqlite3.connect(metrics_store.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT model_id, version, latency_ms, input_summary FROM predictions")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) > 0, "No records persisted in SQLite database"
        assert rows[0][0] == "fraud"
        print("✅ Telemetry and Metrics verified successfully!")

if __name__ == "__main__":
    test_telemetry_and_metrics_logging()