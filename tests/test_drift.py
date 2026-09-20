import pandas as pd
from drift.monitor import DriftMonitor, THRESHOLDS, MIN_WINDOW_SIZE


def test_drift_sanity():
    # Pass window_size >= MIN_WINDOW_SIZE
    monitor = DriftMonitor(db_path="serving/predictions.db", window_size=200)
    monitor.register_reference("fraud", "v1", "models/fraud/reference_sample_v1.parquet")

    ref_df = pd.read_parquet("models/fraud/reference_sample_v1.parquet")

    # -------------------------------------------------------------
    # 1. In-Distribution Baseline (Stream >= MIN_WINDOW_SIZE samples)
    # -------------------------------------------------------------
    sample_count = max(MIN_WINDOW_SIZE, 150)
    for _, row in ref_df.iloc[:sample_count].iterrows():
        monitor.record_sample("fraud", "v1", row)

    monitor._recompute_all()
    latest = monitor.get_latest("fraud", "v1")

    assert latest is not None, f"No PSI computed! Expected at least {MIN_WINDOW_SIZE} samples."

    print("\n--- IN-DISTRIBUTION PSI BREAKDOWN ---")
    if "detail" in latest:
        for col, score in sorted(latest["detail"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"Top high features -> {col:10s}: {score:.6f}")
    print(f"Overall In-Distribution Score: {latest['score']:.4f}\n")

    assert latest["score"] < THRESHOLDS["fraud"]["threshold"], (
        f"False alarm! In-distribution PSI score: {latest['score']:.4f}"
    )
    assert not latest["breached"]
    print(f"✅ In-distribution PSI verified: {latest['score']:.4f}")

    # -------------------------------------------------------------
    # Clear buffer for second scenario
    # -------------------------------------------------------------
    monitor.buffers[("fraud", "v1")].clear()

    # -------------------------------------------------------------
    # 2. Out-Of-Distribution (OOD) Shifted Run
    # -------------------------------------------------------------
    shifted_df = ref_df.iloc[:sample_count].copy()
    shifted_df["Amount"] = shifted_df["Amount"] + 10.0
    shifted_df["V1"] = shifted_df["V1"] + 10.0

    for _, row in shifted_df.iterrows():
        monitor.record_sample("fraud", "v1", row)

    monitor._recompute_all()
    latest_drifted = monitor.get_latest("fraud", "v1")

    print(f"Overall Shifted Score: {latest_drifted['score']:.4f}")

    assert latest_drifted["breached"], f"Failed to breach! PSI score: {latest_drifted['score']:.4f}"
    assert latest_drifted["score"] > THRESHOLDS["fraud"]["threshold"]
    print(f"✅ Out-of-distribution drift breach verified: PSI score: {latest_drifted['score']:.4f}")


if __name__ == "__main__":
    test_drift_sanity()