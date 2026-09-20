import pytest
from PIL import Image
from serving.registry.registry import ModelRegistry

def test_single_vs_batch_parity():
    registry = ModelRegistry()
    
    # 1. Fraud Parity
    fraud = registry.get("fraud")
    sample_fraud = {"Time": 1000, "Amount": 50.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    out_single, conf_single = fraud.predict(sample_fraud)
    out_batch, conf_batch = fraud.predict_batch([sample_fraud])[0]
    assert out_single == out_batch
    assert pytest.approx(conf_single) == conf_batch

    # 2. Satellite Parity
    sat = registry.get("satellite")
    sample_img = Image.new("RGB", (64, 64), color="red")
    out_single, conf_single = sat.predict(sample_img)
    out_batch, conf_batch = sat.predict_batch([sample_img])[0]
    assert out_single == out_batch
    assert pytest.approx(conf_single) == conf_batch

    # 3. AI Text Parity
    text = registry.get("ai_text")
    sample_text = "This is a test prompt to verify batching parity."
    out_single, conf_single = text.predict(sample_text)
    out_batch, conf_batch = text.predict_batch([sample_text])[0]
    assert out_single == out_batch
    assert pytest.approx(conf_single) == conf_batch

    print("✅ Parity verified across all 3 modalities!")

if __name__ == "__main__":
    test_single_vs_batch_parity()