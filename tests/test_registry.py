import numpy as np
from PIL import Image
from serving.registry.registry import ModelRegistry

# Initialize Registry
registry = ModelRegistry("serving/registry/manifest.json")

# 1. Sanity Test: Fraud Model (LightGBM)
print("Testing Fraud Model...")
fraud_model = registry.get("fraud")
raw_fraud = {"Time": 1000, "Amount": 50.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
fraud_out, fraud_conf = fraud_model.predict(raw_fraud)
fraud_emb = fraud_model.embed(raw_fraud)
print(f"  Predict: {fraud_out} | Conf: {fraud_conf:.4f}")
print(f"  Embed Shape: {fraud_emb.shape}\n")

# 2. Sanity Test: Satellite Model (MobileNetV3)
print("Testing Satellite Model...")
sat_model = registry.get("satellite")
dummy_img = Image.fromarray(np.uint8(np.random.rand(64, 64, 3) * 255))
sat_out, sat_conf = sat_model.predict(dummy_img)
sat_emb = sat_model.embed(dummy_img)
print(f"  Predict: {sat_out} | Conf: {sat_conf:.4f}")
print(f"  Embed Shape: {sat_emb.shape}\n")

# 3. Sanity Test: AI Text Model (DistilBERT)
print("Testing AI Text Model...")
text_model = registry.get("ai_text")
raw_text = "This is a test prompt to evaluate the model registry abstraction."
text_out, text_conf = text_model.predict(raw_text)
text_emb = text_model.embed(raw_text)
print(f"  Predict: {text_out} | Conf: {text_conf:.4f}")
print(f"  Embed Shape: {text_emb.shape}\n")

print("All 3 modalities passed registry verification!")