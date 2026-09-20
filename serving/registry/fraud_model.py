# serving/registry/fraud_model.py
import joblib
from .base import ModelVersion

class FraudModel(ModelVersion):
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, raw_input: dict):
        import pandas as pd
        x = pd.DataFrame([raw_input])
        x[["Time", "Amount"]] = self.scaler.transform(x[["Time", "Amount"]])
        proba = self.model.predict_proba(x)[0]
        return {"label": int(proba.argmax()), "fraud_prob": float(proba[1])}, float(proba.max())

    def embed(self, raw_input: dict):
        import pandas as pd
        x = pd.DataFrame([raw_input])
        x[["Time", "Amount"]] = self.scaler.transform(x[["Time", "Amount"]])
        return x.values[0]  # raw feature vector — matches your reference_sample stats