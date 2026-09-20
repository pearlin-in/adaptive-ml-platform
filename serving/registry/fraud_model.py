import joblib
import pandas as pd
from .base import ModelVersion


class FraudModel(ModelVersion):
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, raw_input: dict):
        return self.predict_batch([raw_input])[0]

    def predict_batch(self, raw_inputs: list[dict]) -> list[tuple[dict, float]]:
        df = pd.DataFrame(raw_inputs)
        df[["Time", "Amount"]] = self.scaler.transform(df[["Time", "Amount"]])
        probs = self.model.predict_proba(df)

        results = []
        for proba in probs:
            results.append((
                {"label": int(proba.argmax()), "fraud_prob": float(proba[1])},
                float(proba.max())
            ))
        return results

    def embed(self, raw_input: dict):
        df = pd.DataFrame([raw_input])
        df[["Time", "Amount"]] = self.scaler.transform(df[["Time", "Amount"]])
        return df.iloc[0]