"""
scripts/retrain_fraud_model.py

Retrains the fraud model with the configuration that won cross-validation in
scripts/diagnose_fraud_model.py (regularized LightGBM, square-root class weighting).

It reuses the same stratified 80/20 split (random_state=42), the saved scaler, and the same
artifact names and formats as the training notebook, so the registry, drift references, and
serving code need no changes. The previous model and metrics are backed up first.

Usage (project root, venv active, API stopped):
    python -m scripts.retrain_fraud_model
"""
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/fraud/model_v1.joblib"
METRICS_PATH = "models/fraud/metrics_v1.json"
SCALER_PATH = "models/fraud/scaler_v1.joblib"

df = pd.read_csv("data/raw/creditcard.csv")
X, y = df.drop(columns=["Class"]), df["Class"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# The saved scaler must match what a fresh fit on this split produces; otherwise the split
# differs from the notebook's and the existing drift reference sample would no longer line up.
scaler = joblib.load(SCALER_PATH)
fresh = StandardScaler().fit(X_train[["Time", "Amount"]])
if not (np.allclose(fresh.mean_, scaler.mean_) and np.allclose(fresh.scale_, scaler.scale_)):
    raise SystemExit("Saved scaler does not match this split. Stopping so the reference sample stays valid.")


def scale(X_: pd.DataFrame) -> pd.DataFrame:
    X_ = X_.copy()
    X_[["Time", "Amount"]] = scaler.transform(X_[["Time", "Amount"]])
    return X_


ratio = float((y_train == 0).sum() / (y_train == 1).sum())
params = dict(
    n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=ratio ** 0.5,
)

# Back up the current artifacts and remember the old numbers
shutil.copy2(MODEL_PATH, "models/fraud/model_v1.old.joblib")
shutil.copy2(METRICS_PATH, "models/fraud/metrics_v1.old.json")
with open(METRICS_PATH) as f:
    old = json.load(f)

model = LGBMClassifier(random_state=42, verbose=-1, **params)
model.fit(scale(X_train), y_train)
probs = model.predict_proba(scale(X_val))[:, 1]
pr_auc = float(average_precision_score(y_val, probs))
roc_auc = float(roc_auc_score(y_val, probs))

joblib.dump(model, MODEL_PATH)
with open(METRICS_PATH, "w") as f:
    json.dump({
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "model": "LGBMClassifier",
        "params": {k: float(v) for k, v in params.items()},
        "note": (f"Single stratified 80/20 split (random_state=42), {int(y_val.sum())} validation positives. "
                 "Average precision varied by about +/-0.15 across 5 CV folds, so treat this as an estimate."),
    }, f, indent=2)

print(f"Old model:  PR-AUC {old['pr_auc']:.4f}  ROC-AUC {old['roc_auc']:.4f}")
print(f"New model:  PR-AUC {pr_auc:.4f}  ROC-AUC {roc_auc:.4f}")
print(f"Saved {MODEL_PATH} and {METRICS_PATH}. Backups: models/fraud/*.old.*")
print("Next: copy model_v1.joblib and metrics_v1.json to the v2 names (see the commands in chat).")