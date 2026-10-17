"""
scripts/diagnose_fraud_model.py

The fraud model's validation PR-AUC is 0.107, far below what LightGBM normally reaches on
the Kaggle credit-card dataset. This compares a few training configurations with 5-fold
stratified cross-validation (far less noisy than one split with under 100 positives) and
reports train vs. validation average precision, so overfitting shows up directly.

Tree models are invariant to feature scaling, so raw features are used here.

Usage (project root, venv active):
    python -m scripts.diagnose_fraud_model
Takes a few minutes.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv("data/raw/creditcard.csv")
X, y = df.drop(columns=["Class"]), df["Class"]
ratio = (y == 0).sum() / (y == 1).sum()
print(f"{len(df)} rows, {int(y.sum())} fraud, negative:positive ratio {ratio:.0f}\n")

CONFIGS = {
    "original (scale_pos_weight=ratio)": dict(n_estimators=300, learning_rate=0.05, scale_pos_weight=ratio),
    "no class weighting": dict(n_estimators=300, learning_rate=0.05),
    "sqrt weighting": dict(n_estimators=300, learning_rate=0.05, scale_pos_weight=ratio ** 0.5),
    "regularized + sqrt weighting": dict(
        n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=ratio ** 0.5,
    ),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, params in CONFIGS.items():
    val_ap, val_auc, train_ap = [], [], []
    for tr, va in skf.split(X, y):
        m = LGBMClassifier(random_state=42, verbose=-1, **params)
        m.fit(X.iloc[tr], y.iloc[tr])
        p_va = m.predict_proba(X.iloc[va])[:, 1]
        p_tr = m.predict_proba(X.iloc[tr])[:, 1]
        val_ap.append(average_precision_score(y.iloc[va], p_va))
        val_auc.append(roc_auc_score(y.iloc[va], p_va))
        train_ap.append(average_precision_score(y.iloc[tr], p_tr))
    print(f"{name:<36} val AP {np.mean(val_ap):.3f} +/- {np.std(val_ap):.3f} | "
          f"val ROC-AUC {np.mean(val_auc):.3f} | train AP {np.mean(train_ap):.3f}")

print("\nRead it like this: train AP near 1.0 with a low val AP means overfitting. "
      "Pick the config with the best validation AP, then retrain the notebook with it.")