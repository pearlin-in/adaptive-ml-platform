"""
scripts/compare_drift_detectors.py

Evaluates drift detectors on the fraud model under controlled shifts and links each
detector's output to the model's real performance loss (PR-AUC on the shifted data).

Detectors compared (all see the same 200-row windows, in model space):
  - psi:           this project's max-per-feature PSI at the production threshold (0.25)
  - ks:            scipy two-sample KS per feature, Bonferroni-corrected at alpha=0.05
  - evid_any:      Evidently DataDriftPreset, "any column drifted"
  - evid_dataset:  Evidently DataDriftPreset, its own dataset-level decision
                   (default: drift only if >= 50% of columns drift)

Shift scenarios:
  - none:                     no shift (measures false-positive rate)
  - top_features:             shift the 3 most important V features by `severity` std devs
  - low_importance_features:  shift the 8 least important V features by `severity` std devs
  - amount_scale:             multiply raw Amount by (1 + severity)

The question this answers: does a drift alarm actually track performance loss? Shifting
features the model barely uses should trigger drift detectors without hurting PR-AUC.

Usage (project root, venv active):
    pip install evidently scipy
    python -m scripts.compare_drift_detectors

The Evidently code uses the classic API (evidently.report.Report). If the import fails on
a newer release, run: pip install "evidently<0.5"   (the script skips Evidently otherwise).
"""
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from drift.metrics import compute_psi
from drift.reference import ReferenceData

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)
WINDOW = 200
N_TRIALS = 30              # windows per condition for PSI and KS
N_TRIALS_EVIDENTLY = 8     # Evidently is slower, so fewer windows
PSI_THRESHOLD = 0.25
ALPHA = 0.05
SEVERITIES = [0.5, 1, 2, 4]

# ---------------------------------------------------------------- data and model
df = pd.read_csv("data/raw/creditcard.csv")
X, y = df.drop(columns=["Class"]), df["Class"]
# Same split as the training notebook (stratified, random_state=42)
_, X_val_raw, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = joblib.load("models/fraud/model_v1.joblib")
scaler = joblib.load("models/fraud/scaler_v1.joblib")
ref = ReferenceData("fraud", "models/fraud/reference_sample_v1.parquet")
cols = list(ref.reference_df.columns)
X_val_raw = X_val_raw[cols]

# Keep drift windows disjoint from the reference sample (it was drawn from this split)
in_ref = X_val_raw.index.isin(ref.reference_df.index)
print(f"{int(in_ref.sum())} validation rows overlap the reference sample; excluded from drift windows")
window_mask = ~in_ref


def to_model_space(X_raw: pd.DataFrame) -> pd.DataFrame:
    X_ = X_raw.copy()
    X_[["Time", "Amount"]] = scaler.transform(X_[["Time", "Amount"]])
    return X_[cols]


# ---------------------------------------------------------------- shift scenarios
try:
    names = list(model.feature_name_)
except AttributeError:
    names = cols
importances = pd.Series(model.feature_importances_, index=names)
v_cols = [c for c in cols if c.startswith("V")]
top3 = list(importances[v_cols].nlargest(3).index)
bottom8 = list(importances[v_cols].nsmallest(8).index)
std = X_val_raw.std()
print(f"top-3 important features: {top3}")
print(f"8 least important features: {bottom8}")


def apply_shift(X_raw: pd.DataFrame, kind: str, severity: float) -> pd.DataFrame:
    X_ = X_raw.copy()
    if kind == "top_features":
        for c in top3:
            X_[c] = X_[c] + severity * std[c]
    elif kind == "low_importance_features":
        for c in bottom8:
            X_[c] = X_[c] + severity * std[c]
    elif kind == "amount_scale":
        X_["Amount"] = X_["Amount"] * (1 + severity)
    return X_


# ---------------------------------------------------------------- detectors
def psi_detect(w: pd.DataFrame):
    score = max(
        compute_psi(ref.reference_df[c].values, w[c].values, ref.bin_edges[c]) for c in cols
    )
    return score, score > PSI_THRESHOLD


def ks_detect(w: pd.DataFrame):
    p = min(ks_2samp(ref.reference_df[c].values, w[c].values).pvalue for c in cols)
    return p, p < ALPHA / len(cols)


HAVE_EVIDENTLY = True
try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except Exception as e:  # ImportError or API mismatch
    HAVE_EVIDENTLY = False
    print(f"Evidently unavailable ({e}); skipping it. Try: pip install \"evidently<0.5\"")

_stattests_seen = set()


def evidently_detect(w: pd.DataFrame):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref.reference_df.reset_index(drop=True),
               current_data=w.reset_index(drop=True))
    ds_drift, any_col = None, None
    for m in report.as_dict()["metrics"]:
        r = m.get("result", {})
        if "dataset_drift" in r:
            ds_drift = bool(r["dataset_drift"])
        if "drift_by_columns" in r:
            cols_info = r["drift_by_columns"].values()
            any_col = any(c.get("drift_detected", False) for c in cols_info)
            _stattests_seen.update(c.get("stattest_name") for c in cols_info)
    return ds_drift, any_col


# ---------------------------------------------------------------- experiment
conditions = [("none", 0)] + [(k, s) for k in ("top_features", "low_importance_features", "amount_scale")
                              for s in SEVERITIES]
rows = []
baseline_pr_auc = None

for kind, sev in conditions:
    shifted = to_model_space(apply_shift(X_val_raw, kind, sev))
    pr_auc = average_precision_score(y_val, model.predict_proba(shifted)[:, 1])
    if baseline_pr_auc is None:
        baseline_pr_auc = pr_auc

    pool = shifted[window_mask]
    psi_hits, psi_scores, ks_hits, ev_any, ev_ds = [], [], [], [], []
    for t in range(N_TRIALS):
        w = pool.iloc[RNG.choice(len(pool), WINDOW, replace=False)]
        score, hit = psi_detect(w)
        psi_scores.append(score)
        psi_hits.append(hit)
        ks_hits.append(ks_detect(w)[1])
        if HAVE_EVIDENTLY and t < N_TRIALS_EVIDENTLY:
            try:
                ds, anyc = evidently_detect(w)
                ev_ds.append(ds)
                ev_any.append(anyc)
            except Exception as e:
                print(f"Evidently run failed ({e}); disabling it.")
                HAVE_EVIDENTLY = False

    rows.append({
        "condition": f"{kind}@{sev}",
        "pr_auc": pr_auc,
        "pr_auc_drop": baseline_pr_auc - pr_auc,
        "psi_mean": float(np.mean(psi_scores)),
        "psi_rate": float(np.mean(psi_hits)),
        "ks_rate": float(np.mean(ks_hits)),
        "evid_any_rate": float(np.mean(ev_any)) if ev_any else np.nan,
        "evid_dataset_rate": float(np.mean([bool(v) for v in ev_ds])) if ev_ds else np.nan,
    })
    print(f"done: {kind}@{sev}")

out = pd.DataFrame(rows)
out.to_csv("docs/drift_detector_comparison.csv", index=False)

print("\n=== Detection rate per condition (fraction of windows flagged) ===")
print("The 'none@0' row is the false-positive rate.\n")
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

if _stattests_seen:
    print(f"\nEvidently per-column tests it chose automatically: {sorted(map(str, _stattests_seen))}")

rho, p = spearmanr(out["psi_mean"], out["pr_auc_drop"])
print(f"\nSpearman correlation between mean PSI score and PR-AUC drop across conditions: "
      f"rho={rho:.2f} (p={p:.3f})")
print(f"Baseline PR-AUC (no shift): {baseline_pr_auc:.4f}")
print("Caveats: the validation split has under 100 fraud positives, so PR-AUC is noisy; "
      "N_TRIALS windows per condition gives coarse rates.")
