#!/usr/bin/env python3
"""
Stage 3 final evaluation: the frozen test set, scored exactly once.

Refits the frozen configuration on the full development set (4,733 compounds),
then applies it once to the frozen test set (1,205 compounds). No threshold is
optimised here; the three thresholds were fixed at Checkpoint 3C from
development out-of-fold predictions.

FROZEN CONFIGURATION (unchanged from Checkpoint 3C):
    representation   B_morgan_desc  (2048 Morgan bits + 6 scaled descriptors)
    model            XGBClassifier
    max_depth=6, learning_rate=0.1, min_child_weight=1,
    scale_pos_weight=neg_over_pos, n_estimators=400,
    subsample=0.8, colsample_bytree=0.8, tree_method=hist, random_state=42

    primary threshold  0.625   (MCC-max on development OOF)
    sensitivity        0.390   (F1-optimal on development OOF)
    sensitivity        0.163   (Youden-J on development OOF)
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA = "data/processed"
UP = "data/processed"
OUT = "results"
SEED = 42
BOOTSTRAP_SEED = 20260910          # recorded; fixed before any test metric was seen
N_BOOTSTRAP = 2000

THRESHOLDS = {"primary_mcc_max": 0.625,
              "sensitivity_f1_optimal": 0.390,
              "sensitivity_youden_j": 0.163}

DESC_COLS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
             "NumRotatableBonds"]

notes = []

# ------------------------------------------------------------------ load
train = pd.read_csv(f"{DATA}/tox21_mmp_training.csv")
splits = pd.read_csv(f"{DATA}/splits.csv")
desc = pd.read_csv(f"{DATA}/features_descriptors.csv")
mor = np.load(f"{DATA}/features_morgan.npz", allow_pickle=True)

keys = train.parent_inchikey.values
assert list(desc.parent_inchikey) == list(keys)
assert list(mor["parent_inchikey"]) == list(keys)

sp = splits.set_index("parent_inchikey").loc[keys]
is_dev = (sp.subset == "development").values
is_test = (sp.subset == "test").values
assert is_dev.sum() == 4733 and is_test.sum() == 1205, "split sizes changed"
assert not (is_dev & is_test).any()

y_all = train.label.values.astype(int)
X_all = np.hstack([mor["X"].astype(np.float32),
                   desc[DESC_COLS].values.astype(np.float32)]).astype(np.float32)
n_desc = len(DESC_COLS)
n_feat = X_all.shape[1]

X_dev, y_dev = X_all[is_dev], y_all[is_dev]
X_test, y_test = X_all[is_test], y_all[is_test]
print(f"development {X_dev.shape}  positives {y_dev.sum()}  prevalence {y_dev.mean():.4f}")
print(f"test        {X_test.shape}  positives {y_test.sum()}  prevalence {y_test.mean():.4f}")

# ------------------------------------------------------------------ refit
spw = float((y_dev == 0).sum() / (y_dev == 1).sum())
print(f"scale_pos_weight (development neg/pos) = {spw:.4f}")

fp_idx = list(range(0, n_feat - n_desc))
d_idx = list(range(n_feat - n_desc, n_feat))
pipe = Pipeline([
    ("prep", ColumnTransformer([("fp", "passthrough", fp_idx),
                                ("desc", StandardScaler(), d_idx)], remainder="drop")),
    ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                          min_child_weight=1, scale_pos_weight=spw,
                          subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", eval_metric="aucpr",
                          n_jobs=2, random_state=SEED, verbosity=0))])
pipe.fit(X_dev, y_dev)

# the scaler must have been fitted on development statistics only
sc = pipe.named_steps["prep"].named_transformers_["desc"]
assert sc.n_samples_seen_ == len(X_dev), "scaler saw rows other than development"
print(f"scaler fitted on {sc.n_samples_seen_} development rows only")

p_test = pipe.predict_proba(X_test)[:, 1]
p_dev_insample = pipe.predict_proba(X_dev)[:, 1]


def metrics_at(y_true, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yh, labels=[0, 1]).ravel()
    sens = recall_score(y_true, yh, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"threshold": float(thr),
            "mcc": float(matthews_corrcoef(y_true, yh)),
            "balanced_accuracy": float(0.5 * (sens + spec)),
            "sensitivity_recall": float(sens),
            "specificity": float(spec),
            "precision": float(precision_score(y_true, yh, zero_division=0)),
            "f1": float(f1_score(y_true, yh, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


test_res = {
    "pr_auc": float(average_precision_score(y_test, p_test)),
    "roc_auc": float(roc_auc_score(y_test, p_test)),
    "thresholds": {k: metrics_at(y_test, p_test, v) for k, v in THRESHOLDS.items()},
}

# ------------------------------------------------------------------ bootstrap
rng = np.random.default_rng(BOOTSTRAP_SEED)
n = len(y_test)
acc = {"pr_auc": [], "roc_auc": []}
for k in THRESHOLDS:
    for m in ["mcc", "balanced_accuracy", "sensitivity_recall", "specificity",
              "precision", "f1"]:
        acc[f"{k}::{m}"] = []
skipped = 0
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, n)
    yb, pb = y_test[idx], p_test[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        skipped += 1
        continue
    acc["pr_auc"].append(average_precision_score(yb, pb))
    acc["roc_auc"].append(roc_auc_score(yb, pb))
    for k, t in THRESHOLDS.items():
        mm = metrics_at(yb, pb, t)
        for m in ["mcc", "balanced_accuracy", "sensitivity_recall", "specificity",
                  "precision", "f1"]:
            acc[f"{k}::{m}"].append(mm[m])
if skipped:
    notes.append(f"{skipped} bootstrap resamples skipped (single-class draw)")

ci = {k: {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
          "mean": float(np.mean(v))} for k, v in acc.items() if v}

# ------------------------------------------------------------------ gap
cvres = pd.read_csv(f"{OUT}/cv_results_ranked.csv")
sel = json.load(open(f"{OUT}/model_selection.json"))
win = cvres[cvres.config == sel["winner"]["config"]].iloc[0]
gap = {
    "cv_pr_auc_mean": float(win.pr_auc_mean), "cv_pr_auc_sd": float(win.pr_auc_sd),
    "oof_pr_auc": float(win.oof_pr_auc), "test_pr_auc": test_res["pr_auc"],
    "pr_auc_gap_cv_minus_test": float(win.pr_auc_mean - test_res["pr_auc"]),
    "cv_roc_auc_mean": float(win.roc_auc_mean), "oof_roc_auc": float(win.oof_roc_auc),
    "test_roc_auc": test_res["roc_auc"],
    "roc_auc_gap_cv_minus_test": float(win.roc_auc_mean - test_res["roc_auc"]),
    "oof_mcc_at_primary": sel["threshold_selected"]["mcc"],
    "test_mcc_at_primary": test_res["thresholds"]["primary_mcc_max"]["mcc"],
    "in_sample_development_pr_auc": float(average_precision_score(y_dev, p_dev_insample)),
}

# ------------------------------------------------------------------ per-compound
nn = pd.read_csv(f"{UP}/test_nn_similarity.csv")
pred = pd.DataFrame({"parent_inchikey": keys[is_test], "label": y_test,
                     "predicted_probability": p_test})
for k, t in THRESHOLDS.items():
    pred[f"pred_{k}"] = (p_test >= t).astype(int)
pred = pred.merge(nn[["parent_inchikey", "nn_tanimoto", "nn_development_inchikey"]],
                  on="parent_inchikey", how="left")
assert pred.nn_tanimoto.notna().all(), "missing NN similarity for some test compounds"
pred = pred.merge(train[["parent_inchikey", "parent_smiles", "pubchem_cids",
                         "flag_metal_removed", "flag_true_mixture"]],
                  on="parent_inchikey", how="left")
pred.to_csv(f"{OUT}/test_predictions.csv", index=False)

bands = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]
prox = []
for lo, hi in bands:
    m = (pred.nn_tanimoto >= lo) & (pred.nn_tanimoto < hi)
    sub = pred[m]
    row = {"nn_band": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()),
           "n_pos": int(sub.label.sum())}
    if sub.label.nunique() == 2:
        row["pr_auc"] = float(average_precision_score(sub.label, sub.predicted_probability))
        row["roc_auc"] = float(roc_auc_score(sub.label, sub.predicted_probability))
        row["mcc_at_0.625"] = float(matthews_corrcoef(
            sub.label, (sub.predicted_probability >= 0.625).astype(int)))
    prox.append(row)
prox_df = pd.DataFrame(prox)
prox_df.to_csv(f"{OUT}/test_performance_by_proximity.csv", index=False)

report = {
    "frozen_configuration": {
        "representation": "B_morgan_desc", "model": "xgboost",
        "max_depth": 6, "learning_rate": 0.1, "min_child_weight": 1,
        "scale_pos_weight": spw, "n_estimators": 400, "subsample": 0.8,
        "colsample_bytree": 0.8, "tree_method": "hist", "random_state": SEED},
    "n_development": int(is_dev.sum()), "n_test": int(is_test.sum()),
    "test_positives": int(y_test.sum()), "test_negatives": int((y_test == 0).sum()),
    "test_prevalence": float(y_test.mean()),
    "test_results": test_res,
    "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                  "method": "percentile, 95%", "ci": ci},
    "generalization_gap": gap,
    "performance_by_nn_proximity": prox,
    "implementation_notes": notes,
}
with open(f"{OUT}/final_test_results.json", "w") as fh:
    json.dump(report, fh, indent=2)
print(json.dumps({k: report[k] for k in
                  ["test_results", "generalization_gap"]}, indent=2))
print("\nby proximity:\n" + prox_df.to_string(index=False))
print("\nnotes:", notes or "none")
