#!/usr/bin/env python3
"""
Stage 3 model selection and threshold selection.

Uses ONLY development out-of-fold predictions. The frozen test set is not
loaded, not referenced, and not scored.

Selection rule (pre-registered, unchanged):
    highest mean PR-AUC across the 5 development CV folds;
    ties broken by mean MCC at the CV-selected threshold.

Threshold rule (pre-registered, unchanged):
    maximise MCC on out-of-fold development predictions.
    F1-optimal and Youden-J reported as sensitivity analyses only.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_curve)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox import config

res = pd.read_csv(config.CV_RESULTS_CSV)
z = np.load(config.OOF_PREDICTIONS_NPZ, allow_pickle=True)
y = z["y"]


def metrics_at(y_true, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yh, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "mcc": matthews_corrcoef(y_true, yh),
        "f1": f1_score(y_true, yh, zero_division=0),
        "precision": precision_score(y_true, yh, zero_division=0),
        "recall_sensitivity": recall_score(y_true, yh, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "balanced_accuracy": 0.5 * (recall_score(y_true, yh, zero_division=0)
                                    + (tn / (tn + fp) if (tn + fp) else 0.0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_true, p, objective="mcc"):
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 197), 4))
    best, best_v = 0.5, -2
    for t in grid:
        yh = (p >= t).astype(int)
        v = (matthews_corrcoef(y_true, yh) if objective == "mcc"
             else f1_score(y_true, yh, zero_division=0))
        if v > best_v:
            best_v, best = v, t
    return float(best), float(best_v)


# ---- rank by the pre-registered primary metric
res = res.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)

# ---- tie-break by mean MCC at the CV-selected threshold, over the top ties
top = res.pr_auc_mean.iloc[0]
ties = res[np.isclose(res.pr_auc_mean, top, atol=1e-6)]
mccs = {}
for cfg in ties.config:
    thr, _ = best_threshold(y, z[cfg], "mcc")
    mccs[cfg] = metrics_at(y, z[cfg], thr)["mcc"]
winner_cfg = max(mccs, key=mccs.get) if len(ties) > 1 else res.config.iloc[0]
win = res[res.config == winner_cfg].iloc[0]

p = z[winner_cfg]
thr_mcc, v_mcc = best_threshold(y, p, "mcc")
thr_f1, v_f1 = best_threshold(y, p, "f1")
fpr, tpr, thr_roc = roc_curve(y, p)
thr_j = float(thr_roc[np.argmax(tpr - fpr)])

report = {
    "winner": {
        "config": winner_cfg,
        "representation": win.representation,
        "model": win.model,
        "params": json.loads(win.params),
        "pr_auc_mean": float(win.pr_auc_mean),
        "pr_auc_sd": float(win.pr_auc_sd),
        "pr_auc_folds": win.pr_auc_folds,
        "roc_auc_mean": float(win.roc_auc_mean),
        "roc_auc_sd": float(win.roc_auc_sd),
        "oof_pr_auc": float(win.oof_pr_auc),
        "oof_roc_auc": float(win.oof_roc_auc),
        "n_tied_on_primary": int(len(ties)),
    },
    "threshold_selected": metrics_at(y, p, thr_mcc),
    "sensitivity_f1_optimal": metrics_at(y, p, thr_f1),
    "sensitivity_youden_j": metrics_at(y, p, thr_j),
    "prevalence_development": float(y.mean()),
    "n_development": int(len(y)),
    "note": ("all figures computed on development out-of-fold predictions only; "
             "the frozen test set has not been loaded or scored"),
}
with open(config.MODEL_SELECTION_JSON, "w") as fh:
    json.dump(report, fh, indent=2)

res.to_csv(config.CV_RESULTS_RANKED_CSV, index=False)
print(json.dumps(report, indent=2))
