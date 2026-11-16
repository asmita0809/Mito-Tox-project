#!/usr/bin/env python3
"""
Stage 3 model selection: the pre-registered 540 cross-validation fits.

DEVELOPMENT SET ONLY. The frozen test set is filtered out at load time and its
rows are never materialised. No threshold is chosen here; out-of-fold
predictions are saved so the threshold can be selected afterwards from OOF
development predictions alone.

Search space is read from models.py and is NOT modified.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox import config
SEED = 42
N_CV = 5

# ------------------------------------------------------------------ load
train = pd.read_csv(config.TRAINING_CSV)
splits = pd.read_csv(config.SPLITS_CSV)
desc = pd.read_csv(config.DESCRIPTORS_CSV)
mor = np.load(config.MORGAN_NPZ, allow_pickle=True)
mac = np.load(config.MACCS_NPZ, allow_pickle=True)

keys = train.parent_inchikey.values
assert list(desc.parent_inchikey) == list(keys)
assert list(mor["parent_inchikey"]) == list(keys)
assert list(mac["parent_inchikey"]) == list(keys)

sp = splits.set_index("parent_inchikey").loc[keys]
is_dev = (sp.subset == "development").values
y_all = train.label.values.astype(int)
assert (sp.label.values == y_all).all()

DESC_COLS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
             "NumRotatableBonds"]
X_mor = mor["X"].astype(np.float32)
X_mac = mac["X"].astype(np.float32)
X_desc = desc[DESC_COLS].values.astype(np.float32)

# ---- development only. Test rows are dropped here and never referenced again.
y = y_all[is_dev]
groups = sp.split_group.values[is_dev]
BLOCKS = {"morgan": X_mor[is_dev], "maccs": X_mac[is_dev], "descriptors": X_desc[is_dev]}
N_DEV = int(is_dev.sum())
print(f"development rows {N_DEV}  positives {y.sum()}  prevalence {y.mean():.4f}", flush=True)
print(f"test rows excluded {int((~is_dev).sum())}", flush=True)

REPRESENTATIONS = {
    "A_morgan": ["morgan"],
    "B_morgan_desc": ["morgan", "descriptors"],
    "C_maccs_desc": ["maccs", "descriptors"],
}


def build_X(blocks):
    mats, n_desc = [], 0
    for b in blocks:
        mats.append(BLOCKS[b])
        if b == "descriptors":
            n_desc = BLOCKS[b].shape[1]
    X = np.hstack(mats)
    return X, n_desc


def make_pipeline(est, n_features, n_desc):
    """StandardScaler on the descriptor columns ONLY (they are the last n_desc
    columns), fingerprint bits passed through untouched. Inside the Pipeline so
    it is fitted on the training fold alone."""
    if n_desc == 0:
        return Pipeline([("clf", est)])
    fp_idx = list(range(0, n_features - n_desc))
    d_idx = list(range(n_features - n_desc, n_features))
    ct = ColumnTransformer(
        [("fp", "passthrough", fp_idx), ("desc", StandardScaler(), d_idx)],
        remainder="drop")
    return Pipeline([("prep", ct), ("clf", est)])


def grid(model, npos, nneg):
    if model == "logistic_regression":
        for C in [0.01, 0.1, 1.0, 10.0]:
            for cw in [None, "balanced"]:
                yield ({"C": C, "class_weight": cw},
                       LogisticRegression(penalty="l2", solver="liblinear",
                                          max_iter=2000, C=C, class_weight=cw,
                                          random_state=SEED))
    elif model == "random_forest":
        for mf in ["sqrt", 0.1]:
            for msl in [1, 3, 10]:
                for cw in [None, "balanced_subsample"]:
                    yield ({"max_features": mf, "min_samples_leaf": msl,
                            "class_weight": cw},
                           RandomForestClassifier(n_estimators=500, max_features=mf,
                                                  min_samples_leaf=msl, class_weight=cw,
                                                  n_jobs=2, random_state=SEED))
    elif model == "xgboost":
        spw_pos = nneg / npos
        for md in [3, 6]:
            for lr in [0.05, 0.1]:
                for mcw in [1, 5]:
                    for spw_name, spw in [("1", 1.0), ("neg_over_pos", spw_pos)]:
                        yield ({"max_depth": md, "learning_rate": lr,
                                "min_child_weight": mcw, "scale_pos_weight": spw_name},
                               XGBClassifier(n_estimators=400, max_depth=md,
                                             learning_rate=lr, min_child_weight=mcw,
                                             scale_pos_weight=spw, subsample=0.8,
                                             colsample_bytree=0.8, tree_method="hist",
                                             eval_metric="aucpr", n_jobs=2,
                                             random_state=SEED, verbosity=0))


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    cv = StratifiedGroupKFold(n_splits=N_CV, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.zeros(N_DEV), y, groups=groups))

    rows, oof_store = [], {}
    res_path = config.CV_RESULTS_CSV
    t0 = time.time()
    done = 0
    total = 0
    for rep in REPRESENTATIONS:
        for model in ["logistic_regression", "random_forest", "xgboost"]:
            if only and model != only:
                continue
            total += sum(1 for _ in grid(model, int(y.sum()), int((y == 0).sum())))
    total *= N_CV
    print(f"planned fits: {total}", flush=True)

    for rep, blocks in REPRESENTATIONS.items():
        X, n_desc = build_X(blocks)
        for model in ["logistic_regression", "random_forest", "xgboost"]:
            if only and model != only:
                continue
            for params, est in grid(model, int(y.sum()), int((y == 0).sum())):
                oof = np.full(N_DEV, np.nan)
                per_fold = []
                for fi, (tr, va) in enumerate(folds):
                    pipe = make_pipeline(est, X.shape[1], n_desc)
                    pipe.fit(X[tr], y[tr])
                    p = pipe.predict_proba(X[va])[:, 1]
                    oof[va] = p
                    per_fold.append({
                        "pr_auc": average_precision_score(y[va], p),
                        "roc_auc": roc_auc_score(y[va], p)})
                    done += 1
                    if done % 20 == 0:
                        el = time.time() - t0
                        print(f"  {done}/{total} fits  {el:.0f}s elapsed  "
                              f"eta {el / done * (total - done):.0f}s", flush=True)
                cfg = f"{rep}|{model}|" + ",".join(f"{k}={v}" for k, v in params.items())
                oof_store[cfg] = oof
                rows.append({
                    "representation": rep, "model": model,
                    "params": json.dumps(params), "config": cfg,
                    "pr_auc_mean": np.mean([f["pr_auc"] for f in per_fold]),
                    "pr_auc_sd": np.std([f["pr_auc"] for f in per_fold], ddof=1),
                    "roc_auc_mean": np.mean([f["roc_auc"] for f in per_fold]),
                    "roc_auc_sd": np.std([f["roc_auc"] for f in per_fold], ddof=1),
                    "pr_auc_folds": ";".join(f"{f['pr_auc']:.4f}" for f in per_fold),
                    "oof_pr_auc": average_precision_score(y, oof),
                    "oof_roc_auc": roc_auc_score(y, oof),
                })
                pd.DataFrame(rows).to_csv(res_path, index=False)

    np.savez_compressed(config.OOF_PREDICTIONS_NPZ,
                        y=y, keys=keys[is_dev], groups=groups,
                        **{k: v for k, v in oof_store.items()})
    print(f"DONE {done} fits in {time.time() - t0:.0f}s -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
