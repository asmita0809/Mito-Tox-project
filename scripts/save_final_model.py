#!/usr/bin/env python3
"""Refit the frozen configuration on the development set and save it.

Separated from final_test_eval.py so the artifact can be regenerated in the
environment described by requirements.txt without re-scoring the test set.
The configuration is read from results/model_selection.json - it is not
retyped here, so the saved model cannot drift from the selected one.

Usage:  py -3 scripts/save_final_model.py
Output: models/final_model.joblib
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox import config

sel = json.load(open("results/model_selection.json", encoding="utf-8"))
fin = json.load(open("results/final_test_results.json", encoding="utf-8"))
params = sel["winner"]["params"]
assert sel["winner"]["representation"] == "B_morgan_desc"
assert sel["winner"]["model"] == "xgboost"

DESC = [n for n, _ in config.DESCRIPTORS]
train = pd.read_csv(config.TRAINING_CSV)
splits = pd.read_csv(config.DATA_PROCESSED / "splits.csv")
desc = pd.read_csv(config.DATA_PROCESSED / "features_descriptors.csv")
mor = np.load(config.DATA_PROCESSED / "features_morgan.npz", allow_pickle=True)
keys = train.parent_inchikey.values
sp = splits.set_index("parent_inchikey").loc[keys]
is_dev = (sp.subset == "development").values
y = train.label.values.astype(int)
X = np.hstack([mor["X"].astype(np.float32),
               desc[DESC].values.astype(np.float32)]).astype(np.float32)
nd, nf = len(DESC), X.shape[1]
spw = float((y[is_dev] == 0).sum() / (y[is_dev] == 1).sum())

pipe = Pipeline([
    ("prep", ColumnTransformer([("fp", "passthrough", list(range(nf - nd))),
                                ("desc", StandardScaler(),
                                 list(range(nf - nd, nf)))], remainder="drop")),
    ("clf", XGBClassifier(n_estimators=400, max_depth=params["max_depth"],
                          learning_rate=params["learning_rate"],
                          min_child_weight=params["min_child_weight"],
                          scale_pos_weight=spw, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist",
                          eval_metric="aucpr", n_jobs=2,
                          random_state=config.RANDOM_SEED, verbosity=0))])
pipe.fit(X[is_dev], y[is_dev])
assert pipe.named_steps["prep"].named_transformers_["desc"].n_samples_seen_ == int(is_dev.sum())

os.makedirs("models", exist_ok=True)
joblib.dump({
    "pipeline": pipe,
    "representation": "B_morgan_desc",
    "feature_order": ["morgan_%d" % i for i in range(nf - nd)] + DESC,
    "morgan": {"radius": config.MORGAN_RADIUS, "n_bits": config.MORGAN_NBITS,
               "binary": config.MORGAN_BINARY,
               "include_chirality": config.MORGAN_USE_CHIRALITY},
    "descriptors": DESC,
    "primary_threshold": 0.625,
    "sensitivity_thresholds": {"f1_optimal": 0.390, "youden_j": 0.163},
    "scale_pos_weight": spw,
    "random_state": config.RANDOM_SEED,
    "endpoint": "mitochondrial membrane potential decrease (depolarisation)",
    "label_definition": ("1 = Tox21 AID 720637 Activity Outcome 'Active' "
                         "(= Activity Summary 'active antagonist'); 0 = 'Inactive'"),
    "trained_on": "Tox21 AID 720637 development split, 4,733 parent structures",
    "frozen_test_pr_auc": fin["test_results"]["pr_auc"],
    "frozen_test_roc_auc": fin["test_results"]["roc_auc"],
    "sklearn_version": sklearn.__version__,
    "xgboost_version": xgboost.__version__,
}, "models/final_model.joblib", compress=3)
print("saved models/final_model.joblib  %.2f MB  (sklearn %s, xgboost %s)"
      % (os.path.getsize("models/final_model.joblib") / 1e6,
         sklearn.__version__, xgboost.__version__))
