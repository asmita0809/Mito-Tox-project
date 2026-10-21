"""Pre-registered model and hyperparameter specifications.

Frozen BEFORE any training run. Nothing here fits a model; this module only
declares what will be searched, so the search space cannot be quietly widened
after seeing results.

Three feature representations (locked at Checkpoint 2):
    A  morgan                 2048 binary bits
    B  morgan + descriptors   2048 bits + 6 scaled descriptors
    C  maccs  + descriptors    167 bits + 6 scaled descriptors

Selection rule (locked):
    best mean PR-AUC (average precision) over the 5 development CV folds;
    ties broken by mean MCC at the CV-selected threshold.
    The test set is evaluated exactly once, after this is final.
"""

REPRESENTATIONS = {
    "A_morgan": {"blocks": ["morgan"], "scale_descriptors": False},
    "B_morgan_desc": {"blocks": ["morgan", "descriptors"], "scale_descriptors": True},
    "C_maccs_desc": {"blocks": ["maccs", "descriptors"], "scale_descriptors": True},
}

# Descriptor scaling is StandardScaler applied to the six descriptor columns ONLY
# (fingerprint bits are left untouched), inside a sklearn Pipeline so it is fitted
# on each training fold alone and never on validation or test data.
SCALER = "StandardScaler on descriptor columns only, inside the Pipeline"

MODEL_SPECS = {
    "logistic_regression": {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "fixed": {"penalty": "l2", "solver": "liblinear", "max_iter": 2000},
        "grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        "n_combinations": 8,
        "role": "interpretable linear baseline",
    },
    "random_forest": {
        "estimator": "sklearn.ensemble.RandomForestClassifier",
        "fixed": {"n_estimators": 500, "n_jobs": -1},
        "grid": {
            "max_features": ["sqrt", 0.1],
            "min_samples_leaf": [1, 3, 10],
            "class_weight": [None, "balanced_subsample"],
        },
        "n_combinations": 12,
        "role": "bagged trees",
    },
    "xgboost": {
        "estimator": "xgboost.XGBClassifier",
        "fixed": {"n_estimators": 400, "tree_method": "hist", "subsample": 0.8,
                  "colsample_bytree": 0.8, "eval_metric": "aucpr", "n_jobs": -1},
        "grid": {
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "min_child_weight": [1, 5],
            "scale_pos_weight": [1.0, "neg_over_pos"],
        },
        "n_combinations": 16,
        "role": "boosted trees",
    },
}

PRIMARY_METRIC = "average_precision"      # PR-AUC
SECONDARY_METRICS = ["roc_auc", "mcc", "balanced_accuracy", "sensitivity",
                     "specificity", "precision", "f1", "confusion_matrix"]
THRESHOLD_RULE = ("maximise MCC on the out-of-fold development predictions; "
                  "F1-optimal and Youden-J thresholds reported as a sensitivity "
                  "analysis. Never selected on the test set.")
IMBALANCE_POLICY = ("no resampling of any kind. Class weighting is a searched "
                    "hyperparameter; the decision threshold absorbs the rest. "
                    "No negatives are deleted.")
UNCERTAINTY = ("fold-to-fold spread across the 5 development CV folds, plus "
               "2000-resample bootstrap confidence intervals on the single "
               "frozen test set.")

TOTAL_FITS = (sum(m["n_combinations"] for m in MODEL_SPECS.values())
              * len(REPRESENTATIONS) * 5)   # 36 * 3 * 5 = 540


# ---------------------------------------------------------------- reporting rules
# Locked by the user at Checkpoint 2 approval. These constrain how results are
# WORDED, not how they are computed. Read before writing any interpretation.
REPORTING_CONSTRAINTS = [
    "The MolLogP / MolWt separation between actives and inactives is an "
    "ASSOCIATION consistent with a physicochemical difference between the "
    "classes. Do NOT describe it as proof that membrane partitioning is the "
    "causal mechanism.",
    "Structural alerts are post-hoc interpretation. Never describe them as "
    "model features or as validated mechanisms.",
    "Model output is a predicted mitochondrial-toxicity risk / computational "
    "prioritisation. Never 'confirmed toxic'. Never a regulatory or medical claim.",
    "Alert enrichment quoted in the final report must be recomputed on the "
    "development split alone, not the full labelled set.",
]

# Feature-selection provenance, locked at Checkpoint 2:
# the representation was pre-specified BEFORE modelling. A seventh acidity
# descriptor was proposed after seeing class enrichment and was REJECTED to keep
# outcome-informed feature selection out of the pre-specified representation.
REPRESENTATION_IS_PRESPECIFIED = True
