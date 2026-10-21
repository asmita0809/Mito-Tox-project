#!/usr/bin/env python3
"""
Stage 3 interpretation: TreeSHAP feature importance, plus structural-alert
enrichment recomputed on the DEVELOPMENT SPLIT ONLY.

Run after the frozen test evaluation. Changes no model and no label.

Two deliberate choices:
  * Direction is the Spearman correlation between a feature's value and its own
    SHAP contribution. Mean SIGNED shap is NOT used: on a 15%-prevalence dataset
    its sign is dominated by the majority class and can label a risk-increasing
    feature as protective.
  * Alert odds ratios are recomputed on the 4,733 development compounds. The
    Checkpoint 2 figures used the full labelled set, which included the frozen
    test compounds; this is the pre-registered correction for the final report.

Usage:  py -3 scripts/interpret_model.py
Outputs: results/tables/shap_global_importance.csv
         results/tables/shap_top_morgan_bits.csv
         results/tables/structural_alert_enrichment_dev.csv
         results/interpretation_summary.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import fisher_exact, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox import config

RDLogger.DisableLog("rdApp.*")
P = "data/processed"
T = "results/tables"
DESC = [n for n, _ in config.DESCRIPTORS]
os.makedirs(T, exist_ok=True)

train = pd.read_csv(config.TRAINING_CSV)
splits = pd.read_csv(f"{P}/splits.csv")
desc = pd.read_csv(f"{P}/features_descriptors.csv")
mor = np.load(f"{P}/features_morgan.npz", allow_pickle=True)
alerts = pd.read_csv(f"{P}/structural_alerts.csv")

keys = train.parent_inchikey.values
sp = splits.set_index("parent_inchikey").loc[keys]
is_dev = (sp.subset == "development").values
y = train.label.values.astype(int)
X = np.hstack([mor["X"].astype(np.float32),
               desc[DESC].values.astype(np.float32)]).astype(np.float32)
nd, nf = len(DESC), X.shape[1]
names = [f"morgan_bit_{i}" for i in range(nf - nd)] + DESC

# ------------------------------------------------------------------ SHAP
spw = float((y[is_dev] == 0).sum() / (y[is_dev] == 1).sum())
pipe = Pipeline([
    ("prep", ColumnTransformer([("fp", "passthrough", list(range(nf - nd))),
                                ("desc", StandardScaler(),
                                 list(range(nf - nd, nf)))], remainder="drop")),
    ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                          min_child_weight=1, scale_pos_weight=spw, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist",
                          eval_metric="aucpr", n_jobs=2,
                          random_state=config.RANDOM_SEED, verbosity=0))])
pipe.fit(X[is_dev], y[is_dev])
Xt = pipe.named_steps["prep"].transform(X[is_dev])
contribs = pipe.named_steps["clf"].get_booster().predict(
    xgb.DMatrix(Xt), pred_contribs=True)
shap, bias = contribs[:, :-1], float(contribs[0, -1])

mean_abs = np.abs(shap).mean(axis=0)
rows = []
for j in np.argsort(-mean_abs)[:40]:
    v = Xt[:, j]
    rho = float(spearmanr(v, shap[:, j]).statistic) if np.unique(v).size > 1 else np.nan
    rows.append({"rank": len(rows) + 1, "feature": names[j],
                 "kind": "descriptor" if names[j] in DESC else "morgan_bit",
                 "mean_abs_shap": float(mean_abs[j]),
                 "value_vs_contribution_spearman": rho,
                 "direction": ("higher value -> higher predicted risk"
                               if rho == rho and rho > 0.1 else
                               "higher value -> lower predicted risk"
                               if rho == rho and rho < -0.1 else
                               "no monotone direction"),
                 "n_dev_compounds_with_feature": int((X[is_dev][:, j] > 0).sum())})
imp = pd.DataFrame(rows)
imp.to_csv(f"{T}/shap_global_importance.csv", index=False)
desc_imp = imp[imp.kind == "descriptor"]

# ---- example substructure for the top Morgan bits (one pass over a sample)
want = [int(f.rsplit("_", 1)[1]) for f in imp[imp.kind == "morgan_bit"].feature.head(10)]
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS,
    includeChirality=config.MORGAN_USE_CHIRALITY)
found = {}
dev_smi = train.parent_smiles.values[is_dev]
rng = np.random.default_rng(config.RANDOM_SEED)
for i in rng.permutation(len(dev_smi)):
    if len(found) == len(want):
        break
    m = Chem.MolFromSmiles(dev_smi[i])
    if m is None:
        continue
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(m, additionalOutput=ao)
    info = ao.GetBitInfoMap()
    for b in want:
        if b in found or b not in info:
            continue
        atom, rad = info[b][0]
        env = Chem.FindAtomEnvironmentOfRadiusN(m, rad, atom) if rad else None
        if env:
            at = {a for bd in env for a in
                  (m.GetBondWithIdx(bd).GetBeginAtomIdx(),
                   m.GetBondWithIdx(bd).GetEndAtomIdx())}
            sub = Chem.MolFragmentToSmiles(m, atomsToUse=sorted(at),
                                           bondsToUse=list(env), canonical=True)
        else:
            sub = m.GetAtomWithIdx(atom).GetSymbol()
        found[b] = {"morgan_bit": b, "radius": rad, "example_substructure": sub,
                    "example_parent_smiles": dev_smi[i]}
bits = pd.DataFrame([found[b] for b in want if b in found])
bits = bits.merge(imp[["feature", "mean_abs_shap",
                       "value_vs_contribution_spearman",
                       "n_dev_compounds_with_feature"]],
                  left_on=bits.morgan_bit.map(lambda b: f"morgan_bit_{b}"),
                  right_on="feature", how="left").drop(columns=["feature"])
bits.to_csv(f"{T}/shap_top_morgan_bits.csv", index=False)

# --------------------------------------------- alerts, development only
al = alerts.set_index("parent_inchikey").loc[keys]
ydev = y[is_dev]
npos, nneg = int((ydev == 1).sum()), int((ydev == 0).sum())
old = pd.read_csv(f"{T}/structural_alert_enrichment.csv")
oldmap = dict(zip(old.alert, old.odds_ratio))
recs = []
for k in al.columns:
    hit = al[k].values[is_dev].astype(bool)
    a1 = int((hit & (ydev == 1)).sum())
    a0 = int((hit & (ydev == 0)).sum())
    orr, p = fisher_exact([[a1, a0], [npos - a1, nneg - a0]])
    recs.append({"alert": k, "n_matches_dev": int(hit.sum()), "n_active": a1,
                 "n_inactive": a0, "pct_of_dev_actives": 100.0 * a1 / npos,
                 "odds_ratio_dev": orr, "fisher_p_dev": p,
                 "odds_ratio_full_set_checkpoint2": oldmap.get(k, np.nan)})
en = pd.DataFrame(recs).sort_values("odds_ratio_dev", ascending=False)
en.to_csv(f"{T}/structural_alert_enrichment_dev.csv", index=False)

with open("results/interpretation_summary.json", "w", encoding="utf-8") as fh:
    json.dump({"shap": {"computed_on": "development set only (4733 compounds)",
                        "method": "XGBoost TreeSHAP (pred_contribs)",
                        "bias_term": bias,
                        "direction_method": ("Spearman rho between feature value "
                                             "and its own SHAP contribution"),
                        "descriptors": desc_imp.to_dict("records")},
               "alerts": {"computed_on": "development set only",
                          "n_development": int(is_dev.sum()),
                          "n_positive": npos, "n_negative": nneg}},
              fh, indent=2, default=float)

print("TOP 15 FEATURES BY MEAN |SHAP|")
print(imp.head(15)[["rank", "feature", "mean_abs_shap",
                    "value_vs_contribution_spearman",
                    "n_dev_compounds_with_feature"]].to_string(index=False))
print("\nTHE SIX DESCRIPTORS")
print(desc_imp[["rank", "feature", "mean_abs_shap",
                "value_vs_contribution_spearman", "direction"]].to_string(index=False))
print("\nALERT ENRICHMENT, DEVELOPMENT ONLY (vs Checkpoint 2 full-set value)")
print(en[["alert", "n_matches_dev", "n_active", "n_inactive", "odds_ratio_dev",
          "fisher_p_dev", "odds_ratio_full_set_checkpoint2"]].round(4).to_string(index=False))
print("\nTOP MORGAN BITS - EXAMPLE SUBSTRUCTURES")
print(bits.to_string(index=False))
