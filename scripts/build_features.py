#!/usr/bin/env python3
"""
Stage 2 / Checkpoint 2: molecular feature generation.

Builds three SEPARATE feature blocks from the approved training set and reports
their statistics. Structural alerts are computed too, but as post-hoc
interpretation only - they are never written into a feature matrix.

No model is trained, no split is made, no test set is touched (none exists yet).

Usage:  py -3 scripts/build_features.py
Outputs:
    data/processed/features_morgan.npz        5938 x 2048 uint8
    data/processed/features_maccs.npz         5938 x 167  uint8
    data/processed/features_descriptors.csv   5938 x 6
    data/processed/feature_metadata.json      every setting, for reproducibility
    data/processed/structural_alerts.csv      post-hoc alert matches (NOT features)
    results/tables/descriptor_statistics.csv
    results/tables/structural_alert_enrichment.csv
    results/checkpoint_2_features.txt
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rdkit import RDLogger, rdBase
import rdkit
from mitotox import config, features
from mitotox.interpretation import ALERTS, compile_alerts, match_alerts

RDLogger.DisableLog("rdApp.*")
_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def h(t):
    w("")
    w("=" * 78)
    w(t)
    w("=" * 78)


def main():
    os.makedirs(config.TABLES, exist_ok=True)
    df = pd.read_csv(config.TRAINING_CSV)
    w("input : %s" % config.TRAINING_CSV.name)
    w("rows  : %d   positives %d   negatives %d   prevalence %.2f%%"
      % (len(df), int((df.label == 1).sum()), int((df.label == 0).sum()),
         100.0 * (df.label == 1).mean()))

    h("1. SETTINGS (all from src/mitotox/config.py)")
    w("RDKit version            : %s" % rdkit.__version__)
    w("Morgan radius            : %d   (ECFP4-equivalent)" % config.MORGAN_RADIUS)
    w("Morgan bits              : %d" % config.MORGAN_NBITS)
    w("Morgan binary            : %s" % config.MORGAN_BINARY)
    w("Morgan includeChirality  : %s" % config.MORGAN_USE_CHIRALITY)
    w("MACCS bits               : %d" % config.MACCS_NBITS)
    w("random seed              : %d" % config.RANDOM_SEED)
    w("alerts used as features  : %s" % config.ALERTS_ARE_MODEL_FEATURES)
    w("")
    w("Descriptors (exactly the six in the brief):")
    for name, desc in config.DESCRIPTORS:
        w("  %-20s %s" % (name, desc))

    # ---------------- build ----------------
    h("2. FEATURE GENERATION")
    mols = [features.mol_from_smiles(s) for s in df.parent_smiles]
    n_bad = sum(1 for m in mols if m is None)
    w("molecules built from parent_smiles : %d / %d  (failures %d)"
      % (len(mols) - n_bad, len(mols), n_bad))

    morgan = np.zeros((len(df), config.MORGAN_NBITS), dtype=np.uint8)
    maccs = np.zeros((len(df), config.MACCS_NBITS), dtype=np.uint8)
    fail_fp = []
    for i, m in enumerate(mols):
        fm = features.morgan_fp(m)
        km = features.maccs_fp(m)
        if fm is None or km is None:
            fail_fp.append(i)
            continue
        morgan[i] = fm
        maccs[i] = km
    w("fingerprint failures               : %d" % len(fail_fp))

    desc = pd.DataFrame([features.descriptors(m) for m in mols])
    desc.insert(0, "parent_inchikey", df.parent_inchikey.values)
    n_nan = int(desc[[n for n, _ in config.DESCRIPTORS]].isna().any(axis=1).sum())
    w("rows with any missing descriptor   : %d" % n_nan)
    w("")
    w("MISSING-VALUE POLICY: none applied. No imputation, no dropping. If a")
    w("descriptor were missing it would be written as an empty cell and dealt")
    w("with explicitly at model time. With %d missing rows there is nothing to do."
      % n_nan)

    np.savez_compressed(config.DATA_PROCESSED / "features_morgan.npz",
                        X=morgan, parent_inchikey=df.parent_inchikey.values,
                        label=df.label.values)
    np.savez_compressed(config.DATA_PROCESSED / "features_maccs.npz",
                        X=maccs, parent_inchikey=df.parent_inchikey.values,
                        label=df.label.values)
    desc.to_csv(config.DATA_PROCESSED / "features_descriptors.csv", index=False)

    # ---------------- fingerprint statistics ----------------
    h("3. FINGERPRINT STATISTICS")
    for name, X in [("Morgan", morgan), ("MACCS", maccs)]:
        on_per_mol = X.sum(axis=1)
        col_freq = X.sum(axis=0)
        w("")
        w("%s  shape %s  dtype %s" % (name, X.shape, X.dtype))
        w("  bits set per molecule : mean %.1f  median %d  min %d  max %d"
          % (on_per_mol.mean(), int(np.median(on_per_mol)),
             int(on_per_mol.min()), int(on_per_mol.max())))
        w("  overall bit density   : %.3f%%" % (100.0 * X.mean()))
        w("  all-zero columns      : %d / %d" % (int((col_freq == 0).sum()), X.shape[1]))
        w("  all-one columns       : %d" % int((col_freq == len(X)).sum()))
        w("  columns set in <5 molecules  : %d" % int(((col_freq > 0) & (col_freq < 5)).sum()))
        w("  columns set in >=1%% of set   : %d" % int((col_freq >= 0.01 * len(X)).sum()))
        w("  molecules with zero bits set : %d" % int((on_per_mol == 0).sum()))

    # ---------------- descriptor statistics ----------------
    h("4. DESCRIPTOR STATISTICS")
    names = [n for n, _ in config.DESCRIPTORS]
    stats = desc[names].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    stats.columns = ["n", "mean", "sd", "min", "median", "max"]
    lab = df.label.values
    for n in names:
        v = desc[n].values
        stats.loc[n, "mean_pos"] = np.nanmean(v[lab == 1])
        stats.loc[n, "mean_neg"] = np.nanmean(v[lab == 0])
        stats.loc[n, "diff"] = stats.loc[n, "mean_pos"] - stats.loc[n, "mean_neg"]
    stats.round(3).to_csv(config.TABLES / "descriptor_statistics.csv")
    for line in stats.round(2).to_string().split("\n"):
        w("  " + line)
    w("")
    w("  'mean_pos' / 'mean_neg' are descriptive only. No split exists yet, so")
    w("  these are computed on the whole labelled set and must NOT be used to")
    w("  choose features or thresholds.")

    # ---------------- structural alerts (post-hoc) ----------------
    h("5. STRUCTURAL ALERTS - POST-HOC INTERPRETATION, NOT FEATURES")
    compiled = compile_alerts()
    rows = [match_alerts(m, compiled) for m in mols]
    al = pd.DataFrame(rows)
    al.insert(0, "parent_inchikey", df.parent_inchikey.values)
    al.to_csv(config.DATA_PROCESSED / "structural_alerts.csv", index=False)

    from scipy.stats import fisher_exact
    recs = []
    npos, nneg = int((lab == 1).sum()), int((lab == 0).sum())
    for a in ALERTS:
        hit = al[a.key].values
        a1 = int((hit & (lab == 1)).sum())
        a0 = int((hit & (lab == 0)).sum())
        b1, b0 = npos - a1, nneg - a0
        table = [[a1, a0], [b1, b0]]
        try:
            orr, p = fisher_exact(table)
        except Exception:
            orr, p = float("nan"), float("nan")
        recs.append({"alert": a.key, "name": a.name, "n_matches": int(hit.sum()),
                     "n_active": a1, "n_inactive": a0,
                     "pct_of_actives": 100.0 * a1 / max(1, npos),
                     "odds_ratio": orr, "fisher_p": p,
                     "rationale": a.rationale})
    en = pd.DataFrame(recs).sort_values("odds_ratio", ascending=False)
    en.to_csv(config.TABLES / "structural_alert_enrichment.csv", index=False)
    w("%-20s %7s %7s %7s %9s %11s" % ("alert", "matches", "active", "inact", "OR", "Fisher p"))
    w("-" * 70)
    for _, r in en.iterrows():
        w("%-20s %7d %7d %7d %9.2f %11.2e"
          % (r.alert, r.n_matches, r.n_active, r.n_inactive, r.odds_ratio, r.fisher_p))
    w("")
    w("CAVEAT: enrichment is computed on the full labelled set because no")
    w("train/test split exists yet. These numbers are descriptive. Because the")
    w("alerts are never model features, this is not leakage - but the figures")
    w("must be recomputed on the TRAINING SPLIT ONLY for the final report.")

    # ---------------- metadata ----------------
    meta = {
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rdkit_version": rdkit.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "input_file": str(config.TRAINING_CSV.name),
        "n_compounds": int(len(df)),
        "n_positive": int((df.label == 1).sum()),
        "n_negative": int((df.label == 0).sum()),
        "structure_column": "parent_smiles",
        "identifier_column": "parent_inchikey",
        "blocks": {
            "morgan": {"file": "features_morgan.npz", "shape": list(morgan.shape),
                       "dtype": "uint8", "radius": config.MORGAN_RADIUS,
                       "n_bits": config.MORGAN_NBITS, "binary": config.MORGAN_BINARY,
                       "include_chirality": config.MORGAN_USE_CHIRALITY,
                       "generator": "rdFingerprintGenerator.GetMorganGenerator"},
            "maccs": {"file": "features_maccs.npz", "shape": list(maccs.shape),
                      "dtype": "uint8", "n_bits": config.MACCS_NBITS,
                      "generator": "rdkit.Chem.MACCSkeys.GenMACCSKeys",
                      "note": "bit 0 is unused by RDKit and is always 0"},
            "descriptors": {"file": "features_descriptors.csv",
                            "columns": [{"name": n, "definition": d}
                                        for n, d in config.DESCRIPTORS],
                            "n_missing_rows": n_nan,
                            "missing_value_policy": "none applied; no imputation"},
        },
        "structural_alerts": {
            "file": "structural_alerts.csv",
            "used_as_model_features": config.ALERTS_ARE_MODEL_FEATURES,
            "n_alerts": len(ALERTS),
            "definitions": [{"key": a.key, "name": a.name, "smarts": list(a.smarts),
                             "require": a.require, "rationale": a.rationale}
                            for a in ALERTS],
        },
        "random_seed": config.RANDOM_SEED,
    }
    with open(config.DATA_PROCESSED / "feature_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    h("6. FILES WRITTEN")
    for f in ["features_morgan.npz", "features_maccs.npz", "features_descriptors.csv",
              "structural_alerts.csv", "feature_metadata.json"]:
        p = config.DATA_PROCESSED / f
        w("  data/processed/%-28s %10d bytes" % (f, p.stat().st_size))
    for f in ["descriptor_statistics.csv", "structural_alert_enrichment.csv"]:
        p = config.TABLES / f
        w("  results/tables/%-28s %10d bytes" % (f, p.stat().st_size))

    with open(config.RESULTS / "checkpoint_2_features.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Report: results/checkpoint_2_features.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
