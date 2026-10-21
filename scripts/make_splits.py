#!/usr/bin/env python3
"""
Checkpoint 3A: construct and CHARACTERISE the train / CV / test split.

Builds the split only. NO model is fitted, NO test-set performance is computed,
NO threshold is chosen. Writes the split assignment so Stage 3 is deterministic.

Usage:  py -3 scripts/make_splits.py
Outputs:
    data/processed/splits.csv                 parent_inchikey, group, fold, subset
    results/checkpoint_3A_split_report.txt
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sklearn.model_selection import StratifiedGroupKFold
from rdkit import RDLogger
from mitotox import config, splits

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


def describe(name, groups, y):
    g = pd.Series(groups)
    sizes = g.value_counts()
    w("%-10s groups %5d   largest %5d   singletons %5d   median size %.1f"
      % (name, len(sizes), int(sizes.max()), int((sizes == 1).sum()), sizes.median()))
    return sizes


def fold_table(label, idx_sets, y, groups):
    w("")
    w("  %-14s %7s %8s %8s %11s %8s" % (label, "n", "pos", "neg", "prevalence", "groups"))
    w("  " + "-" * 62)
    for nm, idx in idx_sets:
        yy = y[idx]
        w("  %-14s %7d %8d %8d %10.2f%% %8d"
          % (nm, len(idx), int(yy.sum()), int((yy == 0).sum()),
             100.0 * yy.mean(), len(set(groups[idx]))))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouping", choices=["murcko", "butina"], default=config.SPLIT_GROUPING,
                    help="analogue-grouping key used to keep related compounds together")
    ap.add_argument("--butina-cutoff", type=float, default=config.BUTINA_CUTOFF,
                    help="Butina DISTANCE cutoff; 0.4 == Tanimoto similarity 0.6")
    args = ap.parse_args()
    df = pd.read_csv(config.TRAINING_CSV)
    y = df.label.values.astype(int)
    keys = df.parent_inchikey.values
    w("dataset: %d parents, %d positive, %d negative, prevalence %.2f%%"
      % (len(df), int(y.sum()), int((y == 0).sum()), 100.0 * y.mean()))

    h("1. GROUPING OPTIONS (evidence for the choice)")
    gm = splits.murcko_groups(df.parent_smiles.values, keys)
    sm = describe("murcko", gm, y)
    n_acyc = int(pd.Series(gm).str.startswith("ACYCLIC::").sum())
    w("           acyclic compounds given singleton groups: %d (%.1f%%), prevalence %.1f%%"
      % (n_acyc, 100.0 * n_acyc / len(df),
         100.0 * y[pd.Series(gm).str.startswith("ACYCLIC::").values].mean()))
    w("           largest ring scaffold: %r with %d compounds"
      % (sm.index[0], int(sm.iloc[0])))
    w("           that single group is %.1f%% of the whole dataset"
      % (100.0 * sm.iloc[0] / len(df)))

    gb = splits.butina_groups(df.parent_smiles.values, cutoff=args.butina_cutoff)
    describe("butina", gb, y)
    w("           Butina distance cutoff %.2f (Tanimoto similarity %.2f)"
      % (args.butina_cutoff, 1 - args.butina_cutoff))

    gm_all, gb_all = gm, gb
    gm = gm_all if args.grouping == "murcko" else gb_all

    h("2. SPLIT CONSTRUCTION")
    w("Grouping key      : %s" % ("Bemis-Murcko scaffold (acyclic -> singleton)"
                                  if args.grouping == "murcko" else
                                  "Taylor-Butina cluster of Morgan fingerprints"))
    w("Outer split       : StratifiedGroupKFold(n_splits=5, shuffle=True,")
    w("                    random_state=%d); fold 0 is the FROZEN TEST SET." % config.RANDOM_SEED)
    w("Inner split       : StratifiedGroupKFold(n_splits=5, shuffle=True,")
    w("                    random_state=%d) on the development set, used for all" % config.RANDOM_SEED)
    w("                    hyperparameter search, model selection and threshold")
    w("                    selection. There is no separate fixed validation set:")
    w("                    the 5 CV folds serve that role and give a fold-to-fold")
    w("                    spread for uncertainty.")
    w("Stratified        : YES - StratifiedGroupKFold balances the class ratio")
    w("                    across folds while keeping every group intact.")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    dev_idx, test_idx = next(iter(
        (tr, te) for tr, te in outer.split(np.zeros(len(y)), y, groups=gm)))

    fold_table("subset", [("development", dev_idx), ("TEST (frozen)", test_idx)], y, gm)

    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    dev_y, dev_g = y[dev_idx], gm[dev_idx]
    cv_pairs = list(inner.split(np.zeros(len(dev_idx)), dev_y, groups=dev_g))
    w("")
    w("  development-set cross-validation folds (indices are within development):")
    w("  %-14s %7s %8s %8s %11s %8s" % ("fold", "n_val", "pos", "neg", "prevalence", "groups"))
    w("  " + "-" * 62)
    for i, (tr, va) in enumerate(cv_pairs):
        yy = dev_y[va]
        w("  %-14s %7d %8d %8d %10.2f%% %8d"
          % ("fold %d" % i, len(va), int(yy.sum()), int((yy == 0).sum()),
             100.0 * yy.mean(), len(set(dev_g[va]))))
    w("  %-14s %7d" % ("train per fold", int(np.mean([len(tr) for tr, _ in cv_pairs]))))

    h("3. LEAKAGE CHECKS")
    ok = True
    n_key = len(set(keys[dev_idx]) & set(keys[test_idx]))
    w("parent_inchikey shared between development and TEST : %d" % n_key)
    ok &= n_key == 0
    n_grp = splits.group_overlap(gm, dev_idx, test_idx)
    w("scaffold groups shared between development and TEST : %d" % n_grp)
    ok &= n_grp == 0
    n_smi = len(set(df.parent_smiles.values[dev_idx]) & set(df.parent_smiles.values[test_idx]))
    w("parent SMILES shared between development and TEST   : %d" % n_smi)
    ok &= n_smi == 0
    worst = 0
    for i, (tr, va) in enumerate(cv_pairs):
        o = splits.group_overlap(dev_g, tr, va)
        worst = max(worst, o)
    w("max scaffold-group overlap across the 5 CV folds    : %d" % worst)
    ok &= worst == 0
    w("every compound assigned exactly once                : %s"
      % ("YES" if len(dev_idx) + len(test_idx) == len(df) else "NO"))
    ok &= len(dev_idx) + len(test_idx) == len(df)
    w("")
    w("ALL LEAKAGE CHECKS PASS: %s" % ("YES" if ok else "NO"))


    h("4. ANALOGUE-LEAKAGE AUDIT (structure only - no labels used)")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in df.parent_smiles]
    devfp = [fps[i] for i in dev_idx]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], devfp)) for i in test_idx])
    w("Nearest-neighbour Tanimoto of each TEST compound to the development set:")
    w("  n=%d  mean %.3f  median %.3f  min %.3f  max %.3f"
      % (len(nn), nn.mean(), np.median(nn), nn.min(), nn.max()))
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99):
        w("   NN similarity >= %.2f : %5d (%.1f%%)" % (thr, int((nn >= thr).sum()),
                                                       100.0 * (nn >= thr).mean()))
    w("")
    w("Murcko-scaffold overlap under this split (diagnostic, expected non-zero when")
    w("grouping is Butina - 'all mono-aromatics' is not a meaningful analogue class):")
    sd = set(gm_all[dev_idx]); st = set(gm_all[test_idx])
    w("  scaffolds in development %d, in test %d, shared %d" % (len(sd), len(st), len(sd & st)))
    w("  acyclic compounds: development %d, test %d"
      % (int(pd.Series(gm_all[dev_idx]).str.startswith("ACYCLIC::").sum()),
         int(pd.Series(gm_all[test_idx]).str.startswith("ACYCLIC::").sum())))

    subset = np.array(["development"] * len(df), dtype=object)
    subset[test_idx] = "test"
    fold = np.full(len(df), -1, dtype=int)
    for i, (_, va) in enumerate(cv_pairs):
        fold[dev_idx[va]] = i
    out = pd.DataFrame({"parent_inchikey": keys, "label": y,
                        "scaffold_group": gm_all, "butina_group": gb_all,
                        "split_group": gm, "grouping": args.grouping,
                        "subset": subset, "cv_fold": fold})
    out.to_csv(config.DATA_PROCESSED / "splits.csv", index=False)
    w("")
    w("written: data/processed/splits.csv  (cv_fold = -1 for test rows)")

    with open(config.RESULTS / "checkpoint_3A_split_report.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Report: results/checkpoint_3A_split_report.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
