#!/usr/bin/env python3
"""Retain each TEST compound's nearest-development-neighbour Tanimoto similarity.

STRUCTURE ONLY - no labels are used to compute anything here. Saved so that the
final results report can test whether predictive performance depends on chemical
proximity to the training data.

Usage:  py -3 scripts/compute_test_neighbours.py
Output: data/processed/test_nn_similarity.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from mitotox import config

RDLogger.DisableLog("rdApp.*")

sp = pd.read_csv(config.DATA_PROCESSED / "splits.csv")
tr = pd.read_csv(config.TRAINING_CSV)[["parent_inchikey", "parent_smiles"]]
d = sp.merge(tr, on="parent_inchikey", how="left")

gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.parent_smiles]

dev = d.index[d.subset == "development"].to_numpy()
te = d.index[d.subset == "test"].to_numpy()
devfp = [fps[i] for i in dev]
devkeys = d.parent_inchikey.values[dev]

rows = []
for i in te:
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], devfp)
    j = int(np.argmax(sims))
    rows.append({"parent_inchikey": d.parent_inchikey.values[i],
                 "label": int(d.label.values[i]),
                 "nn_tanimoto": float(sims[j]),
                 "nn_development_inchikey": devkeys[j]})
out = pd.DataFrame(rows)
out.to_csv(config.DATA_PROCESSED / "test_nn_similarity.csv", index=False)
print("test compounds: %d" % len(out))
print("mean %.3f  median %.3f  min %.3f  max %.3f"
      % (out.nn_tanimoto.mean(), out.nn_tanimoto.median(),
         out.nn_tanimoto.min(), out.nn_tanimoto.max()))
for t in (0.6, 0.7, 0.8, 0.9, 0.99):
    print("  >= %.2f : %5d (%.1f%%)" % (t, int((out.nn_tanimoto >= t).sum()),
                                        100.0 * (out.nn_tanimoto >= t).mean()))
print("written: data/processed/test_nn_similarity.csv")
