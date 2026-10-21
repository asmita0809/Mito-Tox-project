#!/usr/bin/env python3
"""Precompute the applicability-domain reference: development-set fingerprints.

predict.py needs the development compounds to measure how far a query sits from
the training data. Packing the bits keeps the file ~1 MB instead of ~10 MB.

Usage:  py -3 scripts/build_applicability_reference.py
Output: models/applicability_reference.npz
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox import config

train = pd.read_csv(config.TRAINING_CSV)
splits = pd.read_csv(config.DATA_PROCESSED / "splits.csv")
mor = np.load(config.DATA_PROCESSED / "features_morgan.npz", allow_pickle=True)
keys = train.parent_inchikey.values
assert list(mor["parent_inchikey"]) == list(keys)

sp = splits.set_index("parent_inchikey").loc[keys]
is_dev = (sp.subset == "development").values
X = mor["X"][is_dev].astype(np.uint8)

os.makedirs(config.ROOT / "models", exist_ok=True)
out = config.ROOT / "models" / "applicability_reference.npz"
np.savez_compressed(
    out,
    packed=np.packbits(X, axis=1),
    inchikey=keys[is_dev],
    label=train.label.values[is_dev].astype(np.int8),
    note=np.array(["development-set Morgan fingerprints (radius 2, 2048 bits, "
                   "chirality on) used only for applicability-domain distance"]))
print("development compounds : %d" % is_dev.sum())
print("written               : models/applicability_reference.npz  %.2f MB"
      % (os.path.getsize(out) / 1e6))
