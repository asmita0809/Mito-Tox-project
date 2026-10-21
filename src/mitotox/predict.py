"""The black-box predictor.

Takes SMILES in, returns a mitochondrial-toxicity risk score out. The caller
does not need to know the model is XGBoost on Morgan fingerprints plus six
descriptors - but everything that qualifies the number is returned alongside it.

Three safeguards are built in, because a probability on its own is easy to
over-read:

  * APPLICABILITY DOMAIN. Every prediction carries the Tanimoto similarity to
    its nearest development-set neighbour, and a band. The bands are not
    invented: they come from measured frozen-test performance per band
    (PR-AUC 0.59 below 0.4, 0.64 for 0.4-0.5, 0.74-0.82 above 0.5).

  * TRAINING-SET MEMBERSHIP. If a query compound's standardised InChIKey is in
    the training set, that is reported along with its experimental label. A
    model reproducing a compound it was fitted on is not a prediction, and the
    output says so rather than letting it pass as one.

  * STRUCTURAL REFUSALS. Inorganic parents, unparseable structures and
    standardisation failures get no score at all. They return a reason.
"""
import json

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from . import config, features
from .standardize import standardize

RDLogger.DisableLog("rdApp.*")

# Bands derived from results/tables/test_performance_by_proximity.csv
DOMAIN_BANDS = [
    (0.50, "within domain",
     "nearest training neighbour Tanimoto >= 0.50; measured test PR-AUC "
     "0.74-0.82 in this range"),
    (0.40, "borderline",
     "nearest training neighbour Tanimoto 0.40-0.50; measured test PR-AUC "
     "0.64 in this range - treat the score as indicative only"),
    (0.00, "outside domain",
     "nearest training neighbour Tanimoto < 0.40; measured test PR-AUC 0.59 "
     "in this range - the score is weakly informative at best"),
]


class MitoToxPredictor:
    """Load once, predict many."""

    def __init__(self, model_path=None, reference_path=None):
        model_path = model_path or (config.ROOT / "models" / "final_model.joblib")
        reference_path = reference_path or (config.ROOT / "models" /
                                            "applicability_reference.npz")
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["primary_threshold"])
        self.meta = {k: v for k, v in bundle.items() if k != "pipeline"}

        ref = np.load(reference_path, allow_pickle=True)
        self._ref_bits = np.unpackbits(ref["packed"], axis=1)[:, :config.MORGAN_NBITS]
        self._ref_popcount = self._ref_bits.sum(axis=1).astype(np.float32)
        self.training_labels = dict(zip(ref["inchikey"].tolist(),
                                        ref["label"].tolist()))
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS,
            includeChirality=config.MORGAN_USE_CHIRALITY)

    # -------------------------------------------------- applicability
    def _nearest(self, bits):
        """Max Tanimoto against the development set, computed on the bit matrix."""
        inter = self._ref_bits @ bits.astype(np.float32)
        union = self._ref_popcount + bits.sum() - inter
        sims = np.divide(inter, union, out=np.zeros_like(inter),
                         where=union > 0)
        j = int(np.argmax(sims))
        return float(sims[j]), j

    @staticmethod
    def _band(sim):
        for lo, name, note in DOMAIN_BANDS:
            if sim >= lo:
                return name, note
        return DOMAIN_BANDS[-1][1], DOMAIN_BANDS[-1][2]

    # -------------------------------------------------- single compound
    def predict_one(self, smiles, name=None):
        row = {"input_smiles": smiles, "name": name, "standardized_smiles": "",
               "inchikey": "", "probability": None, "predicted_class": None,
               "threshold": self.threshold, "applicability": "",
               "applicability_note": "", "nn_tanimoto": None,
               "in_training_set": False, "training_set_label": None,
               "scored": False, "warnings": ""}
        warn = []

        std = standardize(smiles)
        if not std["ok"]:
            row["warnings"] = "NOT SCORED: " + std["reason"]
            row["standardized_smiles"] = std["parent_smiles"]
            return row
        row["standardized_smiles"] = std["parent_smiles"]
        row["inchikey"] = std["inchikey"]

        if not std["parent_has_carbon"]:
            row["warnings"] = ("NOT SCORED: inorganic parent structure. The model "
                               "was trained on organic compounds only; carbon-free "
                               "parents were excluded from the training set.")
            return row
        if std["metal_removed"]:
            warn.append("a metal was removed by salt stripping - for a metal "
                        "complex whose metal is the active species (e.g. a "
                        "dithiocarbamate), the parent is not the same chemical "
                        "entity")
        if std["is_mixture"]:
            warn.append("input had two or more organic components; only the "
                        "largest was scored")

        mol = Chem.MolFromSmiles(std["parent_smiles"])
        bits = features.morgan_fp(mol)
        desc = features.descriptors(mol)
        X = np.hstack([bits.astype(np.float32),
                       np.array([desc[n] for n, _ in config.DESCRIPTORS],
                                dtype=np.float32)]).reshape(1, -1)
        p = float(self.pipeline.predict_proba(X)[0, 1])
        row["probability"] = round(p, 4)
        row["predicted_class"] = int(p >= self.threshold)
        row["scored"] = True

        sim, _ = self._nearest(bits)
        band, note = self._band(sim)
        row["nn_tanimoto"] = round(sim, 4)
        row["applicability"] = band
        row["applicability_note"] = note

        if std["inchikey"] in self.training_labels:
            row["in_training_set"] = True
            row["training_set_label"] = int(self.training_labels[std["inchikey"]])
            warn.append("this compound IS in the training set - the score is a "
                        "fitted value, not a prediction; use the experimental "
                        "label instead")
        row["warnings"] = " | ".join(warn)
        return row

    # -------------------------------------------------- many
    def predict(self, smiles_list, names=None):
        names = names if names is not None else [None] * len(smiles_list)
        return pd.DataFrame([self.predict_one(s, n)
                             for s, n in zip(smiles_list, names)])
