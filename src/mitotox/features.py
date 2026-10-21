"""Molecular feature generation.

Three blocks, kept separate on disk so a model can use any combination:
    morgan       binary circular fingerprint  (primary representation)
    maccs        167 predefined structural keys (comparison representation)
    descriptors  six standard physicochemical descriptors

Structural alerts live in interpretation.py and are NOT features. See config.py.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator

from . import config


def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None


_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=config.MORGAN_RADIUS,
    fpSize=config.MORGAN_NBITS,
    includeChirality=config.MORGAN_USE_CHIRALITY,
)


def morgan_fp(mol):
    """Binary Morgan fingerprint as uint8. Returns None for an unusable mol."""
    if mol is None:
        return None
    fp = _morgan_gen.GetFingerprintAsNumPy(mol)
    return fp.astype(np.uint8)


def maccs_fp(mol):
    """167-bit MACCS key vector as uint8 (bit 0 is unused by RDKit and kept as 0)."""
    if mol is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(config.MACCS_NBITS, dtype=np.uint8)
    for b in fp.GetOnBits():
        arr[b] = 1
    return arr


_DESC_FN = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
}


def descriptors(mol):
    """The six brief-specified descriptors. NaN for any that raises."""
    out = {}
    for name, _ in config.DESCRIPTORS:
        if mol is None:
            out[name] = float("nan")
            continue
        try:
            out[name] = float(_DESC_FN[name](mol))
        except Exception:
            out[name] = float("nan")
    return out
