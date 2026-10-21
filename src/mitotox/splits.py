"""Grouping and splitting.

The modelling unit is the standardised parent structure (full InChIKey). The
split must additionally keep structural ANALOGUES together, otherwise a model
can score well by memorising a scaffold that appears on both sides of the split.

Two grouping definitions are provided so the choice can be made on evidence:

    murcko   Bemis-Murcko scaffold SMILES. Acyclic molecules have no scaffold
             and are given a singleton group keyed on their own InChIKey.
    butina   Taylor-Butina clustering of Morgan fingerprints at a Tanimoto
             cutoff. Groups near-neighbours regardless of ring system.
"""
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from . import config


def murcko_groups(smiles_list, inchikeys):
    """Bemis-Murcko scaffold SMILES; acyclic molecules become singletons."""
    out = []
    for smi, key in zip(smiles_list, inchikeys):
        m = Chem.MolFromSmiles(smi)
        scaf = ""
        if m is not None:
            try:
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            except Exception:
                scaf = ""
        out.append(scaf if scaf else "ACYCLIC::%s" % key)
    return np.array(out)


def butina_groups(smiles_list, cutoff=0.4):
    """Taylor-Butina clusters. `cutoff` is a DISTANCE, so 0.4 means compounds
    with Tanimoto similarity >= 0.6 tend to land in the same cluster."""
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_NBITS)
    fps = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(m) if m is not None else None)
    n = len(fps)
    dists = []
    for i in range(1, n):
        if fps[i] is None:
            dists.extend([1.0] * i)
            continue
        prev = [f for f in fps[:i]]
        sims = [DataStructs.TanimotoSimilarity(fps[i], f) if f is not None else 0.0
                for f in prev]
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    labels = np.empty(n, dtype=object)
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = "BUTINA::%04d" % ci
    return labels


def group_overlap(groups, idx_a, idx_b):
    """Number of groups appearing in both index sets. Must be 0."""
    return len(set(groups[idx_a]) & set(groups[idx_b]))
