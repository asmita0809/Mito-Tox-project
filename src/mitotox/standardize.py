"""The ONE structure-standardisation function.

This module is the contract between the training data and every later
prediction. `standardize()` here produced the `parent_smiles` column of
data/processed/tox21_mmp_training.csv, and predict.py calls the same function.
If the two ever diverged, every prediction would be invalid in a way that no
metric would reveal - so tests/test_standardization.py re-derives the stored
parents from the stored depositor SMILES and asserts an exact match.

Pipeline (RDKit rdMolStandardize, in order):
    1. MolFromSmiles + sanitise
    2. Normalizer()                      functional-group normalisation
    3. MetalDisconnector()               break covalent metal-ligand bonds
    4. LargestFragmentChooser(preferOrganic=True)
    5. Uncharger()                       neutralise counter-ion-derived charges
                                         (permanent charges are preserved)
    6. canonical isomeric SMILES + InChIKey

Stereochemistry is retained throughout: isomericSmiles=True, and nothing here
removes a stereocentre.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_normalizer = rdMolStandardize.Normalizer()
_metal = rdMolStandardize.MetalDisconnector()
_largest = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()

METALS = frozenset(
    "Li Na K Rb Cs Be Mg Ca Sr Ba Al Fe Zn Cu Mn Ni Co Cr Cd Hg Pb Sn "
    "Ag Au Pt Pd Ti V Se As Sb Bi Zr Mo W Gd La Ce".split())


def standardize(smiles):
    """Standardise one SMILES string.

    Returns a dict that is always complete, never raises:
        ok                  bool, False if the structure is unusable
        reason              why, when ok is False
        parent_smiles       canonical isomeric SMILES of the neutral parent
        inchikey            full InChIKey of that parent
        n_components        components in the input (salts, mixtures)
        metal_removed       a metal in the input is absent from the parent
        is_mixture          2+ organic components of >=5 heavy atoms were present,
                            so a real chemical entity was discarded by step 4
        parent_has_carbon   False means an inorganic parent: out of scope
        parent_net_charge   residual charge after uncharging (permanent charges)
    """
    out = {"ok": False, "reason": "", "parent_smiles": "", "inchikey": "",
           "n_components": 0, "metal_removed": False, "is_mixture": False,
           "parent_has_carbon": False, "parent_net_charge": None,
           "parent_heavy_atoms": 0}
    if not isinstance(smiles, str) or not smiles.strip():
        out["reason"] = "empty input"
        return out

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        out["reason"] = "SMILES does not parse or fails sanitisation"
        return out

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    out["n_components"] = len(frags)
    out["is_mixture"] = sum(
        1 for f in frags
        if any(a.GetSymbol() == "C" for a in f.GetAtoms())
        and f.GetNumHeavyAtoms() >= 5) >= 2
    in_metals = {a.GetSymbol() for a in mol.GetAtoms()} & METALS

    try:
        m = _normalizer.normalize(mol)
        m = _metal.Disconnect(m)
        m = _largest.choose(m)
        m = _uncharger.uncharge(m)
        parent = Chem.MolToSmiles(m, isomericSmiles=True)
        key = Chem.MolToInchiKey(m) or ""
    except Exception as exc:
        out["reason"] = "standardisation error: %s" % exc
        return out

    if not parent:
        out["reason"] = "standardisation produced an empty structure"
        return out
    recheck = Chem.MolFromSmiles(parent)
    if recheck is None:
        out["parent_smiles"] = parent
        out["reason"] = ("standardised parent does not re-parse "
                         "(seen with metallocenes, which reduce to 'c1cccc1')")
        return out
    if not key:
        out["parent_smiles"] = parent
        out["reason"] = "InChIKey generation failed"
        return out

    syms = {a.GetSymbol() for a in recheck.GetAtoms()}
    out.update(ok=True, parent_smiles=parent, inchikey=key,
               metal_removed=bool(in_metals) and not (syms & METALS),
               parent_has_carbon="C" in syms,
               parent_net_charge=Chem.GetFormalCharge(recheck),
               parent_heavy_atoms=recheck.GetNumHeavyAtoms())
    return out
