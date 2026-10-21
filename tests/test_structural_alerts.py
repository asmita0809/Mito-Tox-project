#!/usr/bin/env python3
"""Positive- and negative-control tests for the structural alerts.

A SMARTS pattern that never matches anything looks exactly like a substructure
that is absent from the dataset. The only way to tell them apart is to check
each pattern against a molecule that is definitionally an example of it.

Run:  py -3 tests/test_structural_alerts.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rdkit import Chem, RDLogger
from mitotox.interpretation import ALERTS, compile_alerts, match_alerts

RDLogger.DisableLog("rdApp.*")

# One molecule that MUST trigger each alert, named so the intent is obvious.
POSITIVE_CONTROLS = {
    "nitroaromatic":      ("nitrobenzene",        "[O-][N+](=O)c1ccccc1"),
    "nitrophenol":        ("2,4-dinitrophenol",   "Oc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]"),
    "halogenated_phenol": ("pentachlorophenol",   "Oc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl"),
    "salicylanilide":     ("niclosamide",         "Oc1ccc(Cl)cc1C(=O)Nc1ccc([N+](=O)[O-])cc1Cl"),
    "para_quinone":       ("1,4-benzoquinone",    "O=C1C=CC(=O)C=C1"),
    "ortho_quinone":      ("1,2-benzoquinone",    "O=C1C=CC=CC1=O"),
    "anthraquinone":      ("anthraquinone",       "O=C1c2ccccc2C(=O)c2ccccc21"),
    "hindered_phenol":    ("BHT",                 "Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"),
    "michael_acceptor":   ("acrolein",            "C=CC=O"),
    "epoxide":            ("ethylene oxide",      "C1CO1"),
    "aromatic_azo":       ("azobenzene",          "c1ccc(N=Nc2ccccc2)cc1"),
    "biguanide":          ("metformin",           "CN(C)C(=N)N=C(N)N"),
    "dithiocarbamate":    ("dimethyldithiocarbamic acid", "CN(C)C(=S)S"),
}

# Molecules that must NOT trigger the named alerts.
NEGATIVE_CONTROLS = [
    ("benzene", "c1ccccc1",
     ["nitroaromatic", "nitrophenol", "halogenated_phenol", "para_quinone",
      "epoxide", "michael_acceptor", "biguanide", "dithiocarbamate"]),
    ("phenol", "Oc1ccccc1", ["nitrophenol", "halogenated_phenol", "hindered_phenol"]),
    ("glucose", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
     ["epoxide", "michael_acceptor", "nitroaromatic", "aromatic_azo"]),
    ("acetic acid", "CC(=O)O", ["michael_acceptor", "para_quinone"]),
    ("melamine", "Nc1nc(N)nc(N)n1", ["biguanide"]),
    ("guanidine", "NC(N)=N", ["biguanide"]),
    ("urea", "NC(N)=O", ["biguanide"]),
]


def main():
    compiled = compile_alerts()
    keys = [a.key for a in ALERTS]
    failures = []

    missing = set(keys) - set(POSITIVE_CONTROLS)
    if missing:
        failures.append("no positive control defined for: %s" % sorted(missing))

    for key, (name, smi) in POSITIVE_CONTROLS.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append("control SMILES will not parse: %s (%s)" % (name, key))
            continue
        hits = match_alerts(mol, compiled)
        if not hits.get(key):
            failures.append("POSITIVE CONTROL FAILED: %s (%s) does not match alert '%s'"
                            % (name, smi, key))

    for name, smi, must_not in NEGATIVE_CONTROLS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append("negative control will not parse: %s" % name)
            continue
        hits = match_alerts(mol, compiled)
        for key in must_not:
            if hits.get(key):
                failures.append("NEGATIVE CONTROL FAILED: %s wrongly matches '%s'"
                                % (name, key))

    print("alerts checked      : %d" % len(keys))
    print("positive controls   : %d" % len(POSITIVE_CONTROLS))
    print("negative assertions : %d" % sum(len(m) for _, _, m in NEGATIVE_CONTROLS))
    print("-" * 60)
    if failures:
        print("FAIL")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("PASS - every alert matches its reference molecule and none fires on "
          "the negative controls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
