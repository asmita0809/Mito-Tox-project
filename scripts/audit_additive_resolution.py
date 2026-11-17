#!/usr/bin/env python3
"""
Quality-control audit of scripts/resolve_additive_structures.py output.

Nothing here resolves or screens anything. It reads the resolution table and
asks three questions of every VALIDATED row, using only what is already in the
table plus RDKit (no network):

  1. Does PubChem's own title for the chosen CID have anything in common with
     the Union-list name? If a row says "Oxidised starch" and PubChem says
     "Erlotinib Hydrochloride", the resolution is wrong, whatever the
     validation evidence claimed. This check is what caught the superseded
     E-number-only validation rule, and it stays in the pipeline as a standing
     tripwire rather than a one-off investigation.

  2. Is the substance one of variable composition (polymer, gum, extract,
     plural class name)? Taken from src/mitotox/additives.py - the same
     classifier scripts/screen_additives.py uses, so the audit and the screen
     cannot disagree about what is scoreable.

  3. Is the structure an inorganic carbon species, or carbon-free? An organic
     QSAR has nothing to say about talc, calcium carbonate or carbon dioxide.

The title check is a HEURISTIC for human review: a disagreement is a prompt to
look, not a verdict. Spelling and synonym conventions that differ without any
difference in chemistry (sulphite/sulfite, p-hydroxybenzoate/paraben,
Xylitol/Pentitol) are reconciled before flagging, so the disagreements that
remain are worth reading one by one.

Output: results/efsa_resolution_audit.csv, plus a printed summary.

Usage:  python scripts/audit_additive_resolution.py
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rdkit import Chem, RDLogger

from mitotox.additives import (VARIABLE_COMPOSITION, classify_substance,
                               is_inorganic_carbon, normalise_chemical_name)

RDLogger.DisableLog("rdApp.*")

IN_CSV = os.path.join("data", "external", "efsa_additives_resolved.csv")
OUT_CSV = os.path.join("results", "efsa_resolution_audit.csv")

STOP = set("acid acids salt salts of and or the sodium potassium calcium "
           "magnesium ammonium aluminium aluminum iron copper a an".split())

# Pairs that differ by a naming convention, not by chemistry. Applied to both
# sides before comparison.
CONVENTIONS = (
    ("phydroxybenzoate", "paraben"),
    ("hydrogensulfite", "bisulfite"),
    ("metabisulfite", "pyrosulfite"),
    ("carbamide", "urea"),
    ("silicondioxide", "silica"),
    ("tertiarybutyl", "tertbutyl"),
    ("xylitol", "pentitol"),
)


def tokens(s):
    return {t for t in re.split(r"[^a-z0-9]+", str(s).lower())
            if len(t) > 3 and t not in STOP}


def title_agrees(name, title):
    """True if the PubChem title plausibly refers to the same substance."""
    if not title:
        return False
    if tokens(name) & tokens(title):
        return True
    a, b = normalise_chemical_name(name), normalise_chemical_name(title)
    if a and b and (a in b or b in a):
        return True
    for x, y in CONVENTIONS:
        if a.replace(x, y) == b.replace(x, y):
            return True
    return False


def main():
    if not os.path.exists(IN_CSV):
        sys.stderr.write("\nNot found: %s\nRun scripts/resolve_additive_structures.py "
                         "first.\n\n" % IN_CSV)
        return 1
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    rows = list(csv.DictReader(open(IN_CSV, newline="", encoding="utf-8")))
    val = [r for r in rows
           if str(r.get("validated", "")).lower() in ("true", "1")]
    if not val:
        sys.stderr.write("\nNo validated rows in %s.\n\n" % IN_CSV)
        return 1

    out = []
    for r in val:
        name, title = r["name"], r["pubchem_title"]
        mol = Chem.MolFromSmiles(r["smiles"]) if r["smiles"] else None
        agrees = title_agrees(name, title)
        cls = r.get("substance_class") or classify_substance(name)
        variable = cls == VARIABLE_COMPOSITION
        inorg_c = is_inorganic_carbon(mol)
        cfree = (mol is not None
                 and not any(a.GetSymbol() == "C" for a in mol.GetAtoms()))

        flags = []
        if not agrees:
            flags.append("TITLE_DISAGREES_WITH_NAME")
        if variable:
            flags.append("VARIABLE_COMPOSITION")
        if inorg_c:
            flags.append("INORGANIC_CARBON")
        if cfree:
            flags.append("CARBON_FREE")
        if mol is None:
            flags.append("SMILES_DOES_NOT_PARSE")

        out.append({
            "e_number": r["e_number"], "name": name, "substance_class": cls,
            "pubchem_cid": r["pubchem_cid"], "pubchem_title": title,
            "molecular_formula": r["molecular_formula"],
            "resolution_method": r["resolution_method"],
            "validation_evidence": r["validation_evidence"],
            "title_agrees_with_name": agrees,
            "variable_composition": variable,
            "inorganic_carbon": inorg_c,
            "carbon_free": cfree,
            "flags": "|".join(flags),
            "scoreable_single_organic_compound": not flags,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    dis = [r for r in out if not r["title_agrees_with_name"]]
    clean = [r for r in out if r["scoreable_single_organic_compound"]]

    print("=" * 78)
    print("RESOLUTION AUDIT - validated rows only")
    print("=" * 78)
    print("validated rows in the resolution table        : %d" % len(out))
    print("  PubChem title disagrees with the name       : %d" % len(dis))
    print("  substance of variable composition           : %d"
          % sum(1 for r in out if r["variable_composition"]))
    print("  inorganic carbon species                    : %d"
          % sum(1 for r in out if r["inorganic_carbon"]))
    print("  carbon-free structure                       : %d"
          % sum(1 for r in out if r["carbon_free"]))
    print("  scoreable single organic compound           : %d" % len(clean))
    print("")
    if dis:
        print("TITLE DISAGREES WITH THE UNION-LIST NAME - read every one of these")
        print("  %-8s %-40s %-30s" % ("E", "Union-list name", "PubChem title"))
        print("  " + "-" * 82)
        for r in sorted(dis, key=lambda z: z["e_number"]):
            print("  %-8s %-40s %-30s" % (r["e_number"], r["name"][:40],
                                          r["pubchem_title"][:30]))
    else:
        print("No title/name disagreements remain.")
    print("")
    print("written: %s" % OUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
