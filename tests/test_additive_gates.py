#!/usr/bin/env python3
"""Controls for the additive scope gates in src/mitotox/additives.py.

The gates exist because a structure-based model was about to be handed
substances that have no single structure, and inorganic carbon species that
contain a carbon atom but are not organic compounds. These are the positive and
negative controls for both gates, plus the name-normalisation key that the
resolver's corroboration rule depends on.

Run:  python tests/test_additive_gates.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rdkit import Chem, RDLogger

from mitotox.additives import (SINGLE_SUBSTANCE, VARIABLE_COMPOSITION,
                               classify_substance, is_inorganic_carbon,
                               normalise_chemical_name)

RDLogger.DisableLog("rdApp.*")

failures = []


def check(label, got, want):
    ok = got == want
    print("  %-62s %s" % (label, "ok" if ok else "FAIL  got %r want %r"
                          % (got, want)))
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- 1. classes
# Real Union-list names. The left column is verbatim from Annex II Part B.
VARIABLE = [
    "Xanthan gum",
    "Locust bean gum",
    "Gum arabic (acacia gum)",
    "Pectins",
    "Oxidised starch",
    "Acetylated distarch phosphate",
    "Hydroxy propyl starch",
    "Methyl cellulose",
    "Enzymatically hydrolysed carboxy methyl cellulose",
    "Propane-1, 2-diol alginate",
    "Alginic acid",
    "Polyvinylpyrrolidone",
    "Anionic methacrylate copolymer",
    "Polyethylene glycol",
    "Paprika extract, capsanthin, capsorubin",
    "Copper complexes of chlorophylls, chlorophyllins",
    "Mono-and diglycerides of fatty acids",
    "Fatty acid esters of ascorbic acid",
    "Sodium, potassium and calcium salts of fatty acids",
    "Glycerol esters of wood rosins",
    "Enzymatically produced steviol glycosides",
    "Carotenes",
    "Sodium carbonates",
    "Calcium citrates",
    "Triphosphates",
    "Fatty acids",
    "Glycolipids",
]
SINGLE = [
    "Tartrazine",
    "Butylated hydroxyanisole (BHA)",
    "Butylated hydroxytoluene (BHT)",
    "Tertiary-butyl hydroquinone (TBHQ)",
    "Methyl p-hydroxybenzoate",
    "Sodium benzoate",
    "Potassium sorbate",
    "Aspartame",
    "Acesulfame K",
    "Sucralose",
    "Xylitol",
    "Carbamide",
    "Propane-1, 2-diol (propylene glycol)",
    "Glyceryl diacetate (diacetin)",
    "Dimethyl dicarbonate",
    "Natamycin",
    "Nisin",
    "Sodium nitrite",
    "Calcium propionate",
    "Potassium hydroxide",
]

print("substances of variable composition (must all be excluded)")
for n in VARIABLE:
    check(n[:60], classify_substance(n), VARIABLE_COMPOSITION)

print("\nsingle substances (must all remain scoreable)")
for n in SINGLE:
    check(n[:60], classify_substance(n), SINGLE_SUBSTANCE)

# ------------------------------------------------------- 2. inorganic carbon
print("\ninorganic carbon species (must all be excluded)")
INORGANIC = [
    ("carbonate dianion", "[O-]C([O-])=O"),
    ("bicarbonate", "OC([O-])=O"),
    ("carbonic acid", "OC(O)=O"),
    ("carbon dioxide", "O=C=O"),
    ("carbon monoxide", "[C-]#[O+]"),
    ("cyanide", "[C-]#N"),
    ("hydrogen cyanide", "C#N"),
    ("thiocyanate", "[S-]C#N"),
]
for label, smi in INORGANIC:
    m = Chem.MolFromSmiles(smi)
    assert m is not None, "test SMILES does not parse: %s" % smi
    check(label, is_inorganic_carbon(m), True)

print("\norganic compounds (must all pass the inorganic-carbon gate)")
ORGANIC = [
    ("urea (E 927b carbamide)", "NC(N)=O"),
    ("acetate", "CC([O-])=O"),
    ("acetic acid", "CC(O)=O"),
    ("dimethyl carbonate", "COC(=O)OC"),
    ("formic acid", "OC=O"),
    ("oxalic acid", "OC(=O)C(O)=O"),
    ("glycine", "NCC(O)=O"),
    ("citric acid", "OC(=O)CC(O)(CC(O)=O)C(O)=O"),
    ("BHA", "CC(C)(C)c1cc(OC)ccc1O"),
    ("tartrazine fragment (azo dye core)", "c1ccc(cc1)/N=N/c1ccccc1"),
    ("guanidine", "NC(N)=N"),
    ("thiourea", "NC(N)=S"),
]
for label, smi in ORGANIC:
    m = Chem.MolFromSmiles(smi)
    assert m is not None, "test SMILES does not parse: %s" % smi
    check(label, is_inorganic_carbon(m), False)

# ---------------------------------------------------- 3. name normalisation
print("\nname normalisation unifies spelling but never equates different names")
check("sulphite == sulfite",
      normalise_chemical_name("Sodium hydrogen sulphite")
      == normalise_chemical_name("sodium hydrogen sulfite"), True)
check("aluminium == aluminum",
      normalise_chemical_name("Aluminium sulphate")
      == normalise_chemical_name("aluminum sulfate"), True)
check("punctuation and case ignored",
      normalise_chemical_name("Propane-1, 2-diol")
      == normalise_chemical_name("propane-1,2-diol"), True)
check("oxidised starch != erlotinib hydrochloride",
      normalise_chemical_name("Oxidised starch")
      == normalise_chemical_name("Erlotinib Hydrochloride"), False)
check("potassium hydroxide != spiperone",
      normalise_chemical_name("Potassium hydroxide")
      == normalise_chemical_name("Spiperone"), False)
check("locust bean gum != citronellyl propionate",
      normalise_chemical_name("Locust bean gum")
      == normalise_chemical_name("Citronellyl propionate"), False)
check("dimethyl dicarbonate != dimethyl carbonate",
      normalise_chemical_name("Dimethyl dicarbonate")
      == normalise_chemical_name("Dimethyl carbonate"), False)

print("")
if failures:
    print("FAILED %d check(s):" % len(failures))
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("all additive-gate controls passed")
