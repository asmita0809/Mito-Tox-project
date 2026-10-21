"""Structural alerts for POST-HOC interpretation.

These are NOT model features. config.ALERTS_ARE_MODEL_FEATURES is False and the
feature matrices written by scripts/build_features.py never include them. They
exist to answer "what substructures are enriched among the actives, and does the
model's behaviour line up with known mitochondrial-toxicity chemistry".

Design notes
------------
* Each alert holds a TUPLE of SMARTS, never a comma-joined string. SMARTS uses
  commas inside square brackets for atom-level OR, so splitting a pattern on ","
  silently corrupts it. Keeping a tuple makes that impossible.
* `require="all"` means every SMARTS must match somewhere in the molecule
  (used for composite motifs such as "a phenol AND an aromatic nitro").
  `require="any"` means at least one must match.
* The set is small and pre-specified. Alerts were chosen for a stated mechanistic
  reason before looking at enrichment; they are not mined from the labels.
"""
from collections import namedtuple

from rdkit import Chem

Alert = namedtuple("Alert", "key name smarts require rationale")

ALERTS = (
    Alert("nitroaromatic", "Aromatic nitro group",
          ("[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]",), "any",
          "Redox cycling and, on an acidic ring, protonophoric uncoupling."),
    Alert("nitrophenol", "Nitrophenol (phenol + aromatic nitro)",
          ("[OX2H1][c]", "[$([NX3](=O)=O),$([NX3+](=O)[O-])][c]"), "all",
          "Classic protonophoric uncoupler pharmacophore: an acidic, "
          "lipophilic, delocalised anion that shuttles protons across the "
          "inner membrane (the DNP archetype)."),
    Alert("halogenated_phenol", "Halogenated phenol",
          ("[OX2H1][c]", "[F,Cl,Br,I][c]"), "all",
          "Halogens lower phenol pKa and raise logP - the two properties that "
          "make a weak acid an effective uncoupler."),
    Alert("salicylanilide", "Salicylanilide",
          ("[OX2H1]c1ccccc1C(=O)[NX3][c]",), "any",
          "Salicylanilide uncouplers (niclosamide class)."),
    Alert("para_quinone", "para-Quinone",
          ("O=C1C=CC(=O)C=C1",), "any",
          "Redox cycling with the respiratory chain; superoxide generation."),
    Alert("ortho_quinone", "ortho-Quinone",
          ("O=C1C(=O)C=CC=C1",), "any",
          "Redox cycling; also a strong electrophile."),
    Alert("anthraquinone", "Anthraquinone",
          ("O=C1c2ccccc2C(=O)c2ccccc21",), "any",
          "Fused-ring quinone; electron-transport interference."),
    Alert("hindered_phenol", "Sterically hindered phenol (BHT-like)",
          ("[OX2H1]c1c(C(C)(C)C)cccc1",), "any",
          "Lipophilic antioxidant phenols reported to perturb respiration; "
          "the BHT/BHA structural family, directly relevant to food additives."),
    Alert("michael_acceptor", "Michael acceptor (enone)",
          ("[CX3]=[CX3][CX3]=[OX1]",), "any",
          "Thiol-reactive electrophile; glutathione depletion."),
    Alert("epoxide", "Epoxide",
          ("C1OC1",), "any",
          "Strained electrophile; protein and DNA alkylation."),
    Alert("aromatic_azo", "Aromatic azo",
          ("[c][NX2]=[NX2][c]",), "any",
          "Azo dyes; reductive cleavage yields aromatic amines. Common among "
          "colour additives."),
    Alert("biguanide", "Biguanide",
          ("[#7;!R]~[#6;!R](~[#7])~[#7;!R]~[#6;!R](~[#7])~[#7;!R]",), "any",
          "Complex I inhibition (metformin/phenformin class). Bond orders are\n          left unspecified because the biguanide tautomer drawn varies between\n          depositors; ring atoms are excluded so triazines such as melamine do\n          not match."),
    Alert("dithiocarbamate", "Dithiocarbamate",
          ("[NX3][CX3](=[SX1])[SX2,SX1-]",), "any",
          "Metal chelation and thiol reactivity. Flagged specifically because "
          "salt stripping merges the metal complexes with their alkali salts "
          "in this dataset (see flag_metal_removed)."),
)


def compile_alerts(alerts=ALERTS):
    """Compile every SMARTS up front. Raises loudly if any pattern is invalid,
    rather than silently never matching."""
    compiled = {}
    bad = []
    for a in alerts:
        pats = []
        for s in a.smarts:
            p = Chem.MolFromSmarts(s)
            if p is None:
                bad.append((a.key, s))
            else:
                pats.append(p)
        compiled[a.key] = pats
    if bad:
        raise ValueError("invalid SMARTS: %r" % (bad,))
    return compiled


def match_alerts(mol, compiled=None, alerts=ALERTS):
    """Returns {alert_key: bool} for one molecule."""
    if compiled is None:
        compiled = compile_alerts(alerts)
    out = {}
    for a in alerts:
        pats = compiled[a.key]
        if mol is None:
            out[a.key] = False
            continue
        hits = [mol.HasSubstructMatch(p) for p in pats]
        out[a.key] = all(hits) if a.require == "all" else any(hits)
    return out
