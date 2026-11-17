"""Gates that decide which food additives a structure-based model may score.

The EU Union list is a list of PERMITTED SUBSTANCES, not a list of compounds.
Three kinds of entry on it cannot be handed to a QSAR trained on discrete
organic structures, and each is excluded here for a stated reason rather than
being quietly scored:

  1. SUBSTANCES OF VARIABLE COMPOSITION - polymers, gums, modified starches,
     extracts, and plural class names ("Pectins", "Sodium carbonates", "Mono-
     and diglycerides of fatty acids"). These have no single structure. A
     database lookup will usually still return *a* CID for them - a
     representative oligomer or one member of the class - and a prediction for
     that CID is not a prediction for the additive. `classify_substance` reads
     the Union-list name, because the name is where the EU records that the
     entry is a class.

  2. INORGANIC CARBON SPECIES - carbonate, bicarbonate, carbonic acid, carbon
     dioxide, carbon monoxide, cyanide, thiocyanate. These contain carbon, so a
     "does it have a carbon atom?" test lets them through, but they are not
     organic compounds and the training set contains nothing like them.
     `is_inorganic_carbon` matches them by substructure AND requires the whole
     molecule to be tiny, so that a carbonic-acid-like fragment inside a real
     organic molecule is not caught. Urea and carbamates deliberately pass:
     they are organic.

  3. CARBON-FREE STRUCTURES are already refused inside the predictor itself
     (`standardize(...)["parent_has_carbon"]`), so they are not repeated here.

Nothing in this module is a model feature and nothing here touches the model.
These are scope gates on the PREDICTION set only.
"""
import re

from rdkit import Chem

SINGLE_SUBSTANCE = "single_substance"
VARIABLE_COMPOSITION = "polymer_or_variable_composition"

# Union-list wording that marks a substance of variable composition. Each group
# is a word the EU itself uses to name a class rather than a compound.
_VARIABLE_WORDS = re.compile(
    r"\b("
    r"alginate|alginates|alginic|"
    r"gum|gums|"
    r"starch|starches|"
    r"cellulose|celluloses|"
    r"pectin|pectins|"
    r"dextrin|dextrins|"
    r"extract|extracts|"
    r"polymer|polymers|copolymer|copolymers|carbomer|"
    r"polyvinyl|polyvinylpyrrolidone|polyvinylpolypyrrolidone|"
    r"polyethylene|polyoxyethylene|polysorbate|polyglycerol|polydextrose|"
    r"wax|waxes|rosin|rosins|resin|resins|"
    r"lecithin|lecithins|carrageenan|"
    r"glycolipid|glycolipids|glyceride|glycerides|"
    r"carotene|carotenes|carotenoid|carotenoids|"
    r"chlorophyll|chlorophylls|chlorophyllin|chlorophyllins|"
    r"glycoside|glycosides|"
    r"hydrolysed|hydrolyzed|modified|oxidised|oxidized|acetylated|phosphated|"
    r"cross-?linked|enzymatically|"
    r"esters? of|salts? of|"
    r"mono-?\s*and\s*di"
    r")\b", re.I)

# Plural class names: the Union list writes a family of salts or isomers in the
# plural ("Sodium carbonates" covers the carbonate and the bicarbonate).
_PLURAL_CLASSES = re.compile(
    r"\b("
    r"carbonates|citrates|phosphates|diphosphates|triphosphates|polyphosphates|"
    r"tartrates|malates|lactates|acetates|ascorbates|benzoates|sorbates|"
    r"gluconates|glutamates|guanylates|inosinates|ribonucleotides|"
    r"sulphates|sulfates|sulphites|sulfites|"
    r"fatty acids|riboflavins|anthocyanins|tocopherols|"
    r"carminic acids|annattos|paprikas"
    r")\b", re.I)


def classify_substance(name):
    """SINGLE_SUBSTANCE or VARIABLE_COMPOSITION, from the Union-list name.

    Deliberately errs toward VARIABLE_COMPOSITION: scoring a representative
    structure as if it were the additive is the worse error, because the number
    looks exactly like a real prediction.
    """
    s = str(name)
    if _VARIABLE_WORDS.search(s) or _PLURAL_CLASSES.search(s):
        return VARIABLE_COMPOSITION
    return SINGLE_SUBSTANCE


# Inorganic carbon species. Each SMARTS is written so that the single-bonded
# oxygens must be free (anionic or -OH), which is what distinguishes carbonic
# acid and the carbonates from a dialkyl carbonate or an ordinary carboxylic
# acid.
_INORGANIC_CARBON_SMARTS = (
    # carbonate / bicarbonate / carbonic acid
    "[CX3](=[OX1])([$([OX1-]),$([OX2H1])])[$([OX1-]),$([OX2H1])]",
    "O=C=O",                 # carbon dioxide
    "[CX1-]#[OX1+]",         # carbon monoxide
    "[CX1-]#[NX1]",          # cyanide
    "[CX2H1]#[NX1]",         # hydrogen cyanide
    "[NX1]#[CX2][$([SX1-]),$([SX2H1])]",   # thiocyanate
)

# A match only counts if the whole molecule is this small. Carbonate is 4 heavy
# atoms, thiocyanate 3, CO2 3. Anything larger containing such a fragment is a
# real organic molecule and must still be scored.
_MAX_HEAVY_ATOMS = 4

_COMPILED = None


def _compiled():
    global _COMPILED
    if _COMPILED is None:
        pats = []
        bad = []
        for s in _INORGANIC_CARBON_SMARTS:
            p = Chem.MolFromSmarts(s)
            if p is None:
                bad.append(s)
            else:
                pats.append((s, p))
        if bad:
            raise ValueError("invalid inorganic-carbon SMARTS: %r" % (bad,))
        _COMPILED = pats
    return _COMPILED


def is_inorganic_carbon(mol):
    """True for carbonate, CO2, cyanide and friends; False for urea, acetate.

    Returns False for None rather than raising - the caller has already handled
    unparseable input.
    """
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
        return False
    for _, p in _compiled():
        if mol.HasSubstructMatch(p):
            return True
    return False


def inorganic_carbon_reason(mol):
    """The matching SMARTS, for the exclusion record. '' if none matches."""
    if mol is None or mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
        return ""
    for s, p in _compiled():
        if mol.HasSubstructMatch(p):
            return s
    return ""


# ------------------------------------------------------------------ name match
# PubChem and the Union list disagree on spelling conventions far more often
# than they disagree on chemistry. These substitutions are applied to BOTH
# sides before comparison so that "Sodium hydrogen sulphite" can corroborate
# "sodium hydrogen sulfite" without loosening the match to a fuzzy one.
_SPELLING = (
    ("sulph", "sulf"),
    ("aluminium", "aluminum"),
    ("ised", "ized"),
    ("isation", "ization"),
    ("oestr", "estr"),
    ("colour", "color"),
    ("centre", "center"),
)


def normalise_chemical_name(s):
    """Lowercase alphanumerics, with spelling conventions unified.

    Still an EXACT-match key: it removes punctuation and spelling variation,
    never words. "Oxidised starch" and "Erlotinib Hydrochloride" do not become
    equal under it.
    """
    t = str(s).lower()
    for a, b in _SPELLING:
        t = t.replace(a, b)
    return re.sub(r"[^a-z0-9]", "", t)
