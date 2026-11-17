#!/usr/bin/env python3
"""
Resolve EU food additives to chemical structures via PubChem.

The Union list gives an E number and a name. It gives NO structures. This step
turns names into structures, and it is the single largest error source in the
whole screening pipeline: a name silently resolved to the wrong compound would
produce a confident, wrong prediction with nothing to flag it.

WHY THIS FILE WAS REWRITTEN. The first version accepted a candidate compound if
PubChem's synonym list for it contained the E number. That rule is not
specific. Supplier and catalogue codes of the form "E-1404" are common, and
once punctuation is stripped they are indistinguishable from a food-additive E
number. Worse, the failure was circular: PubChem's own name search for
"E 1404" returns the catalogue compound, and the validation step then confirmed
it against the very synonym that produced the hit. The result was 29 wrong
structures in 248, including Erlotinib Hydrochloride for E 1404 (oxidised
starch), Spiperone for E 525 (potassium hydroxide) and Androstenone for E 282
(calcium propionate).

THE RULE NOW. A candidate is validated only on one of two grounds:

  TIER 1  TWO INDEPENDENT LINES OF EVIDENCE. PubChem's synonyms for the
          candidate contain BOTH the E number AND the additive's name (or one
          of its listed synonyms). A catalogue code gives the first and never
          the second.

  TIER 2  EXACT FULL-NAME MATCH. The synonyms contain the additive's full name
          as written in the Union list (or its parenthetical synonym). No E
          number is required, because the full name is specific on its own.

A fragment produced by splitting a name on commas or semicolons may be used to
SEARCH, but never as sole proof. "Sodium, potassium and calcium salts of fatty
acids" splits to "Sodium", whose PubChem record matches the element exactly;
under the old rule that passed.

Everything that fails is recorded with a reason and excluded. Nothing is
guessed. Substances of variable composition (polymers, gums, extracts, plural
class names) are labelled in the `substance_class` column and excluded from
scoring downstream by scripts/screen_additives.py - a representative CID for
xanthan gum is not xanthan gum.

Network: PubChem is usually unreachable from restricted environments; run this
on a machine with ordinary internet. Per-additive records AND per-CID synonym
lists are cached, so a re-run resumes, and re-validating with a changed rule
needs no network at all once the synonyms are cached.

Usage:  py -3 scripts/resolve_additive_structures.py
        py -3 scripts/resolve_additive_structures.py --retry-failed
        py -3 scripts/resolve_additive_structures.py --rebuild
Input:  data/external/efsa_additives_union_list.csv
Output: data/external/efsa_additives_resolved.csv
        data/external/efsa_raw/pubchem_cache/*.json
"""
import sys

if sys.version_info < (3, 8):
    sys.stderr.write("\nNeeds Python 3.8+.  py -3 scripts/resolve_additive_structures.py\n\n")
    raise SystemExit(1)

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    sys.stderr.write("\n'requests' is not installed: py -3 -m pip install requests\n\n")
    raise SystemExit(1)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mitotox.additives import classify_substance, normalise_chemical_name

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
IN_CSV = os.path.join("data", "external", "efsa_additives_union_list.csv")
OUT_CSV = os.path.join("data", "external", "efsa_additives_resolved.csv")
CACHE = os.path.join("data", "external", "efsa_raw", "pubchem_cache")
SYN_CACHE = os.path.join(CACHE, "synonyms")
SEARCH_CACHE = os.path.join(CACHE, "name_search")
PROP_CACHE = os.path.join(CACHE, "properties")
DELAY = 0.25          # PubChem asks for <= 5 requests/second
TIMEOUT = 60
RETRIES = 3
MAX_CANDIDATES = 12   # synonym lists to examine per additive

# PubChem renamed SMILES properties; probe which name this server accepts.
SMILES_CANDIDATES = ["IsomericSMILES", "SMILES", "ConnectivitySMILES", "CanonicalSMILES"]

norm = normalise_chemical_name


def e_variants(e):
    """'E 102' -> the spellings PubChem might carry as a synonym."""
    digits = re.sub(r"[^0-9a-z]", "", str(e).lower().replace("e", "", 1))
    return ["E%s" % digits.upper(), "E %s" % digits.upper()]


def name_variants(name):
    """[(variant, kind)] where kind is 'full', 'paren' or 'split'.

    Splitting a chemical name on every comma destroys locants: "Propane-1,
    2-diol (propylene glycol)" becomes "Propane-1" and "2-diol (...)", neither
    of which is a chemical. So a comma is a separator only when it is NOT
    followed by a digit.

    The KIND matters downstream: only 'full' and 'paren' variants are specific
    enough to validate a compound on their own. 'split' variants are search
    terms. "Sodium, potassium and calcium salts of fatty acids" splits to
    "Sodium", which matches the element exactly and means nothing.
    """
    name = str(name).strip()
    out = []
    seen = set()

    def add(s, kind):
        s = s.strip(" .;,")
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append((s, kind))

    # "Propane-1, 2-diol" -> also "Propane-1,2-diol"
    tightened = re.sub(r",\s+(?=\d)", ",", name)
    add(name, "full")
    if tightened != name:
        add(tightened, "full")

    # parenthetical synonyms: "Glyceryl diacetate (diacetin)" -> "diacetin"
    for inner in re.findall(r"\(([^()]+)\)", name):
        if len(inner) > 2 and not re.fullmatch(r"[\d\s,.\-+()LDl]+", inner):
            add(inner, "paren")
    stripped = re.sub(r"\s*\([^()]*\)", "", tightened).strip()
    add(stripped, "full")

    # separators, but never a locant comma
    for part in re.split(r"[;/]|,(?!\s*\d)", stripped):
        add(part, "split")
    return out


class Pug:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "mitotox-additives/1.0 (academic research)"})
        self.smiles_prop = None
        self.calls = 0

    def get(self, url, tolerate_404=True):
        for attempt in range(1, RETRIES + 1):
            try:
                r = self.s.get(url, timeout=TIMEOUT)
                self.calls += 1
                time.sleep(DELAY)
                if r.status_code == 404 and tolerate_404:
                    return None
                if r.status_code in (429, 503) and attempt < RETRIES:
                    time.sleep(3.0 * attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception:
                if attempt == RETRIES:
                    return None
                time.sleep(2.0 * attempt)
        return None

    def pick_smiles_property(self):
        for prop in SMILES_CANDIDATES:
            j = self.get("%s/compound/cid/702/property/%s/JSON" % (PUG, prop))
            if j and "PropertyTable" in j:
                row = j["PropertyTable"]["Properties"][0]
                if prop in row and row[prop]:
                    self.smiles_prop = prop
                    return prop
        return None

    def cids_for_name(self, name):
        """CIDs PubChem returns for a query string, cached on disk.

        Cached so that changing the VALIDATION rule later costs no network at
        all: the candidate search and the synonym lists are both on disk, and
        only the decision is recomputed.
        """
        key = re.sub(r"[^A-Za-z0-9]+", "_", norm(name))[:120] or "_empty"
        path = os.path.join(SEARCH_CACHE, "%s.json" % key)
        if os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))
            except Exception:
                pass
        from urllib.parse import quote
        j = self.get("%s/compound/name/%s/cids/JSON" % (PUG, quote(str(name), safe="")))
        cids = [] if not j else list(j.get("IdentifierList", {}).get("CID", []))[:25]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(cids, fh)
        return cids

    def properties(self, cid):
        path = os.path.join(PROP_CACHE, "%d.json" % cid)
        if os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))
            except Exception:
                pass
        props = "MolecularFormula,MolecularWeight,Title,%s" % self.smiles_prop
        j = self.get("%s/compound/cid/%d/property/%s/JSON" % (PUG, cid, props))
        if not j:
            return None
        row = j["PropertyTable"]["Properties"][0]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(row, fh)
        return row

    def synonyms(self, cid):
        """Synonyms for one CID, cached on disk so re-validation needs no network."""
        path = os.path.join(SYN_CACHE, "%d.json" % cid)
        if os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))
            except Exception:
                pass
        j = self.get("%s/compound/cid/%d/synonyms/JSON" % (PUG, cid))
        syns = []
        if j:
            info = j.get("InformationList", {}).get("Information", [{}])[0]
            syns = info.get("Synonym", [])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(syns, fh)
        return syns


def resolve_one(pug, e_number, name):
    """Returns a record. `validated` is the only field that should be trusted."""
    rec = {"e_number": e_number, "name": name, "pubchem_cid": None,
           "pubchem_title": "", "smiles": "", "molecular_formula": "",
           "molecular_weight": None, "resolution_method": "", "n_candidates": 0,
           "n_candidates_checked": 0, "validated": False,
           "validation_evidence": "", "reason": ""}

    variants = name_variants(name)
    # Search with everything; validate with only what is specific.
    queries = [("e_number", v) for v in e_variants(e_number)]
    queries += [("name", v) for v, _ in variants]

    seen, candidates = set(), []
    for method, q in queries:
        for cid in pug.cids_for_name(q):
            if cid not in seen:
                seen.add(cid)
                candidates.append((method, q, cid))
    rec["n_candidates"] = len(candidates)
    if not candidates:
        rec["reason"] = "no PubChem record for the E number or any name variant"
        return rec

    wanted_e = {norm(v) for v in e_variants(e_number)}
    specific_names = {norm(v) for v, k in variants if k in ("full", "paren")}
    any_names = {norm(v) for v, _ in variants}

    tier1, tier2 = None, None
    for method, q, cid in candidates[:MAX_CANDIDATES]:
        nsyn = {norm(s) for s in pug.synonyms(cid)}
        rec["n_candidates_checked"] += 1
        e_hit = bool(wanted_e & nsyn)
        specific_hit = bool(specific_names & nsyn)
        any_hit = bool(any_names & nsyn)
        if e_hit and any_hit and tier1 is None:
            tier1 = (cid, "PubChem synonyms for CID %d carry BOTH the E number "
                          "and the listed name" % cid, "e_number_and_name")
            break
        if specific_hit and tier2 is None:
            tier2 = (cid, "PubChem synonyms for CID %d exactly match the full "
                          "Union-list name" % cid, "exact_full_name")

    chosen = tier1 or tier2
    if chosen is None:
        rec["reason"] = ("checked %d of %d candidate CID(s); none carried both the "
                         "E number and the name, nor an exact full-name match"
                         % (rec["n_candidates_checked"], len(candidates)))
        return rec

    cid, evidence, method = chosen
    props = pug.properties(cid)
    if not props:
        rec["reason"] = "CID %d validated but PubChem returned no properties" % cid
        return rec
    smi = props.get(pug.smiles_prop, "")
    if not smi:
        rec["reason"] = "CID %d validated but PubChem returned no SMILES" % cid
        return rec

    rec.update(pubchem_cid=cid, pubchem_title=props.get("Title", ""),
               smiles=smi, molecular_formula=props.get("MolecularFormula", ""),
               molecular_weight=props.get("MolecularWeight"),
               resolution_method=method, validated=True,
               validation_evidence=evidence)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=IN_CSV)
    ap.add_argument("--output", default=OUT_CSV)
    ap.add_argument("--limit", type=int, default=0,
                    help="resolve only the first N (for a trial run)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-query additives that previously failed to resolve "
                         "(validated hits stay cached and are not re-fetched)")
    ap.add_argument("--rebuild", action="store_true",
                    help="discard every cached per-additive record and resolve "
                         "all of them again. Required after a change to the "
                         "validation rule. Cached PubChem synonym lists are "
                         "reused, so this is far cheaper than the first run.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.stderr.write("\nNot found: %s\nRun scripts/parse_efsa_additives.py first.\n\n"
                         % args.input)
        return 1
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(SYN_CACHE, exist_ok=True)
    os.makedirs(SEARCH_CACHE, exist_ok=True)
    os.makedirs(PROP_CACHE, exist_ok=True)

    with open(args.input, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]

    pug = Pug()
    prop = pug.pick_smiles_property()
    if not prop:
        sys.stderr.write("\nCannot reach PubChem, or no SMILES property is available.\n"
                         "Run this on a machine with ordinary internet access.\n\n")
        return 1
    print("PubChem reachable; using SMILES property %r" % prop)
    print("validation rule: E number AND name, or an exact full-name match")
    print("resolving %d additives (cached per E number, so re-runs resume)\n" % len(rows))

    out, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        e, name = row["e_number"], row["name"]
        cpath = os.path.join(CACHE, "%s.json" % re.sub(r"[^A-Za-z0-9]", "_", e))
        rec = None
        if os.path.exists(cpath) and not args.rebuild:
            rec = json.load(open(cpath, encoding="utf-8"))
            if args.retry_failed and not rec.get("validated"):
                rec = None
            elif "n_candidates_checked" not in rec:
                rec = None          # written by the superseded validation rule
        if rec is None:
            rec = resolve_one(pug, e, name)
            rec["_retrieved_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(cpath, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2)
        rec["functional_group"] = row.get("functional_group", "")
        rec["substance_class"] = classify_substance(name)
        out.append(rec)
        if i % 20 == 0 or i == len(rows):
            ok = sum(1 for r in out if r["validated"])
            print("  %4d/%d   validated %4d   %5.0fs elapsed"
                  % (i, len(rows), ok, time.time() - t0), flush=True)

    cols = ["e_number", "name", "functional_group", "substance_class",
            "pubchem_cid", "pubchem_title", "smiles", "molecular_formula",
            "molecular_weight", "resolution_method", "n_candidates",
            "n_candidates_checked", "validated", "validation_evidence", "reason"]
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)

    ok = [r for r in out if r["validated"]]
    from collections import Counter
    print("")
    print("=" * 70)
    print("additives in the Union list        : %d" % len(out))
    print("resolved AND validated             : %d" % len(ok))
    print("  by E number AND name             : %d"
          % sum(1 for r in ok if r["resolution_method"] == "e_number_and_name"))
    print("  by exact full-name match         : %d"
          % sum(1 for r in ok if r["resolution_method"] == "exact_full_name"))
    print("not resolved                       : %d" % (len(out) - len(ok)))
    for reason, n in Counter(r["reason"] for r in out if not r["validated"]).most_common():
        print("    %-58s %d" % (reason[:58], n))
    print("")
    print("substance class (all %d, before any resolution):" % len(out))
    for cls, n in Counter(r["substance_class"] for r in out).most_common():
        print("    %-58s %d" % (cls, n))
    print("validated AND a single substance   : %d"
          % sum(1 for r in ok if r["substance_class"] == "single_substance"))
    print("PubChem calls made                 : %d" % pug.calls)
    print("")
    print("written: %s" % args.output)
    print("Next:    python scripts/audit_additive_resolution.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
