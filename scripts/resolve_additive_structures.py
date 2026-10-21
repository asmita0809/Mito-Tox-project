#!/usr/bin/env python3
"""
Resolve EU food additives to chemical structures via PubChem.

The Union list gives an E number and a name. It gives NO structures. This step
turns names into structures, and it is the single largest error source in the
whole screening pipeline: a name silently resolved to the wrong compound would
produce a confident, wrong prediction with nothing to flag it.

So resolution is treated as evidence-gathering, not lookup:

  * Every candidate is CHECKED, not assumed. A candidate counts as validated
    only if PubChem's own synonym list for that compound contains the E number
    (strongest evidence), or an exact normalised match to the additive name.
  * The E number is tried FIRST, because "E 102" is far less ambiguous than
    "Tartrazine" and much less ambiguous than "Riboflavins".
  * Names carrying several synonyms ("Azorubine, Carmoisine") are split and each
    tried separately.
  * Anything ambiguous, unvalidated or unresolved is RECORDED with a reason and
    excluded. Nothing is guessed.

Many entries cannot have a single structure by their nature - mixtures, extracts,
polymers, "Modified starches", "Chlorophylls". Those are an expected and reported
outcome, not a failure.

Network: PubChem is usually unreachable from restricted environments; run this on
a machine with ordinary internet. Results are cached per E number, so a re-run
resumes rather than re-fetching.

Usage:  py -3 scripts/resolve_additive_structures.py
Input:  data/external/efsa_additives_union_list.csv
Output: data/external/efsa_additives_resolved.csv
        data/external/efsa_raw/pubchem_cache/*.json
"""
import sys

if sys.version_info < (3, 8):
    sys.stderr.write("\nNeeds Python 3.8+.  py -3 scripts/resolve_additive_structures.py\n\n")
    raise SystemExit(1)

import argparse
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

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
IN_CSV = os.path.join("data", "external", "efsa_additives_union_list.csv")
OUT_CSV = os.path.join("data", "external", "efsa_additives_resolved.csv")
CACHE = os.path.join("data", "external", "efsa_raw", "pubchem_cache")
DELAY = 0.25          # PubChem asks for <= 5 requests/second
TIMEOUT = 60
RETRIES = 3

# PubChem renamed SMILES properties; probe which name this server accepts.
SMILES_CANDIDATES = ["IsomericSMILES", "SMILES", "ConnectivitySMILES", "CanonicalSMILES"]


def norm(s):
    """Loose name key: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def e_variants(e):
    """'E 102' -> the spellings PubChem might carry as a synonym."""
    digits = re.sub(r"[^0-9a-z]", "", str(e).lower().replace("e", "", 1))
    return ["E%s" % digits.upper(), "E %s" % digits.upper()]


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
        from urllib.parse import quote
        j = self.get("%s/compound/name/%s/cids/JSON" % (PUG, quote(str(name), safe="")))
        if not j:
            return []
        return list(j.get("IdentifierList", {}).get("CID", []))[:25]

    def properties(self, cid):
        props = "MolecularFormula,MolecularWeight,Title,%s" % self.smiles_prop
        j = self.get("%s/compound/cid/%d/property/%s/JSON" % (PUG, cid, props))
        if not j:
            return None
        return j["PropertyTable"]["Properties"][0]

    def synonyms(self, cid):
        j = self.get("%s/compound/cid/%d/synonyms/JSON" % (PUG, cid))
        if not j:
            return []
        info = j.get("InformationList", {}).get("Information", [{}])[0]
        return info.get("Synonym", [])


def resolve_one(pug, e_number, name):
    """Returns a record. `validated` is the only field that should be trusted."""
    rec = {"e_number": e_number, "name": name, "pubchem_cid": None,
           "pubchem_title": "", "smiles": "", "molecular_formula": "",
           "molecular_weight": None, "resolution_method": "", "n_candidates": 0,
           "validated": False, "validation_evidence": "", "reason": ""}

    name_parts = [p.strip() for p in re.split(r"[,;]", str(name)) if p.strip()]
    queries = [("e_number", v) for v in e_variants(e_number)]
    queries += [("name", p) for p in name_parts]

    seen, candidates = set(), []
    for method, q in queries:
        for cid in pug.cids_for_name(q):
            if cid not in seen:
                seen.add(cid)
                candidates.append((method, q, cid))
        if candidates and method == "e_number":
            break          # an E-number hit is specific; stop widening
    rec["n_candidates"] = len(candidates)
    if not candidates:
        rec["reason"] = "no PubChem record for the E number or any name variant"
        return rec

    wanted_e = {norm(v) for v in e_variants(e_number)}
    wanted_n = {norm(p) for p in name_parts}
    for method, q, cid in candidates[:8]:
        syns = pug.synonyms(cid)
        nsyn = {norm(s) for s in syns}
        if wanted_e & nsyn:
            evidence = "PubChem synonyms for CID %d contain the E number" % cid
        elif wanted_n & nsyn:
            evidence = "PubChem synonyms for CID %d exactly match the listed name" % cid
        else:
            continue
        props = pug.properties(cid)
        if not props:
            continue
        smi = props.get(pug.smiles_prop, "")
        if not smi:
            continue
        rec.update(pubchem_cid=cid, pubchem_title=props.get("Title", ""),
                   smiles=smi, molecular_formula=props.get("MolecularFormula", ""),
                   molecular_weight=props.get("MolecularWeight"),
                   resolution_method=method, validated=True,
                   validation_evidence=evidence)
        return rec

    rec["reason"] = ("found %d candidate CID(s) but none carried the E number or "
                     "an exact name match in its PubChem synonyms" % len(candidates))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=IN_CSV)
    ap.add_argument("--output", default=OUT_CSV)
    ap.add_argument("--limit", type=int, default=0, help="resolve only the first N (for a trial run)")
    args = ap.parse_args()

    import csv
    if not os.path.exists(args.input):
        sys.stderr.write("\nNot found: %s\nRun scripts/parse_efsa_additives.py first.\n\n"
                         % args.input)
        return 1
    os.makedirs(CACHE, exist_ok=True)

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
    print("resolving %d additives (cached per E number, so re-runs resume)\n" % len(rows))

    out, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        e, name = row["e_number"], row["name"]
        cpath = os.path.join(CACHE, "%s.json" % re.sub(r"[^A-Za-z0-9]", "_", e))
        if os.path.exists(cpath):
            rec = json.load(open(cpath, encoding="utf-8"))
        else:
            rec = resolve_one(pug, e, name)
            rec["_retrieved_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(cpath, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2)
        rec["functional_group"] = row.get("functional_group", "")
        out.append(rec)
        if i % 20 == 0 or i == len(rows):
            ok = sum(1 for r in out if r["validated"])
            print("  %4d/%d   validated %4d   %5.0fs elapsed"
                  % (i, len(rows), ok, time.time() - t0), flush=True)

    cols = ["e_number", "name", "functional_group", "pubchem_cid", "pubchem_title",
            "smiles", "molecular_formula", "molecular_weight", "resolution_method",
            "n_candidates", "validated", "validation_evidence", "reason"]
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)

    ok = [r for r in out if r["validated"]]
    print("")
    print("=" * 70)
    print("additives in the Union list        : %d" % len(out))
    print("resolved AND validated             : %d" % len(ok))
    print("  via the E number                 : %d"
          % sum(1 for r in ok if r["resolution_method"] == "e_number"))
    print("  via an exact name match          : %d"
          % sum(1 for r in ok if r["resolution_method"] == "name"))
    print("not resolved                       : %d" % (len(out) - len(ok)))
    from collections import Counter
    for reason, n in Counter(r["reason"] for r in out if not r["validated"]).most_common():
        print("    %-58s %d" % (reason[:58], n))
    print("PubChem calls made                 : %d" % pug.calls)
    print("")
    print("written: %s" % args.output)
    print("Next:    py -3 scripts/screen_additives.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
