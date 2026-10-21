#!/usr/bin/env python3
"""
Checkpoint 1B input: quantify the MitoTox F0108 subset that is INDEPENDENT of
Tox21, i.e. F0108 records excluding paper 163 (Xia et al. 2018, Tox21 10K).

DESCRIPTIVE ONLY. Nothing is merged, labelled or modified.

Usage:  py -3 scripts/inspect_mitotox_independent_subset.py
Output: results/checkpoint_1B_mitotox_independent_subset.txt (and stdout)
"""
import json
import os
import sys
from collections import Counter, defaultdict

RAW = os.path.join("data", "external", "mitotox_raw")
OUT = os.path.join("results", "checkpoint_1B_mitotox_independent_subset.txt")
TOX21_PAPER = "163"   # Xia et al. 2018, Environ Health Perspect, Tox21 10K
_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def load(name):
    with open(os.path.join(RAW, "mitotox_%s.json" % name), encoding="utf-8") as fh:
        return json.load(fh)["results"]


def main():
    recs = load("records")
    papers = {p["pap_id"]: p for p in load("papers")}
    comps = {c["compound_ID"]: c for c in load("compounds")}

    f0108 = [r for r in recs if r.get("func") == "F0108"]
    indep = [r for r in f0108 if r["paper"] != TOX21_PAPER]
    tox21 = [r for r in f0108 if r["paper"] == TOX21_PAPER]

    cid_of = lambda r: r["compound"].split(": ", 1)[0]
    ci, ct = {cid_of(r) for r in indep}, {cid_of(r) for r in tox21}

    w("=" * 74)
    w("MitoTox F0108 minus paper %s (Xia 2018, Tox21 10K)" % TOX21_PAPER)
    w("=" * 74)
    w("records                     : %d   (of %d F0108 total)" % (len(indep), len(f0108)))
    w("unique compounds            : %d" % len(ci))
    w("result True (positive)      : %d" % sum(1 for r in indep if r["result"]))
    w("result False (negative)     : %d" % sum(1 for r in indep if not r["result"]))
    w("source papers               : %d" % len({r["paper"] for r in indep}))
    w("")
    w("compounds also present in paper %s : %d" % (TOX21_PAPER, len(ci & ct)))
    w("compounds NOT in paper %s          : %d" % (TOX21_PAPER, len(ci - ct)))
    w("")
    w("per paper:")
    g = defaultdict(list)
    for r in indep:
        g[r["paper"]].append(r)
    w("  %-7s %5s %5s %5s %6s %6s  %s" % ("pap_id", "n", "True", "False", "cmpds", "year", "journal"))
    w("  " + "-" * 70)
    for pid, rs in sorted(g.items(), key=lambda kv: -len(kv[1])):
        p = papers.get(pid, {})
        w("  %-7s %5d %5d %5d %6d %6s  %s"
          % (pid, len(rs), sum(1 for r in rs if r["result"]),
             sum(1 for r in rs if not r["result"]),
             len({cid_of(r) for r in rs}), p.get("pub_year"),
             str(p.get("journal"))[:34]))
    w("")
    for fld in ("method", "animal_model", "species", "criteria", "func_act"):
        w("%s:" % fld)
        for v, k in Counter(str(r.get(fld) or "(empty)") for r in indep).most_common(12):
            w("   %4d  %s" % (k, v if len(v) <= 62 else v[:59] + "..."))
        w("")
    n_cid = sum(1 for c in ci if comps.get(c, {}).get("cid") not in (None, ""))
    n_mf = sum(1 for c in ci if comps.get(c, {}).get("molecular_formula") not in (None, ""))
    w("PubChem CID present      : %d / %d (%.1f%%)" % (n_cid, len(ci), 100.0 * n_cid / len(ci)))
    w("molecular_formula present: %d / %d (%.1f%%)" % (n_mf, len(ci), 100.0 * n_mf / len(ci)))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Report written to %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
