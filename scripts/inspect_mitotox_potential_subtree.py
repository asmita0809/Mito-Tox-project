#!/usr/bin/env python3
"""
Checkpoint 1A supplementary: descriptive breakdown of the two candidate
"membrane potential" scopes in the raw MitoTox records.

DESCRIPTIVE ONLY. No records are labelled positive or negative here; this is a
contingency table of the raw `func_act` and `result` fields, plus the raw
method/model/species/criteria/dose distributions, so that the endpoint scope
and the label rule can be chosen with numbers in hand.

Usage:  py -3 scripts/inspect_mitotox_potential_subtree.py
Output: results/checkpoint_1A_supplementary.txt (and stdout)
"""
import json
import os
import sys
from collections import Counter, defaultdict

RAW = os.path.join("data", "external", "mitotox_raw")
OUT = os.path.join("results", "checkpoint_1A_supplementary.txt")
_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def load(name):
    with open(os.path.join(RAW, "mitotox_%s.json" % name), encoding="utf-8") as fh:
        return json.load(fh)["results"]


def main():
    recs = load("records")
    funcs = load("functions")
    children = defaultdict(list)
    for f in funcs:
        children[f.get("parent_func_id")].append(f["func_id"])

    def subtree(fid):
        out = [fid]
        stack = list(children.get(fid, []))
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(children.get(c, []))
        return set(out)

    scopes = [
        ("F01 subtree (Transmembrane potential, all nodes)", subtree("F01")),
        ("F0108 only ('membrane potential')", {"F0108"}),
    ]

    for label, scope in scopes:
        r = [x for x in recs if x.get("func") in scope]
        w("=" * 72)
        w("%s -> %d records" % (label, len(r)))
        w("  distinct compounds: %d"
          % len({x["compound"].split(": ", 1)[0] for x in r}))
        w("  distinct papers   : %d" % len({x["paper"] for x in r}))
        w("")
        acts = Counter((str(x.get("func_act") or "(empty)"), x.get("result")) for x in r)
        keys = sorted({a for a, _ in acts},
                      key=lambda a: -(acts[(a, True)] + acts[(a, False)]))
        w("  %-22s %8s %8s %8s" % ("func_act", "True", "False", "total"))
        w("  " + "-" * 50)
        for a in keys:
            t, f = acts.get((a, True), 0), acts.get((a, False), 0)
            w("  %-22s %8d %8d %8d" % (a, t, f, t + f))
        w("  %-22s %8d %8d %8d"
          % ("TOTAL",
             sum(1 for x in r if x["result"] is True),
             sum(1 for x in r if x["result"] is False), len(r)))
        w("")
        for fld, n in (("method", 8), ("animal_model", 8), ("species", 6), ("criteria", 8)):
            c = Counter(str(x.get(fld) or "(empty)") for x in r)
            w("  top %s  (%d distinct):" % (fld, len(c)))
            for v, k in c.most_common(n):
                w("    %5d  %s" % (k, v if len(v) <= 56 else v[:53] + "..."))
            w("")
        w("  dose empty: %d/%d   time empty: %d/%d"
          % (sum(1 for x in r if x.get("dose") in (None, "")), len(r),
             sum(1 for x in r if x.get("time") in (None, "")), len(r)))
        w("")

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("Report written to %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
