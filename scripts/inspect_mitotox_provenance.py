#!/usr/bin/env python3
"""
Checkpoint 1A provenance analysis.

Question: are MitoTox and Tox21 independent sources for the membrane-potential
endpoint, or does MitoTox's F0108 content derive from Tox21?

DESCRIPTIVE ONLY. No labels assigned, no datasets merged or modified.

Usage:  py -3 scripts/inspect_mitotox_provenance.py
Output: results/checkpoint_1A_provenance.txt (and stdout)
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

RAW = os.path.join("data", "external", "mitotox_raw")
OUT = os.path.join("results", "checkpoint_1A_provenance.txt")
_lines = []

TOX21_MARKERS = ("Tox21", "tox21", "qHTS", "quantitative high-throughput",
                 "quantitative high throughput", "NCATS", "10K")
POTENCY = ("IC50", "EC50", "EC20", "EC25", "AC50", "MEC", "UC50", "Ki", "Kd", "ratio")
SIGNIF = ("p <", "p<", "p =", "t-test", "Student")


def w(s=""):
    print(s)
    _lines.append(s)


def load(name):
    with open(os.path.join(RAW, "mitotox_%s.json" % name), encoding="utf-8") as fh:
        return json.load(fh)["results"]


def criteria_kind(c):
    c = str(c or "")
    if not c.strip():
        return "(empty)"
    low = c.lower()
    if any(k.lower() in low for k in POTENCY):
        return "potency-type"
    if any(k.lower() in low for k in SIGNIF):
        return "significance-type"
    return "other"


def dose_kind(d):
    d = str(d or "").strip()
    if not d:
        return "empty"
    if "+-" in d or "±" in d:
        return "value +- SD (no unit)"
    if d.startswith(">") or d.startswith("<"):
        return "relation (> or <)"
    if re.fullmatch(r"[0-9.]+", d):
        return "bare number (no unit)"
    if re.search(r"(M|mol|g/|ppm|%)", d):
        return "number with unit"
    return "other"


def main():
    recs = load("records")
    papers = {p["pap_id"]: p for p in load("papers")}
    f0108 = [r for r in recs if r.get("func") == "F0108"]

    w("=" * 78)
    w("PROVENANCE OF THE F0108 ('membrane potential') RECORDS")
    w("=" * 78)
    w("total F0108 records: %d" % len(f0108))
    w("")

    groups = defaultdict(list)
    for r in f0108:
        groups[r["paper"]].append(r)

    w("%-7s %6s %6s %6s %6s %7s  %-9s %s"
      % ("pap_id", "n", "True", "False", "cmpds", "year", "tox21?", "title"))
    w("-" * 78)
    for pid, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        p = papers.get(pid, {})
        text = " ".join(str(p.get(k) or "") for k in ("title", "abstract", "journal"))
        hits = [m for m in TOX21_MARKERS if m in text]
        title = str(p.get("title") or "")
        w("%-7s %6d %6d %6d %6d %7s  %-9s %s"
          % (pid, len(rs),
             sum(1 for r in rs if r["result"]),
             sum(1 for r in rs if not r["result"]),
             len({r["compound"].split(": ", 1)[0] for r in rs}),
             p.get("pub_year"),
             "YES" if hits else "no",
             title[:64] + ("..." if len(title) > 64 else "")))

    w("")
    w("Full metadata per paper, largest first:")
    for pid, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        p = papers.get(pid, {})
        text = " ".join(str(p.get(k) or "") for k in ("title", "abstract", "journal"))
        hits = [m for m in TOX21_MARKERS if m in text]
        w("")
        w("-" * 78)
        w("pap_id %s   n=%d   year=%s" % (pid, len(rs), p.get("pub_year")))
        w("  title    : %s" % p.get("title"))
        w("  journal  : %s" % p.get("journal"))
        w("  doi/pmid : %s / %s" % (p.get("doi"), p.get("pmid")))
        w("  authors  : %s" % str(p.get("authors") or "")[:150])
        w("  Tox21 markers found in title/abstract/journal: %s" % (hits or "none"))
        for fld in ("method", "animal_model", "species", "criteria"):
            w("  %-9s: %s" % (fld, dict(Counter(str(r.get(fld) or "(empty)")[:40]
                                                for r in rs).most_common(5))))
        w("  criteria kind : %s" % dict(Counter(criteria_kind(r.get("criteria")) for r in rs)))
        w("  dose kind     : %s" % dict(Counter(dose_kind(r.get("dose")) for r in rs)))
        w("  result        : True=%d False=%d"
          % (sum(1 for r in rs if r["result"]), sum(1 for r in rs if not r["result"])))

    w("")
    w("=" * 78)
    w("DOSE-FIELD SEMANTICS: criteria kind vs dose format, ALL %d records" % len(recs))
    w("=" * 78)
    tab = Counter((criteria_kind(r.get("criteria")), dose_kind(r.get("dose"))) for r in recs)
    ks = sorted({a for a, _ in tab})
    ds = sorted({b for _, b in tab})
    w("%-20s" % "criteria kind" + "".join("%22s" % d for d in ds))
    for k in ks:
        w("%-20s" % k + "".join("%22d" % tab.get((k, d), 0) for d in ds))

    w("")
    w("=" * 78)
    w("CROSS-PAPER COMPOUND OVERLAP WITHIN F0108")
    w("=" * 78)
    cp = defaultdict(set)
    for r in f0108:
        cp[r["compound"].split(": ", 1)[0]].add(r["paper"])
    w("compounds by number of source papers: %s"
      % dict(Counter(len(v) for v in cp.values())))
    w("compounds appearing in more than one paper: %d"
      % len([c for c, v in cp.items() if len(v) > 1]))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Report written to %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
