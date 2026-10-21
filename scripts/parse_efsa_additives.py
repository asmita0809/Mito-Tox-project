#!/usr/bin/env python3
"""
Extract the Union list of food additives from Regulation (EC) No 1333/2008.

The list wanted is ANNEX II, PART B, "LIST OF ALL ADDITIVES", which is divided
into three numbered groups: 1. Colours, 2. Sweeteners, 3. Additives other than
colours and sweeteners.

Three things in the document will corrupt a naive parse, and each is handled
explicitly:

  * PART A contains its OWN "E-number / Name" table - "Colours which may be used
    in the form of lakes". It is not the Union list. Parsing starts only after
    the PART B heading, so those rows cannot leak in.
  * Consolidated texts carry amendment markers (an inverted triangle followed by
    M2, M81, ...) interleaved between rows. They are skipped, not treated as names.
  * Footnote rows, the repeated "E-number"/"Name" column headers and the group
    headings must not be read as additive names.

Every E number found is reported with the group it came from, and the counts are
printed so the parse can be sanity-checked against the document.

Usage:  py -3 scripts/parse_efsa_additives.py
Input:  data/external/efsa_raw/reg_1333_2008_consolidated.html
Output: data/external/efsa_additives_union_list.csv
"""
import sys

if sys.version_info < (3, 8):
    sys.stderr.write("\nNeeds Python 3.8+.\n\n")
    raise SystemExit(1)

import argparse
import csv
import html
import json
import os
import re
from collections import Counter

IN_HTML = os.path.join("data", "external", "efsa_raw", "reg_1333_2008_consolidated.html")
OUT_CSV = os.path.join("data", "external", "efsa_additives_union_list.csv")

# "E 100", "E 150a", "E 160b(i)", "E 1521"
E_RE = re.compile(r"^E\s*(\d{3,4})\s*([a-z])?\s*(\((?:i|ii|iii|iv|v|vi)\))?$", re.I)
GROUPS = [
    (re.compile(r"^1\.?\s+Colours\b", re.I), "colour"),
    (re.compile(r"^2\.?\s+Sweeteners\b", re.I), "sweetener"),
    (re.compile(r"^3\.?\s+Additives other than colours and sweeteners\b", re.I), "other"),
]
SKIP_EXACT = {"e-number", "name", "e number"}


def to_lines(raw_html):
    """HTML -> one logical line per table cell / paragraph, entities resolved."""
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", raw_html)
    t = re.sub(r"(?i)<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)</(td|th|tr|p|div|li|h[1-6])\s*>", "\n", t)
    t = re.sub(r"(?i)<(td|th|tr|p|div|li|h[1-6])[^>]*>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = t.replace(" ", " ").replace("’", "'").replace("′", "'")
    out = []
    for line in t.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out


def is_marker(line):
    """Amendment marker such as the inverted-triangle M2 / M81, or a bare arrow."""
    s = line.strip("▼►◄←→ ").strip()
    if not s:
        return True
    return bool(re.fullmatch(r"M\d{1,3}", s, re.I)) or bool(re.fullmatch(r"[▼►]+", line.strip()))


def is_footnote(line):
    return bool(re.match(r"^\(\s*[*]?\s*\d+\s*\)", line)) or line.startswith("OJ L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=IN_HTML)
    ap.add_argument("--output", default=OUT_CSV)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.stderr.write("\nNot found: %s\nRun scripts/download_efsa_additives.py first.\n\n"
                         % args.input)
        return 1
    raw = open(args.input, "rb").read().decode("utf-8", "replace")
    lines = to_lines(raw)

    # --- slice ANNEX II PART B .. PART C, ignoring PART A's lookalike table
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and re.fullmatch(r"PART\s+B", ln, re.I):
            start = i
        elif start is not None and re.fullmatch(r"PART\s+C", ln, re.I):
            end = i
            break
    if start is None:
        sys.stderr.write("\nCould not find the PART B heading. Is this the right "
                         "document, saved in full?\n\n")
        return 1
    if end is None:
        end = len(lines)
    body = lines[start:end]
    print("document lines              : %d" % len(lines))
    print("PART B slice                : lines %d..%d (%d lines)" % (start, end, len(body)))

    # --- walk the slice
    rows, group, seen = [], "", {}
    i = 0
    while i < len(body):
        ln = body[i]
        for rx, gname in GROUPS:
            if rx.match(ln):
                group = gname
                break
        m = E_RE.match(ln)
        if m:
            num, suffix, roman = m.group(1), (m.group(2) or "").lower(), (m.group(3) or "")
            e_number = "E %s%s%s" % (num, suffix, roman)
            # the name is the next line that is not a marker, header or footnote
            j, name = i + 1, None
            while j < len(body) and j <= i + 6:
                cand = body[j]
                if is_marker(cand) or is_footnote(cand) or cand.lower() in SKIP_EXACT:
                    j += 1
                    continue
                if E_RE.match(cand):
                    break
                name = cand
                break
            if name:
                name = re.sub(r"\s*\(\s*[*]?\s*\d+\s*\)\s*$", "", name).strip()
                if e_number in seen:
                    seen[e_number] += 1
                else:
                    seen[e_number] = 1
                    rows.append({"e_number": e_number, "name": name,
                                 "functional_group": group})
            i = j
            continue
        i += 1

    if not rows:
        sys.stderr.write("\nPART B was found but no E numbers were parsed from it.\n\n")
        return 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["e_number", "name", "functional_group"])
        wtr.writeheader()
        wtr.writerows(rows)

    print("")
    print("additives parsed            : %d" % len(rows))
    for g, n in Counter(r["functional_group"] for r in rows).most_common():
        print("   %-24s %d" % (g or "(no group heading seen)", n))
    dupes = {k: v for k, v in seen.items() if v > 1}
    print("duplicate E numbers skipped : %d %s"
          % (len(dupes), sorted(dupes)[:6] if dupes else ""))
    print("")
    print("first 8 and last 4:")
    for r in rows[:8] + rows[-4:]:
        print("   %-12s %-46s %s" % (r["e_number"], r["name"][:46], r["functional_group"]))

    # sanity checks against known landmarks of the Union list
    byid = {r["e_number"]: r["name"] for r in rows}
    print("")
    print("sanity checks:")
    ok = True
    for e, frag in [("E 100", "Curcumin"), ("E 102", "Tartrazine"),
                    ("E 951", "Aspartame"), ("E 621", "Glutamat"),
                    ("E 321", "Butylated"), ("E 1521", "Polyethylene glycol")]:
        got = byid.get(e, "")
        good = frag.lower() in got.lower()
        ok &= good
        print("   %-8s %-34s %s" % (e, got[:34], "OK" if good else "<<< expected %r" % frag))
    print("   %-8s %-34s %s" % ("count", "%d additives" % len(rows),
                                "OK" if 250 <= len(rows) <= 450 else "<<< outside 250-450"))
    print("")
    print("written: %s" % args.output)
    print("Next:    py -3 scripts/resolve_additive_structures.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
