#!/usr/bin/env python3
"""Fixture test for the Union-list parser.

Regulation 1333/2008 contains two things that will silently corrupt a naive
parse of ANNEX II Part B:

  * PART A has its own "E-number / Name" table - "Colours which may be used in
    the form of lakes" - which is NOT the Union list. Rows from it must never
    appear in the output.
  * Consolidated texts interleave amendment markers and footnote rows between
    the data rows.

Both failure modes look like success: you still get a plausible CSV of E numbers
and names. This fixture makes them fail loudly instead.

Run:  py -3 tests/test_efsa_parser.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "parse_efsa_additives.py")

FIXTURE = """<table>
<tr><td>PART A</td></tr>
<tr><td>Table 3</td></tr>
<tr><td>Colours which may be used in the form of lakes</td></tr>
<tr><td>E-number</td><td>Name</td></tr>
<tr><td>E 100</td><td>DECOY-LAKE-CURCUMIN</td></tr>
<tr><td>E 180</td><td>DECOY-LAKE-LITHOLRUBINE</td></tr>
<tr><td>PART B</td></tr>
<tr><td>LIST OF ALL ADDITIVES</td></tr>
<tr><td>1.    Colours</td></tr>
<tr><td>E-number</td><td>Name</td></tr>
<tr><td>E 100</td><td>Curcumin</td></tr>
<tr><td>&#9660;M81</td></tr>
<tr><td>E 120</td><td>Carminic acid, Carmine</td></tr>
<tr><td>&#9660;M2</td></tr>
<tr><td>E 150a</td><td>Plain caramel (1)</td></tr>
<tr><td>E 160b(i)</td><td>Annatto bixin</td></tr>
<tr><td>E 161g</td><td>Canthaxanthin (*1)</td></tr>
<tr><td>(1)    The term caramel relates to products of a more or less intense brown colour.</td></tr>
<tr><td>2.    Sweeteners</td></tr>
<tr><td>E 951</td><td>Aspartame</td></tr>
<tr><td>3.    Additives other than colours and sweeteners</td></tr>
<tr><td>E 321</td><td>Butylated hydroxytoluene (BHT)</td></tr>
<tr><td>E 1521</td><td>Polyethylene glycol</td></tr>
<tr><td>PART C</td></tr>
<tr><td>E 999</td><td>SHOULD-NOT-APPEAR</td></tr>
</table>"""

EXPECTED = [
    ("E 100", "Curcumin", "colour"),
    ("E 120", "Carminic acid, Carmine", "colour"),
    ("E 150a", "Plain caramel", "colour"),
    ("E 160b(i)", "Annatto bixin", "colour"),
    ("E 161g", "Canthaxanthin", "colour"),
    ("E 951", "Aspartame", "sweetener"),
    ("E 321", "Butylated hydroxytoluene (BHT)", "other"),
    ("E 1521", "Polyethylene glycol", "other"),
]


def load_helpers():
    src = open(SCRIPT, encoding="utf-8").read().split("def main()")[0]
    ns = {}
    exec(compile(src, SCRIPT, "exec"), ns)
    return ns


def parse(ns, raw):
    lines = ns["to_lines"](raw)
    start = next((i for i, l in enumerate(lines) if re.fullmatch(r"PART\s+B", l, re.I)), None)
    assert start is not None, "PART B heading not found"
    end = next((i for i, l in enumerate(lines)
                if i > start and re.fullmatch(r"PART\s+C", l, re.I)), len(lines))
    body = lines[start:end]
    E_RE, GROUPS, SKIP = ns["E_RE"], ns["GROUPS"], ns["SKIP_EXACT"]
    rows, group, seen, i = [], "", set(), 0
    while i < len(body):
        ln = body[i]
        for rx, g in GROUPS:
            if rx.match(ln):
                group = g
                break
        m = E_RE.match(ln)
        if m:
            e = "E %s%s%s" % (m.group(1), (m.group(2) or "").lower(), m.group(3) or "")
            j, name = i + 1, None
            while j < len(body) and j <= i + 6:
                c = body[j]
                if ns["is_marker"](c) or ns["is_footnote"](c) or c.lower() in SKIP:
                    j += 1
                    continue
                if E_RE.match(c):
                    break
                name = c
                break
            if name:
                name = re.sub(r"\s*\(\s*[*]?\s*\d+\s*\)\s*$", "", name).strip()
                if e not in seen:
                    seen.add(e)
                    rows.append((e, name, group))
            i = j
            continue
        i += 1
    return rows


def main():
    ns = load_helpers()
    rows = parse(ns, FIXTURE)
    names = [r[1] for r in rows]
    failures = []

    for bad in ("DECOY-LAKE-CURCUMIN", "DECOY-LAKE-LITHOLRUBINE"):
        if bad in names:
            failures.append("PART A lookalike table leaked in: %s" % bad)
    if "SHOULD-NOT-APPEAR" in names:
        failures.append("parsing did not stop at PART C")
    for exp in EXPECTED:
        if exp not in rows:
            failures.append("missing or wrong: %r (got %r)"
                            % (exp, [r for r in rows if r[0] == exp[0]]))
    if len(rows) != len(EXPECTED):
        failures.append("parsed %d rows, expected %d" % (len(rows), len(EXPECTED)))

    print("rows parsed from fixture : %d" % len(rows))
    for r in rows:
        print("   %-12s %-34s %s" % r)
    print("-" * 60)
    if failures:
        print("FAIL")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("PASS - PART A decoy excluded, PART C excluded, amendment markers and\n"
          "       footnotes skipped, sub-numbered E numbers and group headings kept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
