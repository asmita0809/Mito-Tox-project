#!/usr/bin/env python3
"""
Acquire the EU Union list of authorised food additives.

Source of record: ANNEX II, Part B of Regulation (EC) No 1333/2008, consolidated,
on EUR-Lex. EU legislation is reusable under Commission Decision 2011/833/EU.

EUR-Lex sits behind AWS WAF bot protection: a scripted GET receives an HTTP 202
JavaScript challenge page (about 2 KB) rather than the document. This script
therefore does two things:

  1. tries the automated fetch, and VALIDATES the result - it refuses to report
     success on a challenge page;
  2. accepts a page you saved from your own browser, which passes the WAF,
     checksums it and writes the same manifest.

Why not EFSA OpenFoodTox: licensed CC BY-ND, which does not sit comfortably with
publishing a derived screening table, and it spans every EFSA remit rather than
the additive list.

Usage (Windows):
    py -3 scripts/download_efsa_additives.py

    # after saving the page from a browser into data/external/efsa_raw/
    py -3 scripts/download_efsa_additives.py --from-file "data/external/efsa_raw/reg_1333_2008_consolidated.html"
"""
import sys

if sys.version_info < (3, 8):
    sys.stderr.write("\nNeeds Python 3.8+. On Windows: py -3 "
                     "scripts/download_efsa_additives.py\n\n")
    raise SystemExit(1)

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

BASE = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:%s"
DEFAULT_CELEX = "02008R1333-20260818"
OUTDIR = os.path.join("data", "external", "efsa_raw")
FNAME = "reg_1333_2008_consolidated.html"

# A real copy of the consolidated regulation contains all of these. A WAF
# challenge page, a cookie wall or an error page contains none of them.
MARKERS = ["ANNEX II", "E 100", "E 1521", "Curcumin"]
MIN_BYTES = 500_000


def validate(raw):
    """Returns (ok, problems). Never guesses."""
    problems = []
    if len(raw) < MIN_BYTES:
        problems.append("only %d bytes; the consolidated text is several MB "
                        "(a WAF challenge page is about 2 KB)" % len(raw))
    text = raw.decode("utf-8", "replace")
    missing = [m for m in MARKERS if m not in text]
    if missing:
        problems.append("missing expected content: %s" % ", ".join(repr(m) for m in missing))
    if "awswaf" in text.lower() or "gokuProps" in text:
        problems.append("this is an AWS WAF bot-protection challenge page")
    return (not problems), problems


def manual_instructions(url, path):
    return """
EUR-Lex refused the scripted request. Fetch it with your browser instead -
it takes about 30 seconds and the result is the same document:

  1. Open this URL in Chrome or Edge:

     %s

  2. Wait for the full text to load (it is long - scroll to confirm you can
     see "ANNEX II" and entries like "E 100  Curcumin").

  3. Press Ctrl+S and save as "Webpage, HTML Only" into the project folder as:

     %s

  4. Re-run, pointing at the saved file:

     py -3 scripts/download_efsa_additives.py --from-file "%s"

The checksum and provenance manifest are written either way, so the
reproducibility record is identical.
""" % (url, path.replace("/", os.sep), path.replace("\\", "/"))


def write_manifest(raw, path, url, celex, how):
    m = {
        "tool": "scripts/download_efsa_additives.py",
        "source": "EUR-Lex consolidated Regulation (EC) No 1333/2008, ANNEX II Part B",
        "celex": celex,
        "url": url,
        "acquired_how": how,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filename": os.path.basename(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "licence": ("EU legislation; reuse permitted under Commission Decision "
                    "2011/833/EU. Not redistributed in this repository."),
        "note": ("Consolidated texts are amended over time; the CELEX id pins "
                 "the version these results were produced from."),
    }
    with open(os.path.join(os.path.dirname(path), "download_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celex", default=DEFAULT_CELEX)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--from-file", default=None,
                    help="validate and record a page you saved from a browser")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    url = BASE % args.celex
    path = args.from_file or os.path.join(args.outdir, FNAME)

    if args.from_file:
        if not os.path.exists(args.from_file):
            sys.stderr.write("\nNo such file: %s\n\n" % args.from_file)
            return 1
        raw = open(args.from_file, "rb").read()
        how = "saved manually from a browser (EUR-Lex WAF blocks scripted GETs)"
    else:
        try:
            import requests
        except ImportError:
            sys.stderr.write("\n'requests' is not installed: "
                             "py -3 -m pip install requests\n\n")
            return 1
        print("Regulation (EC) No 1333/2008, consolidated text")
        print("CELEX : %s" % args.celex)
        print("URL   : %s\n" % url)
        try:
            r = requests.get(url, timeout=180, headers={
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0 Safari/537.36"),
                "Accept": "text/html,application/xhtml+xml"})
            raw = r.content
            print("HTTP %s, %d bytes" % (r.status_code, len(raw)))
        except Exception as exc:
            sys.stderr.write("\nrequest failed: %s\n" % exc)
            sys.stderr.write(manual_instructions(url, os.path.join(args.outdir, FNAME)))
            return 1
        how = "automated fetch"

    ok, problems = validate(raw)
    print("")
    for m in MARKERS:
        print("  contains %-12s : %s"
              % (repr(m), "yes" if m in raw.decode("utf-8", "replace") else "NO"))
    print("  size                 : %d bytes" % len(raw))

    if not ok:
        print("")
        print("STATUS: FAILED - what came back is not the regulation:")
        for p in problems:
            print("  - %s" % p)
        sys.stderr.write(manual_instructions(url, os.path.join(args.outdir, FNAME)))
        return 1

    if not args.from_file:
        with open(path, "wb") as fh:
            fh.write(raw)
    m = write_manifest(raw, path, url, args.celex, how)
    print("")
    print("saved  : %s" % path)
    print("sha256 : %s" % m["sha256"])
    print("")
    print("STATUS: OK - now run  py -3 scripts/parse_efsa_additives.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
