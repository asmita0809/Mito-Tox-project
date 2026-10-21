#!/usr/bin/env python3
"""
Download raw Tox21 mitochondrial membrane potential data from PubChem BioAssay.

Checkpoint 1B.6. RAW ACQUISITION ONLY.

Assays:
    AID 720637  MMP summary assay (combines 720635 MMP + 720634 viability)
    AID 720634  cell viability counterscreen (intracellular ATP, same wells)

This script does NOT merge the two assays, does NOT create a label column, does
NOT filter, deduplicate, resolve inconclusive records, compute descriptors, or
train anything. It writes every response byte-for-byte and records a manifest
with source URLs, retrieval timestamps, assay IDs, filenames and SHA-256 sums.

For each assay it retrieves four views, because PubChem serves the same data
through different endpoints with different completeness limits and we do not
yet know which one returns the whole assay:

    description/JSON  column definitions and protocol text (needed to read the
                      data table at all - it defines what each TID column means)
    summary/JSON      assay-level counts as PubChem reports them
    concise/CSV       one row per substance: SID, CID, activity outcome, value
    datatable CSV     the full result table, via two different endpoints:
                        (a) PUG REST  /assay/aid/<aid>/CSV
                        (b) pcget.cgi classic download
                      PUG REST caps assay retrieval, so (b) is kept as the
                      authoritative copy if the two differ. BOTH are saved
                      unmodified; nothing is chosen or discarded here.

Usage (Windows PowerShell):
    py -3 -m pip install requests
    py -3 scripts/download_tox21.py
Usage (macOS / Linux):
    python3 scripts/download_tox21.py

Optional:
    --outdir   output directory (default: data/external/tox21_raw)
    --aids     comma-separated AIDs (default: 720637,720634)

Requires: requests

Exit codes:
    0  every file downloaded and read back cleanly
    1  something failed or looked wrong (nothing is silently skipped)

NOTE: pure ASCII, no Python-3.10-only syntax, so a wrong interpreter gives a
readable message rather than a SyntaxError and printing never fails on a
cp1252 Windows console.
"""

import sys

if sys.version_info < (3, 8):
    sys.stderr.write(
        "\n"
        "This script needs Python 3.8 or newer; you are running "
        + sys.version.split()[0] + ".\n"
        "On Windows use the launcher:  py -3 scripts/download_tox21.py\n"
        "Check what you have with:     py -0\n\n"
    )
    raise SystemExit(1)

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.stderr.write(
        "\nThe 'requests' package is not installed for this interpreter.\n"
        "Install it with:  py -3 -m pip install requests   (Windows)\n"
        "              or: python3 -m pip install requests\n\n"
    )
    raise SystemExit(1)


PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PCGET = "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi"

DEFAULT_AIDS = ["720637", "720634"]

REQUEST_TIMEOUT = 600         # the full data table can be slow to build
DELAY_BETWEEN_CALLS = 1.0     # PubChem asks for <= 5 requests/second
MAX_RETRIES = 4
USER_AGENT = (
    "mitotox-additives/0.1 (academic research; PubChem PUG REST client; "
    "contact via project repository)"
)


def utc_now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def views_for(aid):
    """The four raw views to fetch for one assay. Order is fixed."""
    return [
        ("description", "json",
         "%s/assay/aid/%s/description/JSON" % (PUG, aid)),
        ("summary", "json",
         "%s/assay/aid/%s/summary/JSON" % (PUG, aid)),
        ("concise", "csv",
         "%s/assay/aid/%s/concise/CSV" % (PUG, aid)),
        ("datatable_pugrest", "csv",
         "%s/assay/aid/%s/CSV" % (PUG, aid)),
        ("datatable_pcget", "csv",
         "%s?query=download&record_type=datatable&actvty=all"
         "&response_type=save&aid=%s" % (PCGET, aid)),
    ]


def fetch(session, url):
    """GET with backoff. Returns the response. Raises the last error if all fail."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            # PubChem returns 503 when its queue is busy; that is worth retrying.
            if resp.status_code == 503 and attempt < MAX_RETRIES:
                raise RuntimeError("HTTP 503 (PubChem busy)")
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            backoff = 3.0 * attempt
            print("      attempt %d failed (%s); retrying in %.0fs"
                  % (attempt, exc, backoff))
            time.sleep(backoff)
    raise last_exc


def looks_like_waiting(body_head):
    """PubChem can answer a big request with a 'Waiting' listkey envelope."""
    return '"Waiting"' in body_head or "ListKey" in body_head


def main():
    ap = argparse.ArgumentParser(
        description="Download raw Tox21 MMP + viability assay data from PubChem "
                    "(acquisition only, no processing).")
    ap.add_argument("--outdir", default="data/external/tox21_raw",
                    help="output directory (default: data/external/tox21_raw)")
    ap.add_argument("--aids", default=",".join(DEFAULT_AIDS),
                    help="comma-separated AIDs (default: %s)" % ",".join(DEFAULT_AIDS))
    args = ap.parse_args()

    aids = [a.strip() for a in args.aids.split(",") if a.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    started = utc_now_iso()
    print("Tox21 raw download (PubChem BioAssay)")
    print("AIDs            : %s" % ", ".join(aids))
    print("Output directory: %s" % outdir.resolve())
    print("Interpreter     : Python %s" % sys.version.split()[0])
    print("")

    entries = []
    failures = []
    warnings = []

    for aid in aids:
        print("AID %s" % aid)
        for view, ext, url in views_for(aid):
            fname = "aid%s_%s.%s" % (aid, view, ext)
            path = outdir / fname
            print("  %-20s -> %s" % (view, fname))
            try:
                resp = fetch(session, url)
            except Exception as exc:
                print("      !! FAILED - %s" % exc)
                failures.append("aid %s / %s: %s" % (aid, view, exc))
                time.sleep(DELAY_BETWEEN_CALLS)
                continue

            path.write_bytes(resp.content)

            head = resp.content[:2000].decode("utf-8", "replace")
            if looks_like_waiting(head):
                warnings.append(
                    "aid %s / %s: PubChem returned a 'Waiting' envelope rather "
                    "than data - re-run this view later" % (aid, view))

            entries.append({
                "aid": aid,
                "view": view,
                "url": url,
                "filename": fname,
                "http_status": resp.status_code,
                "content_type": resp.headers.get("Content-Type", ""),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "retrieved_utc": utc_now_iso(),
            })
            print("      %d bytes, HTTP %d" % (path.stat().st_size, resp.status_code))
            time.sleep(DELAY_BETWEEN_CALLS)
        print("")

    # ---- integrity checks only: files exist, are non-empty, and parse ----
    print("Verifying written files...")
    problems = []
    for e in entries:
        p = outdir / e["filename"]
        if not p.exists() or p.stat().st_size == 0:
            problems.append("%s: missing or empty" % e["filename"])
            continue
        if e["filename"].endswith(".json"):
            try:
                with open(str(p), "r", encoding="utf-8") as fh:
                    json.load(fh)
            except Exception as exc:
                problems.append("%s: does not parse as JSON (%s)" % (e["filename"], exc))
        else:
            try:
                with open(str(p), "r", encoding="utf-8", errors="replace") as fh:
                    n = sum(1 for _ in fh)
                e["line_count"] = n
                if n < 2:
                    problems.append("%s: only %d line(s) - no data rows"
                                    % (e["filename"], n))
            except Exception as exc:
                problems.append("%s: could not read back (%s)" % (e["filename"], exc))

    manifest = {
        "tool": "scripts/download_tox21.py",
        "purpose": "raw acquisition only - no merging, labelling, filtering, "
                   "deduplication or feature generation",
        "source": "PubChem BioAssay (NCBI)",
        "aids": aids,
        "python_version": sys.version.split()[0],
        "requests_version": requests.__version__,
        "started_utc": started,
        "finished_utc": utc_now_iso(),
        "note": "Files are the response bodies as received (HTTP transfer "
                "encoding removed by the client; content otherwise unmodified).",
        "files": entries,
        "failures": failures,
        "warnings": warnings,
        "verification_problems": problems,
        "status": "ok" if (not failures and not problems) else "incomplete",
    }
    with open(str(outdir / "download_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print("")
    print("=" * 78)
    print("%-34s %8s %10s %12s" % ("file", "HTTP", "bytes", "lines"))
    print("-" * 78)
    for e in entries:
        print("%-34s %8d %10d %12s"
              % (e["filename"], e["http_status"], e["bytes"],
                 e.get("line_count", "-")))
    print("=" * 78)

    if failures:
        print("")
        print("FAILED:")
        for f in failures:
            print("  - %s" % f)
    if warnings:
        print("")
        print("WARNINGS:")
        for x in warnings:
            print("  - %s" % x)
    if problems:
        print("")
        print("Verification problems:")
        for p in problems:
            print("  - %s" % p)

    if failures or problems:
        print("")
        print("STATUS: INCOMPLETE - send download_manifest.json for diagnosis.")
        return 1

    print("")
    print("STATUS: OK - all views downloaded and read back cleanly.")
    print("Nothing was merged, labelled, filtered or transformed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
