#!/usr/bin/env python3
"""
Download raw MitoTox data via the official MitoTox RESTful API.

Stage 1A of the mitochondrial-toxicity project. RAW ACQUISITION ONLY.

This script does NOT filter, label, deduplicate, join, interpret or otherwise
modify any record. It follows API pagination to completion, writes every page
byte-for-byte as received, and records a manifest with checksums so the
download is auditable and reproducible.

API root: https://www.mitotox.org/api/   (documented at https://www.mitotox.org/api/)

Endpoints downloaded:
    compounds/list   compound records
    records/list     experimental observation records
    functions/list   mitochondrial function / endpoint ontology
    targets/list     mitochondrial target ontology
    papers/list      literature references

Usage:
    python scripts/download_mitotox.py
    python scripts/download_mitotox.py --outdir data/external/mitotox_raw
    python scripts/download_mitotox.py --base-url http://127.0.0.1:8000/api  # self-test

Exit codes:
    0  all endpoints downloaded and verified
    1  one or more endpoints failed (nothing is silently skipped)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "https://www.mitotox.org/api"

# Endpoint path -> local basename. Order is fixed so runs are comparable.
ENDPOINTS: dict[str, str] = {
    "compounds/list": "compounds",
    "records/list": "records",
    "functions/list": "functions",
    "targets/list": "targets",
    "papers/list": "papers",
}

REQUEST_TIMEOUT = 60          # seconds
DELAY_BETWEEN_PAGES = 0.5     # be polite to a small academic server
MAX_RETRIES = 4
MAX_PAGES = 10_000            # runaway guard; far above any plausible page count
USER_AGENT = (
    "mitotox-additives/0.1 (academic research; paginated API client; "
    "contact via project repository)"
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_with_retries(session: requests.Session, url: str) -> requests.Response:
    """GET with exponential backoff. Raises the last exception if all attempts fail."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - we re-raise after the loop
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            backoff = 2.0 ** (attempt - 1)
            print(f"      attempt {attempt} failed ({exc}); retrying in {backoff:.0f}s")
            time.sleep(backoff)
    assert last_exc is not None
    raise last_exc


def download_endpoint(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    name: str,
    outdir: Path,
) -> dict[str, Any]:
    """Follow pagination to completion. Returns a manifest entry for this endpoint."""
    pages_dir = outdir / "pages" / name
    pages_dir.mkdir(parents=True, exist_ok=True)

    url: str | None = f"{base_url.rstrip('/')}/{endpoint}"
    first_url = url
    declared_count: int | None = None
    results: list[Any] = []
    page_files: list[str] = []
    page_no = 0

    print(f"  {name}: {url}")

    while url is not None:
        page_no += 1
        if page_no > MAX_PAGES:
            raise RuntimeError(f"{name}: exceeded MAX_PAGES={MAX_PAGES}; aborting")

        resp = get_with_retries(session, url)

        # Write the page exactly as received, before any parsing.
        page_path = pages_dir / f"{name}_page_{page_no:04d}.json"
        page_path.write_bytes(resp.content)
        page_files.append(str(page_path.relative_to(outdir)))

        payload = json.loads(resp.content.decode("utf-8"))

        if not isinstance(payload, dict) or "results" not in payload:
            raise RuntimeError(
                f"{name}: page {page_no} is not a paginated envelope "
                f"(top-level keys: {list(payload)[:10] if isinstance(payload, dict) else type(payload)})"
            )

        if declared_count is None:
            declared_count = payload.get("count")

        results.extend(payload["results"])
        next_url = payload.get("next")

        print(
            f"      page {page_no:>4}  +{len(payload['results']):>4} rows  "
            f"total {len(results):>6}"
            + ("" if next_url else "  [last page]")
        )

        url = next_url
        if url is not None:
            time.sleep(DELAY_BETWEEN_PAGES)

    # Concatenation of the pages' `results` arrays. Individual records are
    # untouched; only the pagination envelope is dropped.
    combined_path = outdir / f"mitotox_{name}.json"
    combined = {
        "_source_endpoint": first_url,
        "_retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "_declared_count": declared_count,
        "_downloaded_count": len(results),
        "_n_pages": page_no,
        "_note": "results are verbatim API records; only the pagination envelope was removed",
        "results": results,
    }
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "name": name,
        "endpoint": first_url,
        "n_pages": page_no,
        "declared_count": declared_count,
        "downloaded_count": len(results),
        "count_matches": declared_count == len(results),
        "combined_file": combined_path.name,
        "combined_sha256": sha256_of(combined_path),
        "combined_bytes": combined_path.stat().st_size,
        "page_files": page_files,
    }


def verify(outdir: Path, entries: list[dict[str, Any]]) -> list[str]:
    """Read every written file back. Returns a list of problem strings (empty == clean)."""
    problems: list[str] = []
    for e in entries:
        combined = outdir / e["combined_file"]

        if not combined.exists():
            problems.append(f"{e['name']}: combined file missing")
            continue
        if combined.stat().st_size == 0:
            problems.append(f"{e['name']}: combined file is empty")
            continue

        try:
            reloaded = json.loads(combined.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{e['name']}: combined file does not parse as JSON ({exc})")
            continue

        n = len(reloaded.get("results", []))
        if n != e["downloaded_count"]:
            problems.append(
                f"{e['name']}: read back {n} records, expected {e['downloaded_count']}"
            )
        if n == 0:
            problems.append(f"{e['name']}: zero records downloaded")
        if not e["count_matches"]:
            problems.append(
                f"{e['name']}: API declared count={e['declared_count']} but "
                f"{e['downloaded_count']} records were downloaded — pagination may be incomplete"
            )

        for rel in e["page_files"]:
            p = outdir / rel
            if not p.exists() or p.stat().st_size == 0:
                problems.append(f"{e['name']}: page file missing or empty: {rel}")
                continue
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{e['name']}: page file does not parse: {rel} ({exc})")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="data/external/mitotox_raw",
                    help="output directory (default: data/external/mitotox_raw)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help=f"API root (default: {DEFAULT_BASE_URL})")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    started = datetime.now(timezone.utc)
    print(f"MitoTox raw download  |  base URL: {args.base_url}")
    print(f"Output directory: {outdir.resolve()}")
    print()

    entries: list[dict[str, Any]] = []
    failures: list[str] = []

    for endpoint, name in ENDPOINTS.items():
        try:
            entries.append(download_endpoint(session, args.base_url, endpoint, name, outdir))
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {name}: FAILED — {exc}")
            failures.append(f"{name}: {exc}")
        print()

    print("Verifying written files...")
    problems = verify(outdir, entries)

    manifest = {
        "tool": "scripts/download_mitotox.py",
        "base_url": args.base_url,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "endpoints": entries,
        "failures": failures,
        "verification_problems": problems,
        "status": "ok" if (not failures and not problems) else "incomplete",
    }
    (outdir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print("=" * 68)
    print(f"{'endpoint':<12}{'pages':>7}{'declared':>11}{'downloaded':>13}{'match':>8}")
    print("-" * 68)
    for e in entries:
        print(
            f"{e['name']:<12}{e['n_pages']:>7}{str(e['declared_count']):>11}"
            f"{e['downloaded_count']:>13}{'yes' if e['count_matches'] else 'NO':>8}"
        )
    print("=" * 68)

    if failures:
        print("\nFAILED endpoints:")
        for f in failures:
            print(f"  - {f}")
    if problems:
        print("\nVerification problems:")
        for p in problems:
            print(f"  - {p}")

    if failures or problems:
        print("\nSTATUS: INCOMPLETE — do not proceed to curation.")
        return 1

    print("\nSTATUS: OK — all endpoints downloaded, counts match, files read back cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
