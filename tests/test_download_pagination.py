#!/usr/bin/env python3
"""
Self-test for scripts/download_mitotox.py.

Serves a fake DRF-style paginated API on localhost with KNOWN record counts and
deliberately uneven page sizes, runs the real downloader against it, and checks
that pagination was followed to completion and that verification catches a
truncated download.

This tests the download machinery only. It uses synthetic placeholder records —
no MitoTox data is involved, and nothing here is scientific output.

Run:  python tests/test_download_pagination.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]

# endpoint name -> (total records, page size). Uneven totals on purpose:
# 1 page exactly, a partial last page, and a single short page.
FIXTURES = {
    "compounds": (250, 100),   # 3 pages, last page has 50
    "records":   (100, 100),   # exactly 1 page, next must be null
    "functions": (7, 100),     # 1 short page
    "targets":   (201, 100),   # 3 pages, last page has 1
    "papers":    (1, 100),     # single record
}
PATH_TO_NAME = {f"/api/{n}/list": n for n in ("compounds", "records", "functions", "targets", "papers")}
PORT = 8731


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        name = PATH_TO_NAME.get(parsed.path)
        if name is None:
            self.send_error(404)
            return

        total, page_size = FIXTURES[name]
        page = int(parse_qs(parsed.query).get("page", ["1"])[0])
        start = (page - 1) * page_size
        end = min(start + page_size, total)

        results = [{"id": f"{name.upper()}{i:05d}", "payload": i} for i in range(start, end)]
        next_url = (
            f"http://127.0.0.1:{PORT}/api/{name}/list?page={page + 1}"
            if end < total else None
        )
        body = json.dumps({
            "count": total,
            "next": next_url,
            "previous": None,
            "results": results,
        }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    failures = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "raw"
            proc = subprocess.run(
                [sys.executable, str(REPO / "scripts" / "download_mitotox.py"),
                 "--outdir", str(out),
                 "--base-url", f"http://127.0.0.1:{PORT}/api"],
                capture_output=True, text=True,
            )
            print(proc.stdout)
            if proc.stderr.strip():
                print("STDERR:", proc.stderr, file=sys.stderr)

            if proc.returncode != 0:
                failures.append(f"downloader exited {proc.returncode}, expected 0")

            manifest = json.loads((out / "download_manifest.json").read_text())
            by_name = {e["name"]: e for e in manifest["endpoints"]}

            for name, (total, page_size) in FIXTURES.items():
                e = by_name.get(name)
                if e is None:
                    failures.append(f"{name}: missing from manifest")
                    continue
                expected_pages = max(1, -(-total // page_size))
                if e["downloaded_count"] != total:
                    failures.append(
                        f"{name}: downloaded {e['downloaded_count']}, expected {total}")
                if e["n_pages"] != expected_pages:
                    failures.append(
                        f"{name}: {e['n_pages']} pages, expected {expected_pages}")
                if not e["count_matches"]:
                    failures.append(f"{name}: count_matches False")

                combined = json.loads((out / e["combined_file"]).read_text())
                ids = [r["id"] for r in combined["results"]]
                if len(set(ids)) != total:
                    failures.append(
                        f"{name}: {len(set(ids))} unique ids for {len(ids)} rows "
                        "— pages overlapped or were skipped")
                if ids != sorted(ids):
                    failures.append(f"{name}: record order not preserved across pages")

            if manifest["status"] != "ok":
                failures.append(f"manifest status {manifest['status']!r}, expected 'ok'")

            # Negative control: verification must FLAG a truncated download.
            from importlib.machinery import SourceFileLoader
            dl = SourceFileLoader(
                "dl", str(REPO / "scripts" / "download_mitotox.py")).load_module()
            bad = dict(by_name["compounds"])
            bad["downloaded_count"] = 249      # pretend one record went missing
            bad["count_matches"] = False
            if not dl.verify(out, [bad]):
                failures.append(
                    "negative control: verify() did not flag a truncated download")

    finally:
        server.shutdown()

    print("-" * 60)
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — pagination followed to completion on all 5 fixtures; "
          "no overlap, no gaps, order preserved; truncation is detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
