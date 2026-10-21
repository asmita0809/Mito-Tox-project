#!/usr/bin/env python3
"""
Checkpoint 1A: basic inspection of the raw MitoTox download.

DESCRIPTIVE ONLY. This script does not filter, label, join for analysis,
deduplicate, or interpret. It reports what is in the files.

Usage:  py -3 scripts/inspect_mitotox_raw.py
Output: results/checkpoint_1A_inspection.txt  (and stdout)
"""
import json
import os
import sys
from collections import Counter, defaultdict

RAW = os.path.join("data", "external", "mitotox_raw")
OUT = os.path.join("results", "checkpoint_1A_inspection.txt")
NAMES = ["compounds", "records", "functions", "targets", "papers"]

_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def h(title):
    w("")
    w("=" * 78)
    w(title)
    w("=" * 78)


def load(name):
    path = os.path.join(RAW, "mitotox_%s.json" % name)
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    return obj


def field_report(rows, label):
    """Union of keys across all rows, with non-empty counts. No assumptions."""
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    w("%s: %d records, %d distinct fields" % (label, len(rows), len(keys)))
    w("")
    w("  %-24s %10s %10s  %s" % ("field", "non-empty", "pct", "example value"))
    w("  " + "-" * 74)
    for k in keys:
        vals = [r.get(k) for r in rows]
        nonempty = [v for v in vals
                    if v is not None and v != "" and v != [] and v != {}]
        ex = ""
        for v in nonempty:
            ex = repr(v)
            break
        if len(ex) > 34:
            ex = ex[:31] + "..."
        w("  %-24s %10d %9.1f%%  %s"
          % (k, len(nonempty), 100.0 * len(nonempty) / max(1, len(rows)), ex))
    return keys


def top_values(rows, key, n=20, label=None):
    c = Counter()
    empty = 0
    for r in rows:
        v = r.get(key)
        if v is None or v == "":
            empty += 1
        else:
            c[str(v)] += 1
    w("")
    w("  %s  (%d distinct non-empty values; %d empty/None)"
      % (label or key, len(c), empty))
    for v, k in c.most_common(n):
        s = v if len(v) <= 52 else v[:49] + "..."
        w("    %6d  %s" % (k, s))
    if len(c) > n:
        w("    ... %d more distinct values not shown" % (len(c) - n))
    return c, empty


def main():
    if not os.path.isdir(RAW):
        sys.stderr.write("Cannot find %s - run this from the project root.\n" % RAW)
        return 1

    data = {}
    h("0. DOWNLOAD INTEGRITY (from download_manifest.json)")
    with open(os.path.join(RAW, "download_manifest.json"), "r", encoding="utf-8") as fh:
        man = json.load(fh)
    w("manifest status : %s" % man["status"])
    w("base URL        : %s" % man["base_url"])
    w("retrieved       : %s -> %s" % (man["started_utc"], man["finished_utc"]))
    w("python          : %s" % man.get("python_version", "?"))
    w("failures        : %s" % (man["failures"] or "none"))
    w("verify problems : %s" % (man["verification_problems"] or "none"))
    w("")
    w("  %-12s %6s %10s %12s %7s  %s"
      % ("endpoint", "pages", "declared", "downloaded", "match", "sha256[:12]"))
    w("  " + "-" * 74)
    for e in man["endpoints"]:
        w("  %-12s %6d %10s %12d %7s  %s"
          % (e["name"], e["n_pages"], e["declared_count"], e["downloaded_count"],
             "yes" if e["count_matches"] else "NO", e["combined_sha256"][:12]))

    for n in NAMES:
        obj = load(n)
        data[n] = obj["results"]
        if len(obj["results"]) != obj["_downloaded_count"]:
            w("  !! %s: re-read %d rows, manifest said %d"
              % (n, len(obj["results"]), obj["_downloaded_count"]))
    w("")
    w("All five files re-read successfully from disk.")

    compounds = data["compounds"]
    records = data["records"]
    functions = data["functions"]
    targets = data["targets"]
    papers = data["papers"]

    h("1-2. RECORD COUNTS")
    for n in NAMES:
        w("  %-12s %6d" % (n, len(data[n])))

    h("3. FIELD NAMES PER ENDPOINT")
    for n in NAMES:
        w("")
        field_report(data[n], n)

    h("4. MITOCHONDRIAL FUNCTION / ENDPOINT ONTOLOGY")
    by_id = {}
    for f in functions:
        by_id[f["func_id"]] = f
    children = defaultdict(list)
    roots = []
    for f in functions:
        p = f.get("parent_func_id")
        if p and p in by_id:
            children[p].append(f["func_id"])
        else:
            roots.append(f["func_id"])
    w("%d function nodes; %d top-level (parent not itself a node)"
      % (len(functions), len(roots)))
    w("")
    w("Top-level nodes and their parent field:")
    for r in sorted(roots):
        w("  %-10s parent=%-6r %s"
          % (r, by_id[r].get("parent_func_id"), by_id[r].get("name")))

    rec_func = Counter(r.get("func") for r in records)
    w("")
    w("Records reference %d distinct func ids; %d records have empty/None func"
      % (len([k for k in rec_func if k]), rec_func.get(None, 0) + rec_func.get("", 0)))
    orphan = [k for k in rec_func if k and k not in by_id]
    w("func ids used by records but absent from the functions table: %s"
      % (sorted(orphan) if orphan else "none"))

    def subtree(fid):
        out = [fid]
        stack = list(children.get(fid, []))
        while stack:
            c = stack.pop()
            out.append(c)
            stack.extend(children.get(c, []))
        return out

    w("")
    w("Direct record counts per function node, whole tree, depth-first.")
    w("'direct' = records whose func field is exactly that node.")
    w("'subtree' = direct + all descendants.")
    w("")

    def walk(fid, depth):
        sub = subtree(fid)
        direct = rec_func.get(fid, 0)
        total = sum(rec_func.get(x, 0) for x in sub)
        if total == 0 and direct == 0:
            marker = "     ."
        else:
            marker = "%6d" % total
        w("  %s %6d  %s%-10s %s"
          % (marker, direct, "  " * depth, fid, by_id[fid].get("name")))
        for c in sorted(children.get(fid, [])):
            walk(c, depth + 1)

    w("  subtree direct  node")
    w("  " + "-" * 74)
    for r in sorted(roots):
        walk(r, 0)

    h("5. RESULT / LABEL FIELDS IN THE EXPERIMENTAL RECORDS")
    tc = Counter(type(r.get("result")).__name__ for r in records)
    w("python type of `result` across all %d records:" % len(records))
    for t, k in tc.most_common():
        w("    %-10s %6d" % (t, k))
    vc = Counter(repr(r.get("result")) for r in records)
    w("")
    w("distinct raw values of `result`:")
    for v, k in vc.most_common():
        w("    %-10s %6d" % (v, k))
    top_values(records, "func_act", 25, "`func_act` (direction of effect on the function)")
    top_values(records, "targ_act", 25, "`targ_act` (direction of effect on the target)")
    top_values(records, "criteria", 25, "`criteria` (positive criterion)")

    h("6. ASSAY / METHOD FIELDS")
    top_values(records, "method", 30, "`method`")

    h("7. DOSE AND TIME FIELDS")
    top_values(records, "dose", 25, "`dose`")
    top_values(records, "time", 25, "`time`")

    h("8. SPECIES AND BIOLOGICAL MODEL FIELDS")
    top_values(records, "species", 30, "`species`")
    top_values(records, "animal_model", 25, "`animal_model`")

    h("9-11. CHEMICAL IDENTIFIERS IN THE COMPOUND TABLE")
    ckeys = set()
    for c in compounds:
        ckeys.update(c.keys())
    w("all compound fields: %s" % ", ".join(sorted(ckeys)))
    w("")
    wanted = ["smiles", "smile", "canonical_smiles", "inchi", "inchikey", "structure",
              "chembl", "chebi", "cas", "cid", "pubchem", "drugbank", "kegg", "unii",
              "molecular_formula", "molecular_weight", "iupac"]
    w("substring search over field names (case-insensitive):")
    for tok in wanted:
        hits = sorted(k for k in ckeys if tok in k.lower())
        w("  %-20s -> %s" % (tok, hits if hits else "ABSENT"))
    w("")
    for k in ["cid", "molecular_formula", "molecular_weight", "name", "compound_ID"]:
        if k in ckeys:
            n = sum(1 for c in compounds
                    if c.get(k) not in (None, "", [], {}))
            w("  %-20s non-empty in %d / %d compounds (%.1f%%)"
              % (k, n, len(compounds), 100.0 * n / len(compounds)))

    h("12. RECORD -> COMPOUND LINKAGE")
    have = 0
    parsed_ok = 0
    ids_in_records = set()
    weird = []
    for r in records:
        v = r.get("compound")
        if v in (None, ""):
            continue
        have += 1
        s = str(v)
        if ": " in s:
            cid_part = s.split(": ", 1)[0]
            if cid_part.startswith("D") and cid_part[1:].isdigit():
                parsed_ok += 1
                ids_in_records.add(cid_part)
                continue
        if len(weird) < 15:
            weird.append(s)
    w("records with a non-empty `compound` field : %d / %d" % (have, len(records)))
    w("matching the 'Dxxxx: name' pattern        : %d" % parsed_ok)
    w("not matching that pattern                 : %d" % (have - parsed_ok))
    if weird:
        w("  examples that do not match:")
        for s in weird:
            w("    %s" % (s if len(s) <= 66 else s[:63] + "..."))
    table_ids = set(c.get("compound_ID") for c in compounds)
    w("")
    w("distinct compound IDs referenced by records : %d" % len(ids_in_records))
    w("compound IDs in the compounds table         : %d" % len(table_ids))
    missing = sorted(ids_in_records - table_ids)
    w("referenced but absent from compounds table  : %d %s"
      % (len(missing), missing[:10] if missing else ""))
    w("in compounds table but never referenced     : %d"
      % len(table_ids - ids_in_records))

    h("13. TARGET AND PAPER LINKAGE")
    rt = Counter(r.get("targ") for r in records)
    n_null = rt.get(None, 0) + rt.get("", 0)
    w("records with a non-empty `targ`  : %d / %d" % (len(records) - n_null, len(records)))
    tids = set(t.get("targ_id") for t in targets)
    orph_t = sorted(k for k in rt if k and k not in tids)
    w("targ ids absent from targets table: %d %s" % (len(orph_t), orph_t[:10]))
    rp = Counter(r.get("paper") for r in records)
    n_null_p = rp.get(None, 0) + rp.get("", 0)
    w("records with a non-empty `paper` : %d / %d" % (len(records) - n_null_p, len(records)))
    pkeys = set()
    for p in papers:
        pkeys.update(p.keys())
    w("paper fields: %s" % ", ".join(sorted(pkeys)))

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Report written to %s (%d lines)" % (OUT, len(_lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
