#!/usr/bin/env python3
"""
Checkpoint 1B.6: inspection of the raw Tox21 download (AID 720637 + AID 720634).

DESCRIPTIVE ONLY. Nothing is merged, labelled, filtered, deduplicated or
transformed. The two assays are read separately; the cross-assay section
computes overlap statistics without writing any joined table.

Usage:  py -3 scripts/inspect_tox21_raw.py
Output: results/checkpoint_1B6_tox21_inspection.txt (and stdout)
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

RAW = os.path.join("data", "external", "tox21_raw")
OUT = os.path.join("results", "checkpoint_1B6_tox21_inspection.txt")

# PubChem data tables prefix the real rows with metadata rows whose
# PUBCHEM_RESULT_TAG is a keyword rather than a row number. Different
# assays use different keywords (720634 adds RESULT_ATTR_CONC_MICROMOL),
# so the rule is: a data row has a numeric tag. Everything else is metadata.

_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def h(t):
    w("")
    w("=" * 78)
    w(t)
    w("=" * 78)


def load_datatable(fname, usecols=None):
    df = pd.read_csv(os.path.join(RAW, fname), dtype=str, low_memory=False,
                     usecols=usecols)
    n_all = len(df)
    tag = df["PUBCHEM_RESULT_TAG"].fillna("")
    df = df[tag.str.match(r"^\d+$")].copy()
    return df, n_all


def value_counts(df, col, label=None, maxn=40):
    w("")
    if col not in df.columns:
        w("  %s : COLUMN NOT PRESENT" % (label or col))
        return
    s = df[col]
    na = int((s.fillna("") == "").sum())   # NaN and "" together, counted once
    c = Counter(v for v in s.fillna("").tolist() if v != "")
    w("  %s  -- %d distinct non-empty values, %d empty/NaN"
      % (label or col, len(c), na))
    for v, k in c.most_common(maxn):
        w("      %8d  %r" % (k, v))
    if len(c) > maxn:
        w("      ... %d more not shown" % (len(c) - maxn))


def main():
    h("0. FILES AND DOWNLOAD PROVENANCE")
    man = json.load(open(os.path.join(RAW, "download_manifest.json"), encoding="utf-8"))
    w("manifest status : %s" % man["status"])
    w("retrieved       : %s -> %s UTC" % (man["started_utc"], man["finished_utc"]))
    w("source          : %s" % man["source"])
    w("")
    for f in man["files"]:
        w("  %-34s HTTP %s  %10d B  sha256 %s"
          % (f["filename"], f["http_status"], f["bytes"], f["sha256"][:24]))
    w("")
    w("failures: %s" % (man["failures"] or "none"))

    # ---------------- AID 720637 ----------------
    h("1. AID 720637 - MMP SUMMARY ASSAY")
    desc = json.load(open(os.path.join(RAW, "aid720637_description.json"), encoding="utf-8"))
    a = desc["PC_AssayContainer"][0]["assay"]["descr"]
    w("assay name : %s" % a["name"])
    w("aid        : %s (version %s)" % (a["aid"]["id"], a["aid"].get("version")))
    w("xrefs      : %s" % [(x["xref"].get("aid"), x.get("comment"))
                           for x in a.get("xref", []) if "aid" in x["xref"]])

    df, n_all = load_datatable("aid720637_datatable_pcget.csv")
    w("")
    w("file format          : CSV (PubChem data table, pcget.cgi export)")
    w("total lines incl. metadata rows : %d" % n_all)
    w("data records (rows)  : %d" % len(df))
    w("unique PUBCHEM_SID   : %d" % df["PUBCHEM_SID"].nunique())
    w("unique PUBCHEM_CID   : %d (empty CID in %d rows)"
      % (df["PUBCHEM_CID"].nunique(),
         int(df["PUBCHEM_CID"].isna().sum() + (df["PUBCHEM_CID"].fillna("") == "").sum())))
    sm = "PUBCHEM_EXT_DATASOURCE_SMILES"
    w("SMILES column        : %s" % ("PRESENT (%s)" % sm if sm in df.columns else "ABSENT"))
    if sm in df.columns:
        nonempty = int((df[sm].fillna("") != "").sum())
        w("  non-empty SMILES   : %d / %d (%.1f%%)"
          % (nonempty, len(df), 100.0 * nonempty / len(df)))
    w("")
    w("all columns (%d):" % len(df.columns))
    for c in df.columns:
        w("    %s" % c)

    w("")
    w("--- EVERY OUTCOME-RELATED FIELD, ACTUAL VALUES FROM THE FILE ---")
    for col in ["PUBCHEM_ACTIVITY_OUTCOME", "Activity Summary", "Ratio Activity",
                "535 nm Activity", "590 nm Activity", "Viability Activity",
                "Sample Source"]:
        value_counts(df, col)

    w("")
    w("--- PUBCHEM_ACTIVITY_OUTCOME x Activity Summary (cross-tab) ---")
    ct = pd.crosstab(df["Activity Summary"].fillna("(empty)"),
                     df["PUBCHEM_ACTIVITY_OUTCOME"].fillna("(empty)"))
    for line in ct.to_string().split("\n"):
        w("  " + line)

    w("")
    w("--- Ratio Efficacy (%) sign, by Activity Summary ---")
    w("  (efficacy sign is the direction evidence: negative = signal decrease)")
    eff = pd.to_numeric(df["Ratio Efficacy (%)"], errors="coerce")
    tmp = pd.DataFrame({"summary": df["Activity Summary"].fillna("(empty)"),
                        "eff": eff})
    for name, grp in tmp.groupby("summary"):
        e = grp["eff"].dropna()
        if len(e) == 0:
            w("  %-28s n=%5d  no numeric efficacy" % (name, len(grp)))
            continue
        w("  %-28s n=%5d  neg=%5d zero=%5d pos=%5d   min=%8.2f median=%8.2f max=%8.2f"
          % (name, len(grp), int((e < 0).sum()), int((e == 0).sum()),
             int((e > 0).sum()), e.min(), e.median(), e.max()))

    w("")
    w("--- Ratio Potency (uM) availability by Activity Summary ---")
    pot = pd.to_numeric(df["Ratio Potency (uM)"], errors="coerce")
    tmp2 = pd.DataFrame({"summary": df["Activity Summary"].fillna("(empty)"), "pot": pot})
    for name, grp in tmp2.groupby("summary"):
        p = grp["pot"].dropna()
        w("  %-28s n=%5d  with potency=%5d  median=%s"
          % (name, len(grp), len(p), ("%.3f" % p.median()) if len(p) else "-"))

    # ---------------- AID 720634 ----------------
    h("2. AID 720634 - CELL VIABILITY COUNTERSCREEN")
    desc2 = json.load(open(os.path.join(RAW, "aid720634_description.json"), encoding="utf-8"))
    a2 = desc2["PC_AssayContainer"][0]["assay"]["descr"]
    w("assay name : %s" % a2["name"])
    w("aid        : %s (version %s)" % (a2["aid"]["id"], a2["aid"].get("version")))

    hdr = pd.read_csv(os.path.join(RAW, "aid720634_datatable_pcget.csv"),
                      nrows=0).columns.tolist()
    w("")
    w("file format     : CSV (PubChem data table, pcget.cgi export)")
    w("total columns   : %d" % len(hdr))
    conc_cols = [c for c in hdr if c.startswith("Activity at ")]
    other = [c for c in hdr if not c.startswith("Activity at ")]
    w("  of which per-concentration 'Activity at X uM' columns : %d" % len(conc_cols))
    w("  remaining columns (%d):" % len(other))
    for c in other:
        w("    %s" % c)

    keep = ["PUBCHEM_RESULT_TAG", "PUBCHEM_SID", "PUBCHEM_CID",
            "PUBCHEM_EXT_DATASOURCE_SMILES", "PUBCHEM_ACTIVITY_OUTCOME",
            "PUBCHEM_ACTIVITY_SCORE"]
    keep += [c for c in hdr if c.startswith("Phenotype-Replicate")]
    keep += [c for c in hdr if c.startswith("Fit_CurveClass-Replicate")]
    keep = [c for c in keep if c in hdr]
    df2, n_all2 = load_datatable("aid720634_datatable_pcget.csv", usecols=keep)
    w("")
    w("total lines incl. metadata rows : %d" % n_all2)
    w("data records (rows) : %d" % len(df2))
    w("unique PUBCHEM_SID  : %d" % df2["PUBCHEM_SID"].nunique())
    w("unique PUBCHEM_CID  : %d" % df2["PUBCHEM_CID"].nunique())
    if sm in df2.columns:
        w("SMILES non-empty    : %d / %d"
          % (int((df2[sm].fillna("") != "").sum()), len(df2)))
    for col in ["PUBCHEM_ACTIVITY_OUTCOME"] + \
               [c for c in df2.columns if c.startswith("Phenotype-Replicate")]:
        value_counts(df2, col)

    # ---------------- cross-assay ----------------
    h("3. CROSS-ASSAY INSPECTION (no join written, no data modified)")
    s1, s2 = set(df["PUBCHEM_SID"].dropna()), set(df2["PUBCHEM_SID"].dropna())
    c1 = set(df["PUBCHEM_CID"].dropna()) - {""}
    c2 = set(df2["PUBCHEM_CID"].dropna()) - {""}
    w("SID: 720637=%d  720634=%d  intersection=%d  only-637=%d  only-634=%d"
      % (len(s1), len(s2), len(s1 & s2), len(s1 - s2), len(s2 - s1)))
    w("CID: 720637=%d  720634=%d  intersection=%d  only-637=%d  only-634=%d"
      % (len(c1), len(c2), len(c1 & c2), len(c1 - c2), len(c2 - c1)))
    w("")
    w("Join uniqueness:")
    w("  720637 rows per SID : max=%d  (1 means SID is a unique key)"
      % df["PUBCHEM_SID"].value_counts().max())
    w("  720634 rows per SID : max=%d" % df2["PUBCHEM_SID"].value_counts().max())
    vc1 = df["PUBCHEM_CID"].replace("", pd.NA).dropna().value_counts()
    vc2 = df2["PUBCHEM_CID"].replace("", pd.NA).dropna().value_counts()
    w("  720637 rows per CID : max=%d  (CIDs with >1 SID: %d)"
      % (vc1.max(), int((vc1 > 1).sum())))
    w("  720634 rows per CID : max=%d  (CIDs with >1 SID: %d)"
      % (vc2.max(), int((vc2 > 1).sum())))

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Report written to %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
