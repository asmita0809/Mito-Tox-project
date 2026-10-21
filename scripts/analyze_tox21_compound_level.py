#!/usr/bin/env python3
"""
Checkpoint 1C: SID -> CID relationship, outcome conflicts, structure availability
and viability annotation for the Tox21 MMP dataset.

ANALYSIS ONLY. Reads the raw files, never writes to them. Produces NO training
dataset and NO label column: the compound-level table it emits carries per-outcome
COUNTS and a descriptive combination category, nothing that could be mistaken for
an assigned label.

Usage:  py -3 scripts/analyze_tox21_compound_level.py
Outputs (all under results/):
    checkpoint_1C_analysis.txt              full report
    checkpoint_1C_compound_summary.csv      one row per CID, counts only
    checkpoint_1C_conflicting_compounds.csv one row per SID of conflicted CIDs
"""
import os
import sys
from collections import Counter

import pandas as pd

RAW = os.path.join("data", "external", "tox21_raw")
RES = "results"
MMP = "aid720637_datatable_pcget.csv"
VIA = "aid720634_datatable_pcget.csv"
SMI = "PUBCHEM_EXT_DATASOURCE_SMILES"
OUT = "PUBCHEM_ACTIVITY_OUTCOME"
_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def h(t):
    w("")
    w("=" * 78)
    w(t)
    w("=" * 78)


def load(fname, usecols=None):
    """Data rows only. A PubChem data row has a numeric PUBCHEM_RESULT_TAG;
    every other tag (RESULT_TYPE, RESULT_DESCR, RESULT_UNIT,
    RESULT_IS_ACTIVE_CONCENTRATION, RESULT_ATTR_CONC_MICROMOL, ...) is metadata."""
    df = pd.read_csv(os.path.join(RAW, fname), dtype=str, low_memory=False, usecols=usecols)
    tag = df["PUBCHEM_RESULT_TAG"].fillna("")
    return df[tag.str.match(r"^\d+$")].copy()


def blank(s):
    return s.fillna("").str.strip() == ""


def main():
    os.makedirs(RES, exist_ok=True)
    df = load(MMP)
    df["_cid"] = df["PUBCHEM_CID"].fillna("").str.strip()
    df["_smi"] = df[SMI].fillna("").str.strip()

    h("1. SID / CID RELATIONSHIP")
    w("SID rows (samples)      : %d" % len(df))
    w("unique SID              : %d" % df["PUBCHEM_SID"].nunique())
    w("unique CID (non-blank)  : %d" % df.loc[df._cid != "", "_cid"].nunique())
    w("rows with no CID        : %d" % int((df._cid == "").sum()))
    w("rows with no SMILES     : %d" % int((df._smi == "").sum()))
    w("rows with no CID AND no SMILES : %d"
      % int(((df._cid == "") & (df._smi == "")).sum()))
    w("rows with no CID but WITH SMILES: %d"
      % int(((df._cid == "") & (df._smi != "")).sum()))

    withcid = df[df._cid != ""].copy()
    per = withcid.groupby("_cid")["PUBCHEM_SID"].nunique()
    w("")
    w("SIDs per CID distribution:")
    for k, v in sorted(Counter(per.values).items()):
        w("   %2d SID(s): %6d CIDs   (%6d rows)" % (k, v, k * v))
    w("   CIDs with >1 SID : %d  covering %d rows"
      % (int((per > 1).sum()), int(per[per > 1].sum())))

    multi = set(per[per > 1].index)
    mrows = withcid[withcid._cid.isin(multi)]

    w("")
    w("WHY do several SIDs share a CID? Evidence:")
    g = mrows.groupby("_cid")
    src_same = int((g["Sample Source"].nunique() == 1).sum())
    smi_same = int((g["_smi"].nunique() == 1).sum())
    w("  CIDs where all SIDs share ONE Sample Source : %d / %d"
      % (src_same, len(multi)))
    w("  CIDs where all SIDs share ONE depositor SMILES string: %d / %d"
      % (smi_same, len(multi)))
    w("")
    w("  distinct Sample Sources per multi-SID CID:")
    for k, v in sorted(Counter(g["Sample Source"].nunique().values).items()):
        w("     %2d source(s): %5d CIDs" % (k, v))
    w("")
    w("  Example multi-SID CIDs (first 5, showing every SID):")
    for cid in list(sorted(multi))[:5]:
        sub = withcid[withcid._cid == cid]
        w("   CID %s  (%d SIDs)" % (cid, len(sub)))
        for _, r in sub.iterrows():
            w("      SID %-12s %-14s src=%-22s smiles=%s"
              % (r["PUBCHEM_SID"], r[OUT], str(r["Sample Source"])[:22],
                 (r["_smi"][:40] + "...") if len(r["_smi"]) > 40 else r["_smi"]))

    w("")
    w("Rows with NO CID - what are they? Sample Source and SMILES presence:")
    nocid = df[df._cid == ""]
    w("  n = %d ; with SMILES = %d" % (len(nocid), int((nocid._smi != "").sum())))
    w("  outcome breakdown : %s" % dict(Counter(nocid[OUT])))
    w("  top sample sources: %s"
      % dict(Counter(nocid["Sample Source"].fillna("(empty)")).most_common(8)))
    w("  contains '.' (multi-component) in SMILES: %d of %d with SMILES"
      % (int(nocid.loc[nocid._smi != "", "_smi"].str.contains(r"\.", regex=True).sum()),
         int((nocid._smi != "").sum())))

    # ---------------- 2. outcome combinations ----------------
    h("2. OUTCOME COMBINATIONS PER UNIQUE CID")
    piv = (withcid.assign(one=1)
           .pivot_table(index="_cid", columns=OUT, values="one",
                        aggfunc="sum", fill_value=0))
    for c in ["Active", "Inactive", "Inconclusive"]:
        if c not in piv.columns:
            piv[c] = 0
    piv = piv[["Active", "Inactive", "Inconclusive"]]
    piv.columns = ["n_active", "n_inactive", "n_inconclusive"]
    piv["n_sid"] = piv.sum(axis=1)

    def combo(r):
        parts = []
        if r.n_active:
            parts.append("Active")
        if r.n_inactive:
            parts.append("Inactive")
        if r.n_inconclusive:
            parts.append("Inconclusive")
        return " + ".join(parts) if parts else "(none)"

    piv["combination"] = piv.apply(combo, axis=1)

    order = ["Active", "Inactive", "Inconclusive",
             "Active + Inactive", "Active + Inconclusive",
             "Inactive + Inconclusive", "Active + Inactive + Inconclusive"]
    cc = Counter(piv["combination"])
    w("%-38s %8s %8s" % ("combination", "CIDs", "SIDs"))
    w("-" * 58)
    tot_c = tot_s = 0
    for k in order:
        n = cc.get(k, 0)
        s = int(piv.loc[piv.combination == k, "n_sid"].sum())
        tot_c += n
        tot_s += s
        w("%-38s %8d %8d" % (k, n, s))
    for k in sorted(set(cc) - set(order)):
        w("%-38s %8d %8d" % (k + "  [unexpected]", cc[k],
                             int(piv.loc[piv.combination == k, "n_sid"].sum())))
    w("-" * 58)
    w("%-38s %8d %8d" % ("TOTAL (CIDs with a CID)", tot_c, tot_s))
    w("")
    w("single-SID CIDs   : %d" % int((piv.n_sid == 1).sum()))
    w("multi-SID CIDs    : %d" % int((piv.n_sid > 1).sum()))
    w("  of which fully concordant (one outcome value): %d"
      % int(((piv.n_sid > 1) &
             ((piv[["n_active", "n_inactive", "n_inconclusive"]] > 0).sum(axis=1) == 1)).sum()))

    # ---------------- 3. conflicts ----------------
    h("3. CONFLICTING COMPOUNDS (Active AND Inactive on the same CID)")
    hard = piv[(piv.n_active > 0) & (piv.n_inactive > 0)]
    w("CIDs with at least one Active and one Inactive SID : %d" % len(hard))
    w("  of these, also carrying Inconclusive SIDs        : %d"
      % int((hard.n_inconclusive > 0).sum()))
    w("  SIDs involved                                     : %d"
      % int(hard.n_sid.sum()))
    w("")
    w("Active:Inactive SID split among conflicted CIDs:")
    for k, v in sorted(Counter(zip(hard.n_active, hard.n_inactive)).items()):
        w("   %dA : %dI   -> %d CIDs" % (k[0], k[1], v))

    detail = withcid[withcid._cid.isin(hard.index)].copy()
    keep = ["_cid", "PUBCHEM_SID", OUT, "Activity Summary", "Ratio Potency (uM)",
            "Ratio Efficacy (%)", "Viability Activity", "Sample Source", "_smi"]
    detail = detail[keep].sort_values(["_cid", OUT])
    detail.rename(columns={"_cid": "PUBCHEM_CID", "_smi": SMI}).to_csv(
        os.path.join(RES, "checkpoint_1C_conflicting_compounds.csv"), index=False)
    w("")
    w("Full per-SID detail written to results/checkpoint_1C_conflicting_compounds.csv")
    w("")
    w("First 12 conflicted CIDs, every SID:")
    shown = 0
    for cid, sub in detail.groupby("_cid"):
        if shown >= 12:
            break
        shown += 1
        w("  CID %s" % cid)
        for _, r in sub.iterrows():
            w("     SID %-12s %-13s %-20s pot=%-9s eff=%-9s via=%-22s src=%s"
              % (r["PUBCHEM_SID"], r[OUT], str(r["Activity Summary"])[:20],
                 str(r["Ratio Potency (uM)"])[:9], str(r["Ratio Efficacy (%)"])[:9],
                 str(r["Viability Activity"])[:22], str(r["Sample Source"])[:18]))

    w("")
    w("EVIDENCE for a tie-break rule - the Active SIDs of conflicted compounds:")
    act = detail[detail[OUT] == "Active"]
    eff = pd.to_numeric(act["Ratio Efficacy (%)"], errors="coerce").dropna()
    pot = pd.to_numeric(act["Ratio Potency (uM)"], errors="coerce").dropna()
    allact = withcid[withcid[OUT] == "Active"]
    eff_all = pd.to_numeric(allact["Ratio Efficacy (%)"], errors="coerce").dropna()
    pot_all = pd.to_numeric(allact["Ratio Potency (uM)"], errors="coerce").dropna()
    w("  %-34s %7s %10s %10s %10s" % ("", "n", "min", "median", "max"))
    w("  %-34s %7d %10.2f %10.2f %10.2f"
      % ("efficacy, Active SIDs in conflicts", len(eff), eff.min(), eff.median(), eff.max()))
    w("  %-34s %7d %10.2f %10.2f %10.2f"
      % ("efficacy, ALL Active SIDs", len(eff_all), eff_all.min(), eff_all.median(), eff_all.max()))
    w("  %-34s %7d %10.3f %10.3f %10.3f"
      % ("potency uM, Active SIDs in conflicts", len(pot), pot.min(), pot.median(), pot.max()))
    w("  %-34s %7d %10.3f %10.3f %10.3f"
      % ("potency uM, ALL Active SIDs", len(pot_all), pot_all.min(), pot_all.median(), pot_all.max()))

    # ---------------- 4. structures ----------------
    h("4. STRUCTURE AVAILABILITY")
    smi_per_cid = withcid.groupby("_cid")["_smi"].agg(
        n_distinct=lambda s: s[s != ""].nunique(),
        n_blank=lambda s: int((s == "").sum()),
        n=lambda s: len(s))
    w("unique CIDs                          : %d" % len(smi_per_cid))
    w("CIDs with at least one SMILES        : %d" % int((smi_per_cid.n_distinct > 0).sum()))
    w("CIDs with NO SMILES on any SID       : %d" % int((smi_per_cid.n_distinct == 0).sum()))
    w("CIDs whose SIDs give >1 distinct SMILES string: %d"
      % int((smi_per_cid.n_distinct > 1).sum()))
    w("")
    w("distinct-SMILES-per-CID distribution: %s"
      % dict(sorted(Counter(smi_per_cid.n_distinct).items())))
    diff = smi_per_cid[smi_per_cid.n_distinct > 1]
    if len(diff):
        w("")
        w("Examples of CIDs with differing depositor SMILES (first 4):")
        for cid in list(diff.index)[:4]:
            sub = withcid[(withcid._cid == cid) & (withcid._smi != "")]
            w("  CID %s" % cid)
            for s in sorted(set(sub._smi)):
                w("     %s" % (s[:100] + ("..." if len(s) > 100 else "")))
    uniq_smi = withcid.loc[withcid._smi != ""].drop_duplicates("_cid")
    w("")
    w("Standardization indicators, one SMILES per CID (n=%d):" % len(uniq_smi))
    w("  contains '.' (salt / mixture / multi-component) : %d"
      % int(uniq_smi._smi.str.contains(r"\.", regex=True).sum()))
    w("  contains a charge token ([+ or -])              : %d"
      % int(uniq_smi._smi.str.contains(r"\[[^\]]*[+-]", regex=True).sum()))
    w("  contains '@' (stereocentre)                     : %d"
      % int(uniq_smi._smi.str.contains("@", regex=False).sum()))
    w("  contains a metal-ish bracket token (Na/K/Ca/Fe/Zn/Cl-) : %d"
      % int(uniq_smi._smi.str.contains(
          r"\[(?:Na|K|Ca|Mg|Fe|Zn|Cu|Mn|Al|Ba|Li|Ag|Hg|Pb|Sn|Cl-|Br-|I-)", regex=True).sum()))

    # ---------------- 5. viability ----------------
    h("5. VIABILITY INFORMATION: 720637 in-row columns vs AID 720634")
    v = load(VIA, usecols=["PUBCHEM_RESULT_TAG", "PUBCHEM_SID", "PUBCHEM_CID",
                           OUT, "Phenotype-Replicate_1"])
    v = v.rename(columns={OUT: "via634_outcome"})
    w("720634 rows: %d ; unique SID: %d" % (len(v), v["PUBCHEM_SID"].nunique()))
    j = df[["PUBCHEM_SID", OUT, "Activity Summary", "Viability Activity",
            "Viability Potency (uM)", "Viability Efficacy (%)"]].merge(
        v[["PUBCHEM_SID", "via634_outcome"]], on="PUBCHEM_SID", how="left")
    w("720637 SIDs matched into 720634: %d / %d (unmatched %d)"
      % (int(j.via634_outcome.notna().sum()), len(j),
         int(j.via634_outcome.isna().sum())))
    w("")
    w("720637 'Viability Activity'  x  720634 PUBCHEM_ACTIVITY_OUTCOME:")
    ct = pd.crosstab(j["Viability Activity"].fillna("(empty)"),
                     j["via634_outcome"].fillna("(unmatched)"))
    for line in ct.to_string().split("\n"):
        w("  " + line)
    w("")
    w("Coverage of the viability fields inside AID 720637 itself:")
    for c in ["Viability Activity", "Viability Potency (uM)", "Viability Efficacy (%)"]:
        w("  %-26s non-empty in %d / %d rows"
          % (c, int((~blank(df[c])).sum()), len(df)))
    w("")
    w("MMP outcome x 720637 Viability Activity (annotation value, NOT a label change):")
    ct2 = pd.crosstab(df[OUT], df["Viability Activity"].fillna("(empty)"))
    for line in ct2.to_string().split("\n"):
        w("  " + line)

    # ---------------- compound-level summary table ----------------
    h("6. COMPOUND-LEVEL SUMMARY TABLE (counts only - NO label column)")
    smi_first = (withcid[withcid._smi != ""]
                 .drop_duplicates("_cid")
                 .set_index("_cid")["_smi"])
    summary = piv.copy()
    summary["n_distinct_smiles"] = smi_per_cid["n_distinct"]
    summary["example_smiles"] = summary.index.map(smi_first).fillna("")
    summary["n_sample_sources"] = withcid.groupby("_cid")["Sample Source"].nunique()
    summary["viability_activity_values"] = withcid.groupby("_cid")["Viability Activity"].agg(
        lambda s: "|".join(sorted(set(x for x in s.fillna("") if x))))
    summary = summary.reset_index().rename(columns={"_cid": "PUBCHEM_CID"})
    summary = summary[["PUBCHEM_CID", "n_sid", "n_active", "n_inactive",
                       "n_inconclusive", "combination", "n_distinct_smiles",
                       "n_sample_sources", "viability_activity_values",
                       "example_smiles"]]
    summary.to_csv(os.path.join(RES, "checkpoint_1C_compound_summary.csv"), index=False)
    w("Written: results/checkpoint_1C_compound_summary.csv")
    w("rows: %d   columns: %s" % (len(summary), list(summary.columns)))
    w("")
    w("NOTE: this table contains COUNTS ONLY. No label has been assigned and no")
    w("      training dataset has been created.")

    with open(os.path.join(RES, "checkpoint_1C_analysis.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Report written to results/checkpoint_1C_analysis.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
