#!/usr/bin/env python3
"""
Checkpoint 1E: build the final curated Tox21 MMP dataset.

Approved rules
--------------
CID level (Rule 1):
    >=1 Active,  0 Inactive                     -> active_consistent
    >=1 Inactive, 0 Active                      -> inactive_consistent
    >=1 Active and >=1 Inactive                 -> conflicted        (excluded)
    only Inconclusive                           -> inconclusive_only (excluded)

Parent level (full InChIKey of the standardised neutral parent):
    >=1 active_consistent CID, no inactive_consistent CID, no conflicted CID
                                                -> label 1
    >=1 inactive_consistent CID, no active_consistent CID, no conflicted CID
                                                -> label 0
    any Active/Inactive disagreement            -> excluded, flagged
    inconclusive-only                           -> excluded, flagged

Structure filter (Option B):
    parent SMILES that does not re-parse        -> excluded
    parent with no carbon (inorganic)           -> excluded
    metal-removed and true-mixture compounds are KEPT, flagged only.

No fingerprints, no descriptors, no models, no splits. Raw files are read only.

Usage:  py -3 scripts/build_tox21_dataset.py
Outputs:
    data/processed/tox21_mmp_observations.csv   SID level, every raw observation
    data/processed/tox21_mmp_compounds.csv      CID level, all 8099
    data/processed/tox21_mmp_training.csv       parent level, ML-ready, labelled
    data/processed/tox21_mmp_excluded.csv       parent level, excluded, unlabelled
    results/checkpoint_1E_accounting.txt        the ledger
"""
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

RAW = os.path.join("data", "external", "tox21_raw")
PROC = os.path.join("data", "processed")
RES = "results"
OUTCOME = "PUBCHEM_ACTIVITY_OUTCOME"
SMI = "PUBCHEM_EXT_DATASOURCE_SMILES"
SOURCE = "Tox21 qHTS mitochondrial membrane potential, PubChem AID 720637"
_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def datarows(fname, usecols=None):
    df = pd.read_csv(os.path.join(RAW, fname), dtype=str, low_memory=False, usecols=usecols)
    return df[df["PUBCHEM_RESULT_TAG"].fillna("").str.match(r"^\d+$")].copy()


def main():
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(RES, exist_ok=True)

    # ---------- 1. SID-level observations (nothing dropped) ----------
    mmp = datarows("aid720637_datatable_pcget.csv")
    via = datarows("aid720634_datatable_pcget.csv",
                   usecols=["PUBCHEM_RESULT_TAG", "PUBCHEM_SID", OUTCOME])
    via = via.rename(columns={OUTCOME: "viability_outcome_aid720634"})[
        ["PUBCHEM_SID", "viability_outcome_aid720634"]]

    obs = mmp.rename(columns={
        "PUBCHEM_SID": "pubchem_sid", "PUBCHEM_CID": "pubchem_cid",
        SMI: "depositor_smiles", OUTCOME: "activity_outcome",
        "PUBCHEM_ACTIVITY_SCORE": "activity_score",
        "Activity Summary": "activity_summary",
        "Ratio Activity": "ratio_activity",
        "Ratio Potency (uM)": "ratio_potency_um",
        "Ratio Efficacy (%)": "ratio_efficacy_pct",
        "Viability Activity": "viability_activity",
        "Viability Potency (uM)": "viability_potency_um",
        "Viability Efficacy (%)": "viability_efficacy_pct",
        "Sample Source": "sample_source"})[[
        "pubchem_sid", "pubchem_cid", "depositor_smiles", "activity_outcome",
        "activity_score", "activity_summary", "ratio_activity", "ratio_potency_um",
        "ratio_efficacy_pct", "viability_activity", "viability_potency_um",
        "viability_efficacy_pct", "sample_source"]]
    obs = obs.merge(via, left_on="pubchem_sid", right_on="PUBCHEM_SID", how="left") \
             .drop(columns=["PUBCHEM_SID"])
    obs["source_aid"] = "720637"
    obs["pubchem_cid"] = obs.pubchem_cid.fillna("").str.strip()
    obs["depositor_smiles"] = obs.depositor_smiles.fillna("").str.strip()
    obs["has_structure"] = obs.depositor_smiles != ""
    obs.to_csv(os.path.join(PROC, "tox21_mmp_observations.csv"), index=False)

    n_sid = len(obs)
    n_nostruct = int((~obs.has_structure).sum())

    # ---------- 2. CID level, from the 1D structure table ----------
    st = pd.read_csv(os.path.join(RES, "checkpoint_1D_structures.csv"), dtype=str)
    for c in ["n_sid", "n_active", "n_inactive", "n_inconclusive"]:
        st[c] = pd.to_numeric(st[c])
    for c in ["parent_parses", "parent_has_carbon", "metal_removed", "flag_true_mixture"]:
        st[c] = st[c].astype(str).str.lower().isin(["true", "1"])
    st = st.rename(columns={
        "parent_neutral_smiles": "parent_smiles",
        "parent_neutral_inchikey": "parent_inchikey",
        "metal_removed": "flag_metal_removed"})
    st["flag_inorganic_parent"] = ~st.parent_has_carbon & st.parent_parses
    st["flag_parent_unparseable"] = ~st.parent_parses

    # ---------- 3. parent level ----------
    g = st.groupby("parent_inchikey")
    par = pd.DataFrame({
        "parent_smiles": g.parent_smiles.first(),
        "n_cids": g.size(),
        "n_active_consistent_cids": g.cid_status.apply(lambda s: (s == "active_consistent").sum()),
        "n_inactive_consistent_cids": g.cid_status.apply(lambda s: (s == "inactive_consistent").sum()),
        "n_conflicted_cids": g.cid_status.apply(lambda s: (s == "conflicted").sum()),
        "n_inconclusive_only_cids": g.cid_status.apply(lambda s: (s == "inconclusive_only").sum()),
        "n_sids": g.n_sid.sum(),
        "n_active_sids": g.n_active.sum(),
        "n_inactive_sids": g.n_inactive.sum(),
        "n_inconclusive_sids": g.n_inconclusive.sum(),
        "pubchem_cids": g.pubchem_cid.apply(lambda s: "|".join(sorted(s))),
        "depositor_smiles": g.depositor_smiles.apply(lambda s: "|".join(sorted(set(s)))),
        "flag_metal_removed": g.flag_metal_removed.any(),
        "flag_true_mixture": g.flag_true_mixture.any(),
        "flag_inorganic_parent": g.flag_inorganic_parent.any(),
        "flag_parent_unparseable": g.flag_parent_unparseable.any(),
    })

    def status(r):
        if r.n_conflicted_cids > 0:
            return "conflicted_inherited"
        if r.n_active_consistent_cids > 0 and r.n_inactive_consistent_cids > 0:
            return "conflicted_new_at_parent"
        if r.n_active_consistent_cids > 0:
            return "active_consistent"
        if r.n_inactive_consistent_cids > 0:
            return "inactive_consistent"
        return "inconclusive_only"

    par["parent_status"] = par.apply(status, axis=1)

    def reason(r):
        if r.parent_status == "conflicted_inherited":
            return "conflict_cid_level"
        if r.parent_status == "conflicted_new_at_parent":
            return "conflict_parent_level"
        if r.parent_status == "inconclusive_only":
            return "inconclusive_only"
        if r.flag_parent_unparseable:
            return "parent_unparseable"
        if r.flag_inorganic_parent:
            return "inorganic_parent"
        return ""

    par["exclusion_reason"] = par.apply(reason, axis=1)
    par["in_training"] = par.exclusion_reason == ""
    par["label"] = pd.NA
    par.loc[par.in_training & (par.parent_status == "active_consistent"), "label"] = 1
    par.loc[par.in_training & (par.parent_status == "inactive_consistent"), "label"] = 0
    par["source"] = SOURCE
    par["endpoint"] = "mitochondrial membrane potential decrease (depolarisation)"
    par["label_definition"] = ("1 = PubChem Activity Outcome 'Active' "
                               "(= Activity Summary 'active antagonist'); "
                               "0 = 'Inactive'; Inconclusive excluded")
    par = par.reset_index()

    cols = ["parent_inchikey", "parent_smiles", "label", "parent_status",
            "in_training", "exclusion_reason", "n_cids", "pubchem_cids",
            "depositor_smiles", "n_sids", "n_active_sids", "n_inactive_sids",
            "n_inconclusive_sids", "n_active_consistent_cids",
            "n_inactive_consistent_cids", "n_conflicted_cids",
            "n_inconclusive_only_cids", "flag_metal_removed", "flag_true_mixture",
            "flag_inorganic_parent", "flag_parent_unparseable",
            "endpoint", "label_definition", "source"]
    train = par[par.in_training][cols].copy()
    train["label"] = train.label.astype(int)
    excl = par[~par.in_training][[c for c in cols if c != "label"]].copy()
    train.to_csv(os.path.join(PROC, "tox21_mmp_training.csv"), index=False)
    excl.to_csv(os.path.join(PROC, "tox21_mmp_excluded.csv"), index=False)

    # CID-level table, carrying the parent verdict back down
    pmap = par.set_index("parent_inchikey")
    cid = st.copy()
    cid["parent_status"] = cid.parent_inchikey.map(pmap.parent_status)
    cid["in_training"] = cid.parent_inchikey.map(pmap.in_training)
    cid["exclusion_reason"] = cid.parent_inchikey.map(pmap.exclusion_reason)
    cid["label"] = cid.parent_inchikey.map(pmap.label)
    cid["source"] = SOURCE
    cidcols = ["pubchem_cid", "depositor_smiles", "parent_smiles", "parent_inchikey",
               "cid_status", "parent_status", "label", "in_training",
               "exclusion_reason", "n_sid", "n_active", "n_inactive", "n_inconclusive",
               "n_components", "n_organic_components", "net_charge", "n_stereocentres",
               "n_unassigned_stereo", "n_bond_stereo", "parent_net_charge",
               "parent_heavy_atoms", "flag_metal_removed", "flag_true_mixture",
               "flag_inorganic_parent", "flag_parent_unparseable", "source"]
    cid[cidcols].to_csv(os.path.join(PROC, "tox21_mmp_compounds.csv"), index=False)

    # ---------- 4. accounting ----------
    w("=" * 78)
    w("CHECKPOINT 1E - DATASET ACCOUNTING")
    w("=" * 78)
    w("")
    w("SAMPLE (SID) LEDGER")
    w("  raw Tox21 samples in AID 720637            : %6d" % n_sid)
    w("  samples with no CID and no SMILES          : %6d  (kept in observations file)" % n_nostruct)
    w("  samples mapping to a CID                   : %6d" % (n_sid - n_nostruct))
    w("")
    w("COMPOUND (CID) LEDGER  -- must reconcile to 8099")
    start = len(cid)
    w("  unique CIDs                                : %6d" % start)
    buckets = []
    b1 = int((cid.cid_status == "conflicted").sum())
    b2 = int((cid.cid_status == "inconclusive_only").sum())
    cons = cid[cid.cid_status.isin(["active_consistent", "inactive_consistent"])]
    b3 = int((cons.exclusion_reason == "conflict_cid_level").sum())
    b4 = int((cons.exclusion_reason == "conflict_parent_level").sum())
    b5 = int((cons.exclusion_reason == "parent_unparseable").sum())
    b6 = int((cons.exclusion_reason == "inorganic_parent").sum())
    keep = cons[cons.in_training == True]
    b7 = int((keep.label == 1).sum())
    b8 = int((keep.label == 0).sum())
    for name, n in [("- CID-level conflicts (Active & Inactive)", b1),
                    ("- inconclusive-only CIDs", b2),
                    ("- consistent CIDs absorbed into a CID-level-conflicted parent", b3),
                    ("- consistent CIDs in a NEW parent-level conflict", b4),
                    ("- consistent CIDs with an unparseable parent", b5),
                    ("- consistent CIDs with an inorganic parent", b6),
                    ("= final POSITIVE CIDs", b7),
                    ("= final NEGATIVE CIDs", b8)]:
        w("  %-58s : %6d" % (name, n))
        buckets.append(n)
    w("  %-58s : %6d" % ("RECONCILIATION total", sum(buckets)))
    w("  %-58s : %s" % ("matches 8099?", "YES" if sum(buckets) == start else "NO"))
    w("")
    w("PARENT-STRUCTURE LEDGER")
    w("  unique parent structures (full InChIKey)   : %6d" % len(par))
    for name, n in [("- conflicted, inherited from CID level",
                     int((par.exclusion_reason == "conflict_cid_level").sum())),
                    ("- conflicted, created by the parent collapse",
                     int((par.exclusion_reason == "conflict_parent_level").sum())),
                    ("- inconclusive only",
                     int((par.exclusion_reason == "inconclusive_only").sum())),
                    ("- unparseable parent structure",
                     int((par.exclusion_reason == "parent_unparseable").sum())),
                    ("- inorganic / carbon-free parent",
                     int((par.exclusion_reason == "inorganic_parent").sum())),
                    ("= FINAL TRAINING parents, label 1",
                     int(((par.label == 1)).sum())),
                    ("= FINAL TRAINING parents, label 0",
                     int(((par.label == 0)).sum()))]:
        w("  %-58s : %6d" % (name, n))
    tot = int((par.exclusion_reason != "").sum()) + int(par.in_training.sum())
    w("  %-58s : %6d" % ("RECONCILIATION total", tot))
    w("  %-58s : %s" % ("matches parent count?", "YES" if tot == len(par) else "NO"))
    w("")
    w("FINAL TRAINING SET")
    w("  parents (modelling units)                  : %6d" % len(train))
    w("  positives (label 1)                        : %6d" % int((train.label == 1).sum()))
    w("  negatives (label 0)                        : %6d" % int((train.label == 0).sum()))
    w("  prevalence                                 : %6.2f%%"
      % (100.0 * (train.label == 1).sum() / len(train)))
    w("  underlying CIDs                            : %6d" % int(train.n_cids.sum()))
    w("  underlying SIDs                            : %6d" % int(train.n_sids.sum()))
    w("")
    w("STRUCTURE QUALITY OF THE FINAL TRAINING SET")
    ok = train.parent_smiles.map(lambda s: Chem.MolFromSmiles(s) is not None)
    w("  parents whose SMILES re-parses in RDKit    : %6d / %d" % (int(ok.sum()), len(train)))
    for f in ["flag_metal_removed", "flag_true_mixture",
              "flag_inorganic_parent", "flag_parent_unparseable"]:
        sub = train[train[f]]
        w("  %-40s   : %6d  (pos %d / neg %d)"
          % (f, len(sub), int((sub.label == 1).sum()), int((sub.label == 0).sum())))
    anyflag = train[train.flag_metal_removed | train.flag_true_mixture]
    w("  %-40s   : %6d" % ("any structure-quality flag", len(anyflag)))
    w("")
    w("EXCLUDED TABLE")
    w("  parents                                    : %6d" % len(excl))
    for k, v in excl.exclusion_reason.value_counts().items():
        w("    %-40s : %6d" % (k, v))
    w("  none of these rows carries a label column.")

    with open(os.path.join(RES, "checkpoint_1E_accounting.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Written: data/processed/tox21_mmp_{observations,compounds,training,excluded}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
