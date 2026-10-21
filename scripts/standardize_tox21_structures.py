#!/usr/bin/env python3
"""
Checkpoint 1D: structure diagnostics, RDKit standardisation and a SECOND
conflict analysis at the standardised parent level.

ANALYSIS ONLY. Raw files are never written to. The depositor SMILES is preserved
verbatim in every output. No fingerprints, no models, no splits.

Two standardised variants are produced side by side so the effect of
neutralisation can be judged rather than assumed:
    parent_charged  = normalise -> disconnect metals -> largest organic fragment
    parent_neutral  = the above, then Uncharger

Stereochemistry is RETAINED throughout (isomericSmiles=True, no stereo removal).

Usage:  py -3 scripts/standardize_tox21_structures.py
Outputs (results/):
    checkpoint_1D_standardisation.txt
    checkpoint_1D_structures.csv          one row per CID
    checkpoint_1D_parent_groups.csv       one row per parent InChIKey
    checkpoint_1D_parent_conflicts.csv    CIDs inside conflicted parents
"""
import os
import sys
from collections import Counter, defaultdict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

RAW = os.path.join("data", "external", "tox21_raw")
RES = "results"
MMP = "aid720637_datatable_pcget.csv"
OUTCOME = "PUBCHEM_ACTIVITY_OUTCOME"
SMI = "PUBCHEM_EXT_DATASOURCE_SMILES"
METALS = set("Li Na K Rb Cs Be Mg Ca Sr Ba Al Fe Zn Cu Mn Ni Co Cr Cd Hg Pb Sn "
             "Ag Au Pt Pd Ti V Se As Sb Bi Zr Mo W Gd La Ce".split())
_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def h(t):
    w("")
    w("=" * 78)
    w(t)
    w("=" * 78)


def load_cid_level():
    """Rebuild the CID-level table from raw, applying the approved Rule 1."""
    df = pd.read_csv(os.path.join(RAW, MMP), dtype=str, low_memory=False)
    df = df[df["PUBCHEM_RESULT_TAG"].fillna("").str.match(r"^\d+$")].copy()
    df["cid"] = df["PUBCHEM_CID"].fillna("").str.strip()
    df["smi"] = df[SMI].fillna("").str.strip()
    df = df[df.cid != ""]
    piv = (df.assign(o=1).pivot_table(index="cid", columns=OUTCOME, values="o",
                                      aggfunc="sum", fill_value=0))
    for c in ["Active", "Inactive", "Inconclusive"]:
        if c not in piv.columns:
            piv[c] = 0
    piv = piv[["Active", "Inactive", "Inconclusive"]]
    piv.columns = ["n_active", "n_inactive", "n_inconclusive"]

    def status(r):
        if r.n_active > 0 and r.n_inactive > 0:
            return "conflicted"
        if r.n_active > 0:
            return "active_consistent"
        if r.n_inactive > 0:
            return "inactive_consistent"
        return "inconclusive_only"

    piv["cid_status"] = piv.apply(status, axis=1)
    piv["n_sid"] = piv[["n_active", "n_inactive", "n_inconclusive"]].sum(axis=1)
    smi = df.drop_duplicates("cid").set_index("cid")["smi"]
    piv["depositor_smiles"] = piv.index.map(smi)
    return piv.reset_index().rename(columns={"cid": "pubchem_cid"})


# ------------------------- diagnostics -------------------------

def diagnose(smiles):
    """Facts about the depositor structure. No transformation."""
    d = {"parsed": False, "parse_sanitized": False, "n_components": 0,
         "net_charge": None, "n_charged_atoms": 0, "has_metal": False,
         "has_carbon": False, "n_stereocentres": 0, "n_unassigned_stereo": 0,
         "n_bond_stereo": 0, "heavy_atoms": 0, "n_organic_components": 0,
         "largest_tie": False, "elements": ""}
    m = Chem.MolFromSmiles(smiles, sanitize=True)
    if m is not None:
        d["parsed"] = True
        d["parse_sanitized"] = True
    else:
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        if m is None:
            return d
        d["parsed"] = True
        try:
            Chem.SanitizeMol(m)
            d["parse_sanitized"] = True
        except Exception:
            return d
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    d["elements"] = ".".join(sorted(set(syms)))
    d["has_metal"] = any(s in METALS for s in syms)
    d["has_carbon"] = "C" in syms
    d["heavy_atoms"] = m.GetNumHeavyAtoms()
    d["net_charge"] = Chem.GetFormalCharge(m)
    d["n_charged_atoms"] = sum(1 for a in m.GetAtoms() if a.GetFormalCharge() != 0)
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    d["n_components"] = len(frags)
    sizes = sorted((f.GetNumHeavyAtoms() for f in frags), reverse=True)
    if len(sizes) > 1 and sizes[0] == sizes[1]:
        d["largest_tie"] = True
    d["n_organic_components"] = sum(
        1 for f in frags
        if any(a.GetSymbol() == "C" for a in f.GetAtoms()) and f.GetNumHeavyAtoms() >= 5)
    try:
        centres = Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                            useLegacyImplementation=False)
        d["n_stereocentres"] = len(centres)
        d["n_unassigned_stereo"] = sum(1 for _, v in centres if v == "?")
    except Exception:
        pass
    d["n_bond_stereo"] = sum(1 for b in m.GetBonds()
                             if b.GetStereo() != Chem.BondStereo.STEREONONE)
    return d


# ------------------------- standardisation -------------------------

_normalizer = rdMolStandardize.Normalizer()
_metal = rdMolStandardize.MetalDisconnector()
_largest = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()


def standardise(smiles):
    """Returns dict with both variants. Stereochemistry is never stripped."""
    out = {"std_ok": False, "fail_reason": "", "parent_charged_smiles": "",
           "parent_neutral_smiles": "", "parent_charged_inchikey": "",
           "parent_neutral_inchikey": "", "changed_by_metal_disconnect": False,
           "changed_by_fragment_choice": False, "changed_by_uncharge": False,
           "parent_net_charge": None, "parent_heavy_atoms": 0}
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        out["fail_reason"] = "unparseable / failed sanitisation"
        return out
    try:
        before = Chem.MolToSmiles(m)
        m = _normalizer.normalize(m)
        m2 = _metal.Disconnect(m)
        out["changed_by_metal_disconnect"] = Chem.MolToSmiles(m2) != Chem.MolToSmiles(m)
        m = m2
        m3 = _largest.choose(m)
        out["changed_by_fragment_choice"] = Chem.MolToSmiles(m3) != Chem.MolToSmiles(m)
        m = m3
        charged = Chem.MolToSmiles(m, isomericSmiles=True)
        out["parent_charged_smiles"] = charged
        out["parent_charged_inchikey"] = Chem.MolToInchiKey(m) or ""
        mn = _uncharger.uncharge(Chem.Mol(m))
        neutral = Chem.MolToSmiles(mn, isomericSmiles=True)
        out["changed_by_uncharge"] = neutral != charged
        out["parent_neutral_smiles"] = neutral
        out["parent_neutral_inchikey"] = Chem.MolToInchiKey(mn) or ""
        out["parent_net_charge"] = Chem.GetFormalCharge(mn)
        out["parent_heavy_atoms"] = mn.GetNumHeavyAtoms()
        out["std_ok"] = bool(out["parent_neutral_inchikey"])
        if not out["std_ok"]:
            out["fail_reason"] = "InChIKey generation failed"
        _ = before
    except Exception as exc:
        out["fail_reason"] = "standardisation error: %s" % exc
    return out


def combo_of(statuses):
    parts = []
    if "active_consistent" in statuses or "A" in statuses:
        parts.append("Active")
    return parts


def main():
    os.makedirs(RES, exist_ok=True)
    cid = load_cid_level()
    w("CID-level input (Rule 1 applied): %d compounds" % len(cid))
    w("  %s" % dict(Counter(cid.cid_status)))

    # -------- Phase A: diagnostics on the untouched depositor SMILES --------
    h("PHASE A. DIAGNOSTICS ON THE DEPOSITOR SMILES (no transformation applied)")
    diag = pd.DataFrame([diagnose(s) for s in cid.depositor_smiles])
    d = pd.concat([cid.reset_index(drop=True), diag], axis=1)

    w("parseable by RDKit                : %d / %d" % (int(d.parsed.sum()), len(d)))
    w("parse + sanitise cleanly          : %d / %d" % (int(d.parse_sanitized.sum()), len(d)))
    bad = d[~d.parse_sanitized]
    w("FAILED                            : %d" % len(bad))
    for _, r in bad.head(15).iterrows():
        w("    CID %-12s %s" % (r.pubchem_cid, r.depositor_smiles[:90]))

    ok = d[d.parse_sanitized].copy()
    w("")
    w("--- disconnected components ---")
    w("component-count distribution: %s" % dict(sorted(Counter(ok.n_components).items())))
    multi = ok[ok.n_components > 1]
    w("multi-component structures        : %d (%.1f%%)"
      % (len(multi), 100.0 * len(multi) / len(ok)))
    w("  ... where 2+ components are organic and >=5 heavy atoms : %d"
      % int((multi.n_organic_components >= 2).sum()))
    w("      ^ these are TRUE MIXTURES / co-crystals, not simple salts;")
    w("        taking the largest fragment discards a real chemical entity.")
    w("  ... where the two largest fragments tie on heavy-atom count: %d"
      % int(multi.largest_tie.sum()))
    w("")
    w("  examples of multi-organic-component structures (first 8):")
    for _, r in multi[multi.n_organic_components >= 2].head(8).iterrows():
        w("    CID %-10s %s" % (r.pubchem_cid, r.depositor_smiles[:95]))

    w("")
    w("--- charges ---")
    w("structures with >=1 charged atom  : %d (%.1f%%)"
      % (int((ok.n_charged_atoms > 0).sum()), 100.0 * (ok.n_charged_atoms > 0).sum() / len(ok)))
    w("net formal charge distribution    : %s"
      % dict(sorted(Counter(ok.net_charge).items())))
    w("net charge != 0                   : %d" % int((ok.net_charge != 0).sum()))
    w("charged but net charge == 0 (zwitterion / balanced salt): %d"
      % int(((ok.n_charged_atoms > 0) & (ok.net_charge == 0)).sum()))

    w("")
    w("--- metals and elements ---")
    w("structures containing a metal     : %d" % int(ok.has_metal.sum()))
    w("structures with NO carbon (inorganic): %d" % int((~ok.has_carbon).sum()))
    for _, r in ok[~ok.has_carbon].head(10).iterrows():
        w("    CID %-10s %-28s %s" % (r.pubchem_cid, r.depositor_smiles[:28], r.elements))

    w("")
    w("--- stereochemistry ---")
    w("structures with >=1 stereocentre  : %d" % int((ok.n_stereocentres > 0).sum()))
    w("  total stereocentres             : %d" % int(ok.n_stereocentres.sum()))
    w("  with >=1 UNASSIGNED stereocentre: %d" % int((ok.n_unassigned_stereo > 0).sum()))
    w("structures with double-bond stereo: %d" % int((ok.n_bond_stereo > 0).sum()))

    # -------- Phase B: standardisation --------
    h("PHASE B. STANDARDISATION")
    w("Pipeline, in order. Every step is RDKit rdMolStandardize; nothing bespoke.")
    w("  1. MolFromSmiles + sanitise")
    w("  2. Normalizer()            functional-group normalisation (e.g. nitro forms)")
    w("  3. MetalDisconnector()     break covalent metal-ligand bonds into ions")
    w("  4. LargestFragmentChooser(preferOrganic=True)   keep the parent fragment")
    w("  5a. -> parent_charged      canonical isomeric SMILES + InChIKey")
    w("  5b. Uncharger() -> parent_neutral   canonical isomeric SMILES + InChIKey")
    w("  isomericSmiles=True throughout: stereochemistry is never stripped.")
    w("")
    std = pd.DataFrame([standardise(s) for s in d.depositor_smiles])
    d = pd.concat([d, std], axis=1)
    w("standardised successfully         : %d / %d" % (int(d.std_ok.sum()), len(d)))
    fails = d[~d.std_ok]
    w("failed                            : %d" % len(fails))
    for _, r in fails.head(15).iterrows():
        w("    CID %-12s %-18s %s"
          % (r.pubchem_cid, r.fail_reason[:18], r.depositor_smiles[:60]))

    g = d[d.std_ok]
    w("")
    w("CIDs affected by each step:")
    w("  metal disconnection changed the molecule : %d" % int(g.changed_by_metal_disconnect.sum()))
    w("  largest-fragment choice changed it       : %d" % int(g.changed_by_fragment_choice.sum()))
    w("  Uncharger changed it                     : %d" % int(g.changed_by_uncharge.sum()))
    w("  any change vs depositor SMILES           : %d"
      % int((g.parent_neutral_smiles != g.depositor_smiles).sum()))
    w("")
    w("Residual net charge AFTER Uncharger (permanently charged species):")
    w("  %s" % dict(sorted(Counter(g.parent_net_charge).items())))
    perm = g[g.parent_net_charge != 0]
    w("  CIDs still charged after uncharging: %d  (quaternary N, N-oxides etc.)" % len(perm))
    for _, r in perm.head(8).iterrows():
        w("    CID %-10s charge %+d  %s"
          % (r.pubchem_cid, r.parent_net_charge, r.parent_neutral_smiles[:70]))

    w("")
    w("Stereochemistry retention check:")
    had = g[g.n_stereocentres > 0]
    kept = had.parent_neutral_smiles.str.contains("@", regex=False).sum()
    w("  CIDs with assigned stereocentres in depositor SMILES : %d"
      % int((had.n_stereocentres - had.n_unassigned_stereo > 0).sum()))
    w("  CIDs whose parent_neutral SMILES still contains '@'  : %d" % int(kept))
    bond_had = g[g.n_bond_stereo > 0]
    bond_kept = bond_had.parent_neutral_smiles.str.contains("/|\\\\", regex=True).sum()
    w("  CIDs with double-bond stereo / retained in parent    : %d / %d"
      % (int(bond_kept), len(bond_had)))
    nonstd_key = g.parent_neutral_inchikey.str[15:25]
    w("  InChIKeys with a non-standard-stereo second block    : %d"
      % int((nonstd_key != "UHFFFAOYSA").sum()))

    # -------- Phase C: parent grouping + second conflict analysis --------
    h("PHASE C. PARENT GROUPING AND SECOND CONFLICT ANALYSIS")
    for variant, key in [("parent_neutral", "parent_neutral_inchikey"),
                         ("parent_charged", "parent_charged_inchikey")]:
        n_groups = g[key].nunique()
        w("%-16s -> %d distinct full InChIKeys from %d CIDs (collapse of %d)"
          % (variant, n_groups, len(g), len(g) - n_groups))
    w("")
    w("connectivity-only check (InChIKey first block, stereo ignored):")
    w("  %d skeletons -- the %d extra groups under the full key are stereoisomer pairs"
      % (g.parent_neutral_inchikey.str[:14].nunique(),
         g.parent_neutral_inchikey.nunique() - g.parent_neutral_inchikey.str[:14].nunique()))
    w("  (reported for information only; the FULL InChIKey is the grouping key,")
    w("   so stereoisomers stay separate compounds.)")

    key = "parent_neutral_inchikey"
    grp = g.groupby(key)
    parent = pd.DataFrame({
        "n_cids": grp.size(),
        "n_active_consistent": grp.cid_status.apply(lambda s: (s == "active_consistent").sum()),
        "n_inactive_consistent": grp.cid_status.apply(lambda s: (s == "inactive_consistent").sum()),
        "n_conflicted_cid": grp.cid_status.apply(lambda s: (s == "conflicted").sum()),
        "n_inconclusive_only": grp.cid_status.apply(lambda s: (s == "inconclusive_only").sum()),
        "n_sid": grp.n_sid.sum(),
        "sid_active": grp.n_active.sum(),
        "sid_inactive": grp.n_inactive.sum(),
        "sid_inconclusive": grp.n_inconclusive.sum(),
        "cids": grp.pubchem_cid.apply(lambda s: "|".join(sorted(s))),
        "parent_smiles": grp.parent_neutral_smiles.first(),
    })

    def pstatus(r):
        a = r.n_active_consistent > 0 or r.n_conflicted_cid > 0
        i = r.n_inactive_consistent > 0 or r.n_conflicted_cid > 0
        if r.n_conflicted_cid > 0:
            return "conflicted (inherited)"
        if r.n_active_consistent > 0 and r.n_inactive_consistent > 0:
            return "conflicted (new at parent level)"
        if r.n_active_consistent > 0:
            return "active_consistent"
        if r.n_inactive_consistent > 0:
            return "inactive_consistent"
        return "inconclusive_only"

    parent["parent_status"] = parent.apply(pstatus, axis=1)

    w("")
    w("collapse size distribution (CIDs per parent):")
    for k, v in sorted(Counter(parent.n_cids).items()):
        w("   %2d CID(s): %6d parents" % (k, v))

    w("")
    w("PARENT-LEVEL STATUS:")
    w("  %-36s %8s %8s" % ("status", "parents", "CIDs"))
    w("  " + "-" * 56)
    for k in ["active_consistent", "inactive_consistent",
              "conflicted (new at parent level)", "conflicted (inherited)",
              "inconclusive_only"]:
        sub = parent[parent.parent_status == k]
        w("  %-36s %8d %8d" % (k, len(sub), int(sub.n_cids.sum())))
    w("  " + "-" * 56)
    w("  %-36s %8d %8d" % ("TOTAL", len(parent), int(parent.n_cids.sum())))

    newc = parent[parent.parent_status == "conflicted (new at parent level)"]
    w("")
    w("NEWLY CREATED conflicts (CIDs individually consistent, disagreeing once merged): %d"
      % len(newc))
    for k, v in sorted(Counter(zip(newc.n_active_consistent, newc.n_inactive_consistent)).items()):
        w("   %dA : %dI  -> %d parents" % (k[0], k[1], v))

    w("")
    w("Examples of newly conflicted parents (first 10):")
    for ik, r in newc.head(10).iterrows():
        w("  %s  %s" % (ik, r.parent_smiles[:70]))
        for c in r.cids.split("|"):
            row = g[g.pubchem_cid == c].iloc[0]
            w("     CID %-10s %-20s depositor: %s"
              % (c, row.cid_status, row.depositor_smiles[:60]))

    w("")
    w("Examples of salt / free-parent collapses that did NOT create a conflict (first 8):")
    benign = parent[(parent.n_cids > 1) &
                    (parent.parent_status.isin(["active_consistent", "inactive_consistent"]))]
    for ik, r in benign.head(8).iterrows():
        w("  %s  %s  [%s]" % (ik, r.parent_smiles[:60], r.parent_status))
        for c in r.cids.split("|"):
            row = g[g.pubchem_cid == c].iloc[0]
            w("     CID %-10s %s" % (c, row.depositor_smiles[:70]))

    w("")
    w("CANDIDATE COUNTS AFTER PARENT-LEVEL ANALYSIS (provisional, pending approval):")
    pos = int((parent.parent_status == "active_consistent").sum())
    neg = int((parent.parent_status == "inactive_consistent").sum())
    w("  candidate positives (parent active_consistent)   : %d" % pos)
    w("  candidate negatives (parent inactive_consistent) : %d" % neg)
    w("  excluded - conflicted (new)                      : %d" % len(newc))
    w("  excluded - conflicted (inherited from CID level)  : %d"
      % int((parent.parent_status == "conflicted (inherited)").sum()))
    w("  excluded - inconclusive only                     : %d"
      % int((parent.parent_status == "inconclusive_only").sum()))
    w("  candidate training set                           : %d  (prevalence %.2f%%)"
      % (pos + neg, 100.0 * pos / max(1, pos + neg)))

    # -------- Phase D: structure-quality flags on the standardised parent --------
    h("PHASE D. STRUCTURE-QUALITY FLAGS ON THE STANDARDISED PARENT")

    def parent_ok(s):
        return Chem.MolFromSmiles(s) is not None if isinstance(s, str) and s else False

    def parent_has_carbon(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        return bool(m) and any(a.GetSymbol() == "C" for a in m.GetAtoms())

    def metals_in(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        return set(a.GetSymbol() for a in m.GetAtoms()) & METALS if m else set()

    d["parent_parses"] = d.parent_neutral_smiles.map(parent_ok)
    d["parent_has_carbon"] = d.parent_neutral_smiles.map(parent_has_carbon)
    d["metal_removed"] = [len(metals_in(a)) > 0 and len(metals_in(b)) == 0
                          for a, b in zip(d.depositor_smiles, d.parent_neutral_smiles)]
    d["flag_true_mixture"] = d.n_organic_components >= 2

    w("parent SMILES fails to re-parse      : %d CIDs (%d distinct parents)"
      % (int((~d.parent_parses).sum()),
         d.loc[~d.parent_parses, "parent_neutral_inchikey"].nunique()))
    for s, k in Counter(d.loc[~d.parent_parses, "parent_neutral_smiles"]).most_common(5):
        w("    %r x %d" % (s, k))
    w("parent parses but has NO carbon      : %d CIDs (%d distinct parents)"
      % (int((d.parent_parses & ~d.parent_has_carbon).sum()),
         d.loc[d.parent_parses & ~d.parent_has_carbon, "parent_neutral_inchikey"].nunique()))
    w("metal present in depositor, absent from parent : %d CIDs" % int(d.metal_removed.sum()))
    w("depositor was a true multi-organic mixture     : %d CIDs" % int(d.flag_true_mixture.sum()))

    # -------- outputs --------
    cols = ["pubchem_cid", "depositor_smiles", "cid_status", "n_sid", "n_active",
            "n_inactive", "n_inconclusive", "parsed", "parse_sanitized",
            "n_components", "n_organic_components", "net_charge", "n_charged_atoms",
            "has_metal", "has_carbon", "n_stereocentres", "n_unassigned_stereo",
            "n_bond_stereo", "std_ok", "fail_reason", "changed_by_metal_disconnect",
            "changed_by_fragment_choice", "changed_by_uncharge",
            "parent_charged_smiles", "parent_charged_inchikey",
            "parent_neutral_smiles", "parent_neutral_inchikey",
            "parent_net_charge", "parent_heavy_atoms", "parent_parses",
            "parent_has_carbon", "metal_removed", "flag_true_mixture"]
    d[cols].to_csv(os.path.join(RES, "checkpoint_1D_structures.csv"), index=False)
    parent.reset_index().to_csv(os.path.join(RES, "checkpoint_1D_parent_groups.csv"), index=False)
    conf_keys = set(parent[parent.parent_status.str.startswith("conflicted")].index)
    gd = d[d.std_ok]
    gd[gd[key].isin(conf_keys)].sort_values(key)[cols].to_csv(
        os.path.join(RES, "checkpoint_1D_parent_conflicts.csv"), index=False)

    with open(os.path.join(RES, "checkpoint_1D_standardisation.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")
    print("")
    print("Reports written to results/checkpoint_1D_*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
