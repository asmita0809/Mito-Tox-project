#!/usr/bin/env python3
"""
Screen EU-authorised food additives with the frozen model.

The model is used exactly as frozen. Nothing here retrains, re-selects or
re-thresholds anything: the additives are an independent prediction set, not
data.

Every compound that drops out is counted and given a reason, so the final table
can be reconciled against the Union list.

Output is a PRIORITISATION. It is not evidence of toxicity, and the column names
and the report both say so.

Usage:  py -3 scripts/screen_additives.py
Input:  data/external/efsa_additives_resolved.csv
Output: results/tables/efsa_additive_screen.csv
        results/efsa_screen_accounting.txt
        results/figures/efsa_screen.png
"""
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from mitotox import config
from mitotox.predict import MitoToxPredictor
from mitotox.additives import (SINGLE_SUBSTANCE, classify_substance,
                               inorganic_carbon_reason, is_inorganic_carbon)
from mitotox.interpretation import compile_alerts, match_alerts
from mitotox.standardize import standardize

RDLogger.DisableLog("rdApp.*")
IN_CSV = os.path.join("data", "external", "efsa_additives_resolved.csv")
OUT_CSV = os.path.join("results", "tables", "efsa_additive_screen.csv")
REPORT = os.path.join("results", "efsa_screen_accounting.txt")
FIG = os.path.join("results", "figures", "efsa_screen.png")

SURFACE, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
_lines = []


def w(s=""):
    print(s)
    _lines.append(s)


def main():
    if not os.path.exists(IN_CSV):
        sys.stderr.write("\nNot found: %s\nRun scripts/resolve_additive_structures.py "
                         "first.\n\n" % IN_CSV)
        return 1
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)

    res = pd.read_csv(IN_CSV)
    res["validated"] = res.validated.astype(str).str.lower().isin(["true", "1"])
    n_list = len(res)

    # TRAINING-SPACE BOUND, read off the training set rather than chosen. The
    # model saw molecules from 27 to about 1700 Da; a 3,354 Da polycyclic
    # peptide (nisin) or a 14 kDa enzyme (lysozyme) is not a molecule this
    # fingerprint representation describes at all, and the applicability band
    # alone understates that - it reports "outside domain" for a near-neighbour
    # miss and for an object of the wrong kind in the same words.
    max_train_mw = float(pd.read_csv(config.DESCRIPTORS_CSV).MolWt.max())

    w("=" * 74)
    w("EU FOOD ADDITIVE SCREEN - ACCOUNTING")
    w("=" * 74)
    w("additives in the Union list (Annex II Part B)   : %d" % n_list)
    w("  resolved to a validated PubChem structure     : %d" % int(res.validated.sum()))
    w("  (training-set molecular-weight maximum, used as a scope bound "
      "below: %.1f Da)" % max_train_mw)
    w("  not resolved                                  : %d" % int((~res.validated).sum()))
    for reason, n in Counter(res.loc[~res.validated, "reason"].fillna("")).most_common():
        w("      %-54s %d" % (str(reason)[:54], n))

    cand = res[res.validated].copy()
    pred = MitoToxPredictor()
    compiled = compile_alerts()

    rows, dropped = [], []
    for _, r in cand.iterrows():
        # SCOPE GATES, before anything is scored. A resolved structure is not
        # the same thing as a scoreable one: a representative CID for xanthan
        # gum is not xanthan gum, and a carbonate is not an organic compound.
        # Both gates come from src/mitotox/additives.py, the same module the
        # audit uses, so the two cannot disagree.
        # Always recompute from the name. The resolution table carries a
        # substance_class column, but it was written when the resolver ran:
        # reading it back would let a stale label outvote the current
        # classifier, which is precisely the drift this pipeline exists to
        # prevent. The classifier is the single source of truth.
        cls = classify_substance(r["name"])
        if cls != SINGLE_SUBSTANCE:
            dropped.append((r.e_number, r["name"],
                            "variable composition: polymer, mixture, extract or "
                            "class name, so no single structure represents it"))
            continue
        std = standardize(r.smiles)
        if not std["ok"]:
            dropped.append((r.e_number, r["name"], "structure refused: " + std["reason"]))
            continue
        if not std["parent_has_carbon"]:
            dropped.append((r.e_number, r["name"], "inorganic parent (no carbon)"))
            continue
        parent = Chem.MolFromSmiles(std["parent_smiles"])
        if is_inorganic_carbon(parent):
            dropped.append((r.e_number, r["name"],
                            "inorganic carbon species (matched %s)"
                            % inorganic_carbon_reason(parent)))
            continue
        parent_mw = Descriptors.MolWt(parent)
        if parent_mw > max_train_mw:
            dropped.append((r.e_number, r["name"],
                            "above the training-set molecular-weight maximum "
                            "(%.1f Da vs %.1f Da)" % (parent_mw, max_train_mw)))
            continue
        p = pred.predict_one(r.smiles, r["name"])
        if not p["scored"]:
            dropped.append((r.e_number, r["name"], p["warnings"]))
            continue
        mol = Chem.MolFromSmiles(p["standardized_smiles"])
        hits = match_alerts(mol, compiled)
        rows.append({
            "e_number": r.e_number,
            "name": r["name"],
            "functional_group": r.get("functional_group", ""),
            "pubchem_cid": r.pubchem_cid,
            "standardized_smiles": p["standardized_smiles"],
            "inchikey": p["inchikey"],
            "predicted_probability": p["probability"],
            "predicted_class_t0625": p["predicted_class"],
            "predicted_class_t0390": int(p["probability"] >= 0.390),
            "applicability": p["applicability"],
            "nn_tanimoto_to_training": p["nn_tanimoto"],
            "in_tox21_training_set": p["in_training_set"],
            "tox21_experimental_label": p["training_set_label"],
            "structural_alerts": "|".join(k for k, v in hits.items() if v),
            "n_structural_alerts": sum(hits.values()),
            "warnings": p["warnings"],
        })

    scr = pd.DataFrame(rows).sort_values("predicted_probability", ascending=False)
    w("")
    w("  excluded before scoring                       : %d" % len(dropped))

    def bucket(why):
        s = str(why)
        if s.startswith("variable composition"):
            return "variable composition (no single structure)"
        if s.startswith("inorganic carbon species"):
            return "inorganic carbon species (carbonate, CO2, cyanide)"
        if s.startswith("inorganic parent"):
            return "inorganic parent (carbon-free)"
        if s.startswith("above the training-set"):
            return "larger than anything in the training set"
        if s.startswith("structure refused"):
            return "structure refused at standardisation"
        return "not scored by the predictor"

    for reason, n in Counter(bucket(w_) for _, _, w_ in dropped).most_common():
        w("      %-52s %d" % (reason, n))
    w("")
    w("SCREENED                                        : %d" % len(scr))
    w("  reconciles: %d resolved - %d refused = %d" %
      (int(cand.shape[0]), len(dropped), len(scr)))
    assert len(scr) + len(dropped) == len(cand), "accounting does not reconcile"

    w("")
    w("OF THE %d SCREENED COMPOUNDS" % len(scr))
    w("  already in the Tox21 training set             : %d  (experimental label "
      "known; not a prediction)" % int(scr.in_tox21_training_set.sum()))
    novel = scr[~scr.in_tox21_training_set]
    w("  genuinely out-of-sample                       : %d" % len(novel))
    w("")
    w("  applicability domain:")
    for band, n in scr.applicability.value_counts().items():
        w("      %-44s %d" % (band, n))
    w("")
    w("  flagged at the primary threshold (p >= 0.625) : %d of %d (%.1f%%)"
      % (int(scr.predicted_class_t0625.sum()), len(scr),
         100.0 * scr.predicted_class_t0625.mean()))
    w("  flagged at the screening threshold (p >= 0.390): %d of %d (%.1f%%)"
      % (int(scr.predicted_class_t0390.sum()), len(scr),
         100.0 * scr.predicted_class_t0390.mean()))
    w("")
    w("  of the out-of-sample compounds only:")
    w("      flagged at 0.625                          : %d of %d"
      % (int(novel.predicted_class_t0625.sum()), len(novel)))
    w("      flagged at 0.390                          : %d of %d"
      % (int(novel.predicted_class_t0390.sum()), len(novel)))

    w("")
    w("TOP 25 BY PREDICTED PROBABILITY")
    w("  %-9s %-34s %7s %-15s %6s %s"
      % ("E number", "name", "prob", "applicability", "NN", "in training"))
    w("  " + "-" * 92)
    for _, r in scr.head(25).iterrows():
        w("  %-9s %-34s %7.3f %-15s %6.2f %s"
          % (r.e_number, str(r["name"])[:34], r.predicted_probability,
             r.applicability, r.nn_tanimoto_to_training,
             "YES" if r.in_tox21_training_set else ""))

    w("")
    w("NOTE. These are PREDICTED mitochondrial-toxicity risk scores: a")
    w("computational prioritisation for further investigation. They are not")
    w("experimental evidence, and they support no regulatory or safety claim")
    w("about any compound. Rows marked 'in training' are fitted values, not")
    w("predictions. Scores for compounds outside the applicability domain are")
    w("weakly informative at best (measured test PR-AUC 0.59 in that band).")

    scr.to_csv(OUT_CSV, index=False)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
        "savefig.dpi": 200, "savefig.bbox": "tight"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 5.4),
                                 gridspec_kw={"width_ratios": [1, 1.5],
                                              "wspace": 0.46})

    a1.hist(scr.predicted_probability, bins=np.linspace(0, 1, 26), color=BLUE, zorder=3)
    a1.axvline(0.625, color=ORANGE, lw=1.6)
    a1.axvline(0.390, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    a1.text(0.645, a1.get_ylim()[1] * 0.90, "0.625\nprimary", color=ORANGE,
            fontsize=8, ha="left", va="top")
    a1.text(0.375, a1.get_ylim()[1] * 0.90, "0.390", color=MUTED, fontsize=8,
            ha="right", va="top")
    a1.set_xlabel("predicted probability of MMP disruption")
    a1.set_ylabel("EU food additives")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.grid(True, axis="y")
    a1.set_axisbelow(True)
    a1.text(0, 1.045, "Score distribution", transform=a1.transAxes, color=INK,
            fontsize=10.5, fontweight="600", va="bottom")

    top = novel.head(18).iloc[::-1]
    cols = [ORANGE if p >= 0.625 else MUTED if p < 0.390 else BLUE
            for p in top.predicted_probability]
    b = a2.barh(np.arange(len(top)), top.predicted_probability.values, 0.6,
                color=cols, zorder=3)
    for r_, v, ad in zip(b, top.predicted_probability.values, top.applicability):
        # A probability of 0.998 printed as "1.00" reads as certainty, which
        # is not what the model said. Keep a third decimal where rounding
        # would reach 1.00 or 0.00.
        txt = "%.3f" % v if (round(v, 2) >= 1.0 or round(v, 2) <= 0.0) else "%.2f" % v
        a2.text(v + 0.012, r_.get_y() + r_.get_height() / 2,
                "%s%s" % (txt, "" if ad == "within domain" else "  (%s)" % ad),
                va="center", color=INK, fontsize=8)
    a2.set_yticks(np.arange(len(top)))
    def _short(s, w=34):
        s = str(s)
        return s if len(s) <= w else s[:w - 1].rstrip() + "\u2026"
    a2.set_yticklabels(["%s  %s" % (e, _short(n))
                        for e, n in zip(top.e_number, top["name"])],
                       color=INK2, fontsize=8.5)
    a2.set_xlim(0, 1.28)
    a2.set_xlabel("predicted probability")
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    a2.grid(True, axis="x")
    a2.set_axisbelow(True)
    a2.text(0, 1.09, "Highest-ranked additives NOT in the Tox21 training set",
            transform=a2.transAxes, color=INK, fontsize=10.5, fontweight="600",
            va="bottom")
    a2.text(0, 1.035, "predicted risk, for prioritisation only - not evidence of "
            "toxicity", transform=a2.transAxes, color=INK2, fontsize=8.5, va="bottom")
    fig.savefig(FIG)
    plt.close(fig)

    print("")
    print("written: %s" % OUT_CSV)
    print("written: %s" % REPORT)
    print("written: %s" % FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
