#!/usr/bin/env python3
"""
Generate the repository figures from saved results. Computes no new model.

Palette and mark rules follow one validated categorical set (blue/orange/aqua),
a recessive grid, thin marks with gaps, and direct value labels on every bar
(the aqua slot sits below 3:1 on the light surface, so labels are required, not
optional). Single y-axis everywhere; no dual-axis charts.

Usage:  py -3 scripts/make_figures.py
Output: results/figures/*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

P, T, F, R = "data/processed", "results/tables", "results/figures", "results"
os.makedirs(F, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 2.0, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def title(ax, main, sub=None):
    """Title and optional subtitle placed as explicit text above the axes.
    set_title's pad and a separate text both measure from the axes top, so
    mixing them collides; this places both on one known scale."""
    ax.text(0, 1.145 if sub else 1.045, main, transform=ax.transAxes,
            color=INK, fontsize=10.5, fontweight="600", va="bottom")
    if sub:
        ax.text(0, 1.045, sub, transform=ax.transAxes, color=INK2,
                fontsize=8.5, va="bottom")


# ---------------------------------------------------------------- load
train = pd.read_csv(f"{P}/tox21_mmp_training.csv")
splits = pd.read_csv(f"{P}/splits.csv")
cv = pd.read_csv(f"{T}/cv_results_ranked.csv")
sel = json.load(open(f"{R}/model_selection.json"))
fin = json.load(open(f"{R}/final_test_results.json"))
testp = pd.read_csv(f"{T}/test_predictions.csv")
prox = pd.read_csv(f"{T}/test_performance_by_proximity.csv")
shap = pd.read_csv(f"{T}/shap_global_importance.csv")
alerts = pd.read_csv(f"{T}/structural_alert_enrichment_dev.csv")
oof = np.load(f"{P}/oof_predictions_dev.npz", allow_pickle=True)

# ============================================== 1. dataset composition
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.25],
                                          "wspace": 0.42})
sub = splits.groupby(["subset", "label"]).size().unstack(fill_value=0)
sub = sub.loc[["development", "test"]]
xs = np.arange(2)
w = 0.34
for i, (lab, col, nm) in enumerate([(0, BLUE, "Inactive (0)"), (1, ORANGE, "Active (1)")]):
    v = sub[lab].values
    b = a1.bar(xs + (i - 0.5) * (w + 0.03), v, w, color=col, label=nm, zorder=3)
    for r, val in zip(b, v):
        a1.text(r.get_x() + r.get_width() / 2, val + 70, f"{val:,}", ha="center",
                color=INK, fontsize=8.5)
a1.set_xticks(xs)
a1.set_xticklabels([f"Development\nn = {sub.loc['development'].sum():,}",
                    f"Frozen test\nn = {sub.loc['test'].sum():,}"], color=INK2)
a1.set_ylabel("compounds (parent structures)")
a1.set_ylim(0, 4600)
a1.legend(loc="upper right")
tidy(a1)
_pa = sub[1].sum() / sub.values.sum()
_pd_ = sub.loc["development", 1] / sub.loc["development"].sum()
_pt = sub.loc["test", 1] / sub.loc["test"].sum()
title(a1, "Final curated dataset",
      f"{_pa:.2%} active overall; {_pd_:.2%} development, {_pt:.2%} test")

# Read from the excluded table rather than hard-coding. These five counts were
# literals here, which meant the figure would keep showing the old numbers if
# the dataset were ever rebuilt - a silent disagreement between a figure and
# the data it claims to describe.
_LABELS = {"inconclusive_only": "Inconclusive only",
           "inorganic_parent": "Inorganic parent",
           "conflict_cid_level": "CID-level conflict",
           "conflict_parent_level": "Parent-level conflict",
           "parent_unparseable": "Unparseable parent"}
_excl = pd.read_csv(f"{P}/tox21_mmp_excluded.csv")
_counts = _excl.exclusion_reason.value_counts()
exc = [(_LABELS.get(k, k), int(v)) for k, v in _counts.items()]
n_excluded = int(len(_excl))
n_parents = n_excluded + int(len(splits))
ys = np.arange(len(exc))[::-1]
b = a2.barh(ys, [e[1] for e in exc], 0.55, color=BLUE, zorder=3)
for r, (_, v) in zip(b, exc):
    a2.text(v + 25, r.get_y() + r.get_height() / 2, f"{v:,}", va="center",
            color=INK, fontsize=8.5)
a2.set_yticks(ys)
a2.set_yticklabels([e[0] for e in exc], color=INK2)
a2.set_xlabel("parent structures excluded")
a2.set_xlim(0, max(v for _, v in exc) * 1.16)
tidy(a2, "x")
title(a2, "Excluded and retained, never discarded",
      f"{n_excluded:,} of {n_parents:,} parents excluded; all kept in a flagged table")
fig.savefig(f"{F}/dataset_composition.png")
plt.close(fig)

# ============================================== 2. model comparison
best = (cv.sort_values("pr_auc_mean", ascending=False)
          .groupby(["representation", "model"]).head(1))
reps = ["A_morgan", "B_morgan_desc", "C_maccs_desc"]
replab = ["A: Morgan", "B: Morgan + descriptors", "C: MACCS + descriptors"]
models = ["logistic_regression", "random_forest", "xgboost"]
modlab = ["Logistic regression", "Random forest", "XGBoost"]
cols = [BLUE, ORANGE, AQUA]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
xs = np.arange(len(reps))
w = 0.24
for i, (m, nm, c) in enumerate(zip(models, modlab, cols)):
    vals, errs = [], []
    for r in reps:
        row = best[(best.representation == r) & (best.model == m)]
        vals.append(float(row.pr_auc_mean.iloc[0]))
        errs.append(float(row.pr_auc_sd.iloc[0]))
    b = ax.bar(xs + (i - 1) * (w + 0.025), vals, w, color=c, label=nm, zorder=3)
    ax.errorbar(xs + (i - 1) * (w + 0.025), vals, yerr=errs, fmt="none",
                ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
    for r_, v, e in zip(b, vals, errs):
        ax.text(r_.get_x() + r_.get_width() / 2, v + e + 0.015, f"{v:.3f}",
                ha="center", color=INK, fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(replab, color=INK2)
ax.set_ylabel("cross-validated PR-AUC (mean of 5 folds)")
ax.set_ylim(0, 0.92)
ax.axhline(train.label.mean(), color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
# The baseline is named in the subtitle rather than annotated in the plot.
# The bars run to y=0, so there is no horizontal band inside the axes where
# a label that long does not collide with one of them.
ax.legend(loc="upper left", ncol=3)
tidy(ax)
title(ax, "Best configuration per representation and model family",
      "error bars: standard deviation across the 5 development CV folds; "
      "108 configurations, 540 fits; dashed line is the random baseline at "
      f"prevalence {train.label.mean():.3f}")
fig.savefig(f"{F}/model_comparison.png")
plt.close(fig)

# ============================================== 3. ROC and PR
wcfg = sel["winner"]["config"]
y_dev, p_dev = oof["y"], oof[wcfg]
y_te, p_te = testp.label.values, testp.predicted_probability.values
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.3))
for y_, p_, c, nm in [(y_dev, p_dev, BLUE, "Development (out-of-fold)"),
                      (y_te, p_te, ORANGE, "Frozen test")]:
    fpr, tpr, _ = roc_curve(y_, p_)
    a1.plot(fpr, tpr, color=c, label=f"{nm}  AUC {roc_auc_score(y_, p_):.3f}")
    pr, rc, _ = precision_recall_curve(y_, p_)
    a2.plot(rc, pr, color=c,
            label=f"{nm}  AP {average_precision_score(y_, p_):.3f}")
a1.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=(0, (4, 3)))
a1.set_xlabel("false positive rate")
a1.set_ylabel("true positive rate")
a1.legend(loc="lower right")
tidy(a1, "both")
title(a1, "ROC", "frozen XGBoost on Morgan + descriptors")
a2.axhline(y_te.mean(), color=MUTED, lw=1.0, ls=(0, (4, 3)))
a2.text(0.98, y_te.mean() + 0.02, "prevalence 0.148", color=MUTED, fontsize=7.8,
        ha="right")
a2.set_xlabel("recall")
a2.set_ylabel("precision")
a2.set_ylim(0, 1.02)
a2.legend(loc="upper right")
tidy(a2, "both")
title(a2, "Precision-recall", "PR-AUC is the primary metric at 15% prevalence")
fig.savefig(f"{F}/roc_pr_curves.png")
plt.close(fig)

# ============================================== 4. confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
order = [("primary_mcc_max", "Primary: MCC-max, t = 0.625"),
         ("sensitivity_f1_optimal", "Sensitivity: F1-optimal, t = 0.390"),
         ("sensitivity_youden_j", "Sensitivity: Youden-J, t = 0.163")]
for ax, (k, lab) in zip(axes, order):
    d = fin["test_results"]["thresholds"][k]
    cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n{frac:.1%}", ha="center", va="center",
                    color=("#fcfcfb" if frac > 0.55 else INK), fontsize=9.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], color=INK2)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0, 1.06, lab, transform=ax.transAxes, color=INK, fontsize=9,
            fontweight="600", va="bottom")
    ax.text(0, -0.28, f"MCC {d['mcc']:.3f}   sens {d['sensitivity_recall']:.3f}"
                      f"   spec {d['specificity']:.3f}",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
fig.subplots_adjust(top=0.74, bottom=0.18)
fig.text(0.075, 1.015, "Frozen test confusion matrices, 1,205 compounds",
         ha="left", color=INK, fontsize=11, fontweight="600")
fig.text(0.075, 0.955, "thresholds were fixed on development out-of-fold "
                       "predictions before the test set was scored",
         ha="left", color=INK2, fontsize=8.5)
fig.savefig(f"{F}/confusion_matrices.png")
plt.close(fig)

# ============================================== 5. SHAP importance
top = shap.head(16).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.4, 5.2))
cols_ = [ORANGE if r > 0.1 else BLUE if r < -0.1 else MUTED
         for r in top.value_vs_contribution_spearman]
b = ax.barh(np.arange(len(top)), top.mean_abs_shap.values, 0.6, color=cols_, zorder=3)
for r_, v in zip(b, top.mean_abs_shap.values):
    ax.text(v + 0.012, r_.get_y() + r_.get_height() / 2, f"{v:.3f}", va="center",
            color=INK, fontsize=8)
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels([f.replace("morgan_bit_", "Morgan bit ") for f in top.feature],
                   color=INK2)
ax.set_xlabel("mean |SHAP| contribution (development set)")
ax.set_xlim(0, 1.60)
tidy(ax, "x")
h = [plt.Line2D([0], [0], color=ORANGE, lw=6),
     plt.Line2D([0], [0], color=BLUE, lw=6)]
ax.legend(h, ["higher value -> higher predicted risk",
              "higher value -> lower predicted risk"], loc="lower right")
title(ax, "What the model uses", "TreeSHAP on the development set; direction is "
      "Spearman rho between a feature's value and its own contribution")
fig.savefig(f"{F}/shap_importance.png")
plt.close(fig)

# ============================================== 6. alert enrichment
al = alerts.dropna(subset=["odds_ratio_dev"])
al = al[al.n_matches_dev > 0].sort_values("odds_ratio_dev")
fig, ax = plt.subplots(figsize=(8.6, 4.6))
cols_ = [ORANGE if p < 0.05 else MUTED for p in al.fisher_p_dev]
b = ax.barh(np.arange(len(al)), al.odds_ratio_dev.values, 0.58, color=cols_, zorder=3)
for r_, v, n, p in zip(b, al.odds_ratio_dev.values, al.n_matches_dev.values,
                       al.fisher_p_dev.values):
    ax.text(max(v, 0.012) * 1.18, r_.get_y() + r_.get_height() / 2,
            f"{v:.1f}   n={n}" + ("" if p < 0.05 else "   n.s."), va="center",
            color=INK, fontsize=8)
ax.axvline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.set_xscale("symlog", linthresh=0.3)
ax.set_xlim(0, 300)
ax.set_yticks(np.arange(len(al)))
ax.set_yticklabels([a.replace("_", " ") for a in al.alert], color=INK2)
ax.set_xlabel("odds ratio (log scale), development set only")
tidy(ax, "x")
h = [plt.Line2D([0], [0], color=ORANGE, lw=6), plt.Line2D([0], [0], color=MUTED, lw=6)]
ax.legend(h, ["Fisher p < 0.05", "not significant"], loc="lower right")
title(ax, "Structural alerts: enrichment among actives",
      "interpretation only - these are never model features")
fig.savefig(f"{F}/alert_enrichment.png")
plt.close(fig)

# ============================================== 7. performance by proximity
pr = prox.dropna(subset=["pr_auc"]).copy()
fig, ax = plt.subplots(figsize=(8.4, 4.2))
xs = np.arange(len(pr))
b = ax.bar(xs, pr.pr_auc.values, 0.5, color=BLUE, zorder=3)
for r_, v, n, npos in zip(b, pr.pr_auc.values, pr.n.values, pr.n_pos.values):
    ax.text(r_.get_x() + r_.get_width() / 2, v + 0.02,
            f"{v:.2f}\nn={n}, {npos} pos", ha="center", color=INK, fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(pr.nn_band, color=INK2)
ax.set_xlabel("nearest development neighbour, Tanimoto similarity")
ax.set_ylabel("PR-AUC on that band of the test set")
ax.set_ylim(0, 1.05)
tidy(ax)
title(ax, "Performance depends on chemical proximity to the training data",
      "the two most distant bands hold 480 of 1,205 test compounds; "
      "the top band has only 2 positives and is noise")
fig.savefig(f"{F}/performance_by_proximity.png")
plt.close(fig)

print("figures written:")
for f in sorted(os.listdir(F)):
    print("  results/figures/%-32s %7d bytes" % (f, os.path.getsize(f"{F}/{f}")))
