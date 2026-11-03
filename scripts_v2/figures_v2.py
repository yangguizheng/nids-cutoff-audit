#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
LAB = {"lstm": "LSTM", "tcn": "TCN", "transformer": "Transformer", "rf": "RF", "hgb": "HGB"}
COL = {"lstm": "#4C72B0", "tcn": "#DD8452", "transformer": "#55A868",
       "rf": "#C44E52", "hgb": "#8172B3"}
MET = [("excess", "excess"), ("roc_auc", "ROC-AUC"), ("aplift", "AP-lift")]
ARMS = [("CIC17", "-", "CIC-IDS2017"), ("CIC18", "-", "CSE-CIC-IDS2018"),
        ("UNSW", "f0", "NF-UNSW-NB15 f0"), ("UNSW", "f1", "NF-UNSW-NB15 f1")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.linewidth": 0.4,
    "grid.color": "0.85", "grid.linestyle": "-",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def fig1(wr: pd.DataFrame, out: str) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey=True)
    for ax, (ds, fold, title) in zip(axes.ravel(), ARMS):
        sub = wr[(wr.dataset == ds) & (wr.fold == fold) & (wr.route_subset == "all")]
        x = np.arange(len(MODELS))
        w = 0.26
        for k, (met, mlab) in enumerate(MET):
            s = sub[(sub.metric == met) & (sub.eps == 0.0)].set_index("model")
            if s.empty:
                continue
            v = np.array([s.loc[m, "winrate"] * 100 for m in MODELS])
            lo = np.array([s.loc[m, "wilson_lo"] * 100 for m in MODELS])
            hi = np.array([s.loc[m, "wilson_hi"] * 100 for m in MODELS])
            ax.bar(x + (k - 1) * w, v, w, label=mlab,
                   color=["#6F8FBF", "#C9A227", "#7FA87F"][k],
                   edgecolor="black", linewidth=0.4,
                   yerr=[v - lo, hi - v], capsize=1.6,
                   error_kw=dict(lw=0.7, capthick=0.7))
        ax.axhline(20, color="black", lw=0.8, ls="--")
        ax.text(0.985, 0.045, "dashed line: no-difference (20%)", ha="right", va="bottom",
                fontsize=5.8, color="0.25", transform=ax.transAxes,
                bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.9))
        ax.set_xticks(x)
        ax.set_xticklabels([LAB[m] for m in MODELS], fontsize=7)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.set_ylim(0, 100)
        ax.set_ylabel("paired win rate (%)" if ax in axes[:, 0] else "")
    axes[0, 0].legend(frameon=False, fontsize=6.5, ncol=3, loc="upper left",
                      bbox_to_anchor=(0.0, 1.02), handlelength=1.2)
    fig.suptitle("Pairwise win rates under two scoring functions "
                 "(error bars: Wilson 95% CI; n = 180 groups per cell)",
                 fontsize=9, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig4(sm: pd.DataFrame, out: str) -> None:

    rules = sm.rule.unique()
    piv = {}
    for r in rules:
        row = {}
        for setting in ["cic18", "unsw_pair", "cic17"]:
            g = sm[(sm.rule == r) & (sm.setting == setting)]
            if not g.empty and np.isfinite(g.collapse.iloc[0]):
                row[setting] = (g.collapse.iloc[0] * 100,
                                g.ci_lo.iloc[0] * 100, g.ci_hi.iloc[0] * 100)
        piv[r] = row

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, ysetting, ylab in [(axes[0], "unsw_pair", "UNSW collapse rate (%)"),
                               (axes[1], "cic17", "CIC-IDS2017 collapse rate (%)")]:
        for r, row in piv.items():
            if "cic18" not in row or ysetting not in row:
                continue
            x, xl, xh = row["cic18"]
            y, yl, yh = row[ysetting]
            if not all(np.isfinite([x, xl, xh, y, yl, yh])):
                continue
            base = r in ("fixed0.5", "qtrain", "otsu", "bbse", "saerens")
            xerr = [[max(x - xl, 0.0)], [max(xh - x, 0.0)]]
            yerr = [[max(y - yl, 0.0)], [max(yh - y, 0.0)]]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                        fmt="o", ms=3.2 if base else 2.2,
                        color="black" if base else "0.55",
                        mfc="white" if base else "0.75",
                        ecolor="0.55", elinewidth=0.6, capsize=1.2, lw=0)
            if base:
                off = {"val_rank": (-34, 4), "qtrain": (5, -8), "bbse": (-30, -9),
                       "saerens": (5, 3)}.get(r, (4, 3))
                ax.annotate(r, (x, y), textcoords="offset points", xytext=off,
                            fontsize=6.2)
        ax.set_xlabel("CSE-CIC-IDS2018 collapse rate (%)")
        ax.set_ylabel(ylab)
        ax.set_xlim(-4, 100)
        ax.set_ylim(-4, 100)
        ax.plot([0, 100], [0, 100], color="0.7", lw=0.7, ls=":")
    axes[0].set_title("(a) vs. UNSW (450 runs)", fontsize=8)
    axes[1].set_title("(b) vs. CIC-IDS2017 (900 runs)", fontsize=8)
    fig.suptitle("Fifteen cut rules on two axes (error bars: Wilson 95% CI)",
                 fontsize=9, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig3(out: str) -> None:

    dbs = [("CIC17", "results_v2/det_v2_cic17.db"),
           ("CIC18", "results_v2/det_v2_cic18.db"),
           ("UNSW", "results_v2/det_unsw_s20.db")]
    frames = []
    for name, p in dbs:
        if not os.path.exists(p):
            continue
        con = sqlite3.connect(p)
        d = pd.read_sql_query("SELECT route, cell, model, seed, macro_f1, "
                              "test_pos_rate, roc_auc FROM runs", con)
        con.close()
        d["dataset"] = name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)

    def floor_of(p):
        return np.maximum((1 - p) / (2 - p), p / (1 + p))

    d["excess"] = d["macro_f1"] - floor_of(d["test_pos_rate"])
    d["route3"] = np.where(d["route"] == "time_window", "tw30s",
                           np.where(d["route"].isin(["host", "service"]),
                                    d["route"], "other"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2),
                             gridspec_kw=dict(width_ratios=[1.55, 1]))
    ax = axes[0]
    groups, labels, positions = [], [], []
    pos = 0
    for ds in ["CIC17", "CIC18", "UNSW"]:
        for rt in ["tw30s", "host", "service"]:
            v = d[(d.dataset == ds) & (d.route3 == rt)]["excess"].dropna().values
            if len(v) == 0:
                continue
            groups.append(v)
            labels.append(rt)
            positions.append(pos)
            pos += 1
        pos += 0.6
    bp = ax.boxplot(groups, positions=positions, widths=0.62, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=0.9),
                    boxprops=dict(lw=0.5, edgecolor="0.35"),
                    whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    palette = {"tw30s": "#C44E52", "host": "#4C72B0", "service": "#55A868"}
    for patch, lb in zip(bp["boxes"], labels):
        patch.set_facecolor(palette[lb])
        patch.set_alpha(0.75)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.text(0.015, 0.035, "dashed: floor (excess = 0)", ha="left", va="bottom",
            fontsize=5.8, color="0.25", transform=ax.transAxes,
            bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel(r"excess = macro-F1@0.5 $-$ floor($p$)")
    ax.set_title("(a) per-run distribution by segment route", fontsize=8)

    for xsep in [2.31, 5.22]:
        ax.axvline(xsep, color="0.6", lw=0.6, ls=":")
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.14)
    for xtxt, ds in [(1, "CIC-IDS2017"), (4, "CSE-CIC-IDS2018"), (7, "NF-UNSW-NB15")]:
        ax.text(xtxt, ax.get_ylim()[1], ds, ha="center", va="top", fontsize=6.8)

    ax2 = axes[1]
    sw = []
    for ds in ["CIC17", "CIC18", "UNSW"]:
        sub = d[d.dataset == ds]
        m = sub.groupby("route3")["excess"].mean()
        sw.append((ds, m.get("tw30s", np.nan), m.get("host", np.nan),
                   m.get("service", np.nan)))
    x = np.arange(3)
    w = 0.26
    for k, (rt, lab) in enumerate([("tw30s", "tw30s"), ("host", "host"),
                                   ("service", "service")]):
        vals = [s[k + 1] for s in sw]
        ax2.bar(x + (k - 1) * w, vals, w, label=lab, color=palette[rt],
                edgecolor="black", linewidth=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[0] for s in sw], fontsize=7)
    ax2.set_ylabel("mean excess")
    ax2.set_title("(b) marginal means", fontsize=8)
    ax2.legend(frameon=False, fontsize=6.2, ncol=3, loc="upper left",
               bbox_to_anchor=(0.0, 1.0), handlelength=1.0)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.22)
    ax2.axhline(0, color="black", lw=0.7)
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig2(out: str) -> None:

    con = sqlite3.connect("results_v2/det_v2_cic18.db")
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    floor = np.maximum((1 - d.test_pos_rate) / (2 - d.test_pos_rate),
                       d.test_pos_rate / (1 + d.test_pos_rate))
    d["excess"] = d.macro_f1 - floor
    col = "thr_oracle" if "thr_oracle" in d.columns else None
    if col is None:
        print("  [fig2] thr_oracle column absent from DB; skipping")
        return
    d[col] = pd.to_numeric(d[col], errors="coerce")
    ok = d[np.isfinite(d[col])]
    coll = ok[(ok.excess < 0.02) & (ok[col] < 1)]
    norm = ok[ok.excess >= 0.02]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    bins = np.logspace(-6, 0, 40)
    ax.hist(coll[col].clip(1e-6, 1), bins=bins, color="#C44E52", alpha=0.75,
            label="collapsed (n=%d)" % len(coll), edgecolor="black", linewidth=0.3)
    ax.hist(norm[col].clip(1e-6, 1), bins=bins, color="#4C72B0", alpha=0.6,
            label="healthy (n=%d)" % len(norm), edgecolor="black", linewidth=0.3)
    ax.axvline(0.5, color="black", lw=1.0, ls="--")
    ax.text(0.13, 0.93, "fixed 0.5", fontsize=6.3, color="0.15",
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    ax.set_xscale("log")
    ax.set_xlabel("oracle threshold (optimal on test labels)")
    ax.set_ylabel("runs")
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("(a) where the optimal cut actually sits", fontsize=8)

    ax2 = axes[1]
    data = [coll[col].clip(1e-6, 1).values, norm[col].clip(1e-6, 1).values]
    bp = ax2.boxplot(data, labels=["collapsed", "healthy"], widths=0.5,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", lw=0.9))
    for patch, c in zip(bp["boxes"], ["#C44E52", "#4C72B0"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax2.axhline(0.5, color="black", lw=1.0, ls="--")
    ax2.set_yscale("log")
    ax2.set_ylabel("oracle threshold")
    ax2.set_title("(b) same, as boxes", fontsize=8)
    fig.suptitle("CSE-CIC-IDS2018, 900 runs: a fixed 0.5 cut is too high for collapsed runs"
                 " and too low for healthy ones", fontsize=8.3, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, format="pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_v2/figures_v2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    wr = pd.read_csv("results_v2/wp1_winrate_s20.csv")
    fig1(wr, os.path.join(args.outdir, "fig1_winrate.pdf"))
    print("fig1 ok")

    sm = pd.read_csv("results_v2/wp4d_summary.csv")
    fig4(sm, os.path.join(args.outdir, "fig4_rules.pdf"))
    print("fig4 ok")

    fig3(os.path.join(args.outdir, "fig3_segment.pdf"))
    print("fig3 ok")

    fig2(os.path.join(args.outdir, "fig2_threshold.pdf"))
    print("fig2 ok")


if __name__ == "__main__":
    main()
