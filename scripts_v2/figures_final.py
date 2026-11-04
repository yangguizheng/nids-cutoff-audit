#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import io
import sys
import os
import sqlite3

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter, Transformation

MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
LAB = {"lstm": "LSTM", "tcn": "TCN", "transformer": "Transformer", "rf": "RF", "hgb": "HGB"}
HATCH = {"lstm": "", "tcn": "///", "transformer": "...", "rf": "\\\\\\", "hgb": "xx"}
ARMS = [("CIC17", "-", "CIC-IDS2017"), ("CIC18", "-", "CSE-CIC-IDS2018"),
        ("UNSW", "f0", "NF-UNSW-NB15 fold 0"), ("UNSW", "f1", "NF-UNSW-NB15 fold 1")]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DS_COL = {"CIC17": BLUE, "CIC18": ORANGE, "UNSW": AQUA, "UNSW f0": AQUA, "UNSW f1": AQUA}
GREY = {"excess": BLUE, "roc_auc": ORANGE}
SIZE = dict(base=8.0, tick=8.0, lab=8.5, panel=9.5, note=8.0)


def style(font: str) -> None:
    plt.rcParams.update({
        "font.family": font, "font.size": SIZE["base"],
        "axes.labelsize": SIZE["lab"], "xtick.labelsize": SIZE["tick"],
        "ytick.labelsize": SIZE["tick"], "legend.fontsize": SIZE["base"],
        "axes.linewidth": 0.8, "axes.grid": True, "grid.linewidth": 0.4,
        "grid.color": "0.88", "grid.linestyle": "-",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def _ticks_in_view(axis, lim):

    lo, hi = lim
    out = []
    for tk in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
        loc = tk.get_loc()
        if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9) or not tk.get_visible():
            continue
        for lb in (tk.label1, tk.label2):
            if lb is not None and lb.get_visible() and (lb.get_text() or "").strip():
                out.append(lb)
    return out


def check(fig, name, report):

    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    texts, seen = [], set()

    def add(objs):
        for t in objs:
            if id(t) in seen:
                continue
            seen.add(id(t))
            texts.append(t)

    for ax in fig.axes:
        add(ax.texts)
        add(_ticks_in_view(ax.xaxis, ax.get_xlim()))
        add(_ticks_in_view(ax.yaxis, ax.get_ylim()))
        add([ax.xaxis.label, ax.yaxis.label])
        lg = ax.get_legend()
        if lg is not None:
            add(lg.get_texts())
    add(fig.texts)
    for lg in fig.legends:
        add(lg.get_texts())

    items = []
    for t in texts:
        txt = (t.get_text() or "").strip()
        if not txt or not t.get_visible():
            continue
        try:
            bb = t.get_window_extent(renderer=ren)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((txt, bb))


    dpi = fig.dpi
    tb = fig.get_tightbbox(ren)
    bx0, by0, bx1, by1 = tb.x0 * dpi, tb.y0 * dpi, tb.x1 * dpi, tb.y1 * dpi
    start = len(report)
    n_clip = 0
    for i in range(len(items)):
        ti, bi = items[i]
        if bi.x0 < bx0 - 1 or bi.y0 < by0 - 1 or bi.x1 > bx1 + 1 or bi.y1 > by1 + 1:
            report.append("%s CLIPPED %r" % (name, ti[:40]))
            n_clip += 1
        for j in range(i + 1, len(items)):
            tj, bj = items[j]
            if bi.overlaps(bj):
                ix = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                iy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if ix > 1.5 and iy > 1.5:
                    report.append("%s OVERLAP %r <-> %r (%.1fx%.1f px)"
                                  % (name, ti[:36], tj[:36], ix, iy))
    report.append("%s: %d texts, %d clipped, %d overlaps"
                  % (name, len(items), n_clip, len(report) - start - n_clip))
    return report


REPORT = []


PAGE_W, PAGE_H, MARGIN = 8.5, 11.0, 1.0
LETTER = True


def _save(fig, out):
    name = os.path.splitext(os.path.basename(out))[0]
    check(fig, name, REPORT)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    if not LETTER:
        with open(out, "wb") as fh:
            fh.write(buf.getvalue())
        return

    buf.seek(0)
    src = PdfReader(buf).pages[0]
    w = float(src.mediabox.width) / 72.0
    h = float(src.mediabox.height) / 72.0
    s = min(1.0, (PAGE_W - 2 * MARGIN) / w, (PAGE_H - 2 * MARGIN) / h)

    writer = PdfWriter()
    page = writer.add_blank_page(width=PAGE_W * 72, height=PAGE_H * 72)
    tx = (PAGE_W - w * s) / 2.0 * 72
    ty = (PAGE_H - MARGIN - h * s) * 72
    page.merge_transformed_page(src, Transformation().scale(s).translate(tx, ty))
    with open(out, "wb") as fh:
        writer.write(fh)
    REPORT.append("%s: artwork %.2f x %.2f in, placed at scale %.3f (1 pt -> %.3f pt)"
                  % (name, w, h, s, s))


def panel(ax, letter: str) -> None:
    ax.text(-0.14, 1.06, "(%s)" % letter, transform=ax.transAxes,
            fontsize=SIZE["panel"], fontweight="bold", va="bottom", ha="left")


def floor_of(p):
    return np.maximum((1 - p) / (2 - p), p / (1 + p))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (ph, max(0.0, c - h), min(1.0, c + h))


def fig1(wr, tau, out):
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.0), sharey=True)
    met = [("excess", "excess"), ("roc_auc", "ROC-AUC")]
    for ax, (ds, fold, title), letter in zip(axes.ravel(), ARMS, "ABCD"):
        sub = wr[(wr.dataset == ds) & (wr.fold == fold) & (wr.route_subset == "all")]
        x = np.arange(len(MODELS))
        w = 0.34
        for k, (m, mlab) in enumerate(met):
            s = sub[(sub.metric == m) & (sub.eps == 0.0)].set_index("model")
            if s.empty:
                continue
            v = np.array([s.loc[mm, "winrate"] * 100 for mm in MODELS])
            lo = np.array([s.loc[mm, "wilson_lo"] * 100 for mm in MODELS])
            hi = np.array([s.loc[mm, "wilson_hi"] * 100 for mm in MODELS])
            ax.bar(x + (k - 0.5) * w, v, w, label=mlab,
                   color=GREY[m], edgecolor="white", linewidth=0.5,
                   yerr=[np.maximum(v - lo, 0), np.maximum(hi - v, 0)],
                   capsize=1.8, error_kw=dict(lw=0.7, capthick=0.7, ecolor="black"))
        ax.axhline(20, color="black", lw=1.0, ls=(0, (4, 2, 1, 2)))
        ax.set_xticks(x)
        ax.set_xticklabels([LAB[m] for m in MODELS])
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        panel(ax, letter)
        ax.text(-0.045, 1.02, title, transform=ax.transAxes,
                fontsize=SIZE["base"], va="bottom", ha="left")
        t = tau[(tau.dataset == ds) & (tau.fold == fold) & (tau.route_subset == "all")
                & (tau.pair == "excess~roc_auc") & (tau.aggregation == "winrate")]
        note = "dashed: no difference (20%)"
        if not t.empty:
            note += "\nKendall tau = %.2f, n = %d groups" % (t.tau.iloc[0], t.n_groups.iloc[0])
        ax.text(0.975, 0.975, note, transform=ax.transAxes, fontsize=SIZE["note"],
                va="top", ha="right", linespacing=1.35,
                bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    for ax in axes[:, 0]:
        ax.set_ylabel("paired win rate (%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), handlelength=1.6)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    _save(fig, out)


def fig2(out, db="results_v2/det_v2_cic18.db"):
    con = sqlite3.connect(db)
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    d["excess"] = d.macro_f1 - floor_of(d.test_pos_rate)
    d["thr_oracle"] = pd.to_numeric(d.thr_oracle, errors="coerce")
    ok = d[np.isfinite(d.thr_oracle)].copy()
    ok["collapsed"] = ok.excess < 0.02
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1),
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    rng = np.random.default_rng(0)
    jit = rng.normal(0, 0.012, len(ok))
    for x0, row in zip(jit, ok.itertuples()):
        ax.plot([0 + x0, 1 + x0], [row.macro_f1 * 100, row.macro_f1_oracle * 100],
                color=ORANGE if row.collapsed else "#a9c8ef",
                lw=0.45 if row.collapsed else 0.3,
                alpha=0.7 if row.collapsed else 0.45, zorder=3 if row.collapsed else 1)
    ax.scatter(np.zeros(len(ok)) + jit, ok.macro_f1 * 100, s=3.5,
               c=np.where(ok.collapsed, ORANGE, BLUE), edgecolors="none",
               linewidths=0.3, zorder=4)
    ax.scatter(np.ones(len(ok)) + jit, ok.macro_f1_oracle * 100, s=3.5,
               c=np.where(ok.collapsed, ORANGE, BLUE), edgecolors="none",
               linewidths=0.3, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cut at 0.5", "oracle cut"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("macro-F1 (%)")
    n_c = int(ok.collapsed.sum())
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=3.6, mfc=ORANGE, mec=ORANGE,
               label="collapsed (n = %d)" % n_c),
        Line2D([], [], marker="o", ls="", ms=3.6, mfc=BLUE, mec=BLUE,
               label="other runs (n = %d)" % (len(ok) - n_c))],
        frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
        handletextpad=0.3, columnspacing=1.4)
    panel(ax, "A")

    ax2 = axes[1]
    bins = np.logspace(-4, 0, 33)
    a = ok.loc[ok.collapsed, "thr_oracle"].clip(1e-4, 1)
    b = ok.loc[~ok.collapsed, "thr_oracle"].clip(1e-4, 1)
    ax2.hist(a, bins=bins, color=ORANGE, alpha=0.85, edgecolor="white",
             linewidth=0.3, label="collapsed")
    ax2.hist(b, bins=bins, color="white", alpha=0.9, edgecolor=BLUE,
             linewidth=0.5, hatch="///", label="other runs")
    ax2.axvline(0.5, color="black", lw=1.0, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlabel("oracle cut-off")
    ax2.set_ylabel("runs")
    ax2.set_xlim(1e-4, 1.2)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.10)
    ax2.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
               handletextpad=0.4, columnspacing=1.1)
    ax2.text(0.5, 0.97, "fixed 0.5", fontsize=SIZE["note"], ha="right", va="top",
             transform=ax2.get_xaxis_transform(),
             bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.9))
    panel(ax2, "B")
    fig.tight_layout()
    _save(fig, out)


def load_runs():
    out = []
    for name, p in [("CIC17", "results_v2/det_v2_cic17.db"),
                    ("CIC18", "results_v2/det_v2_cic18.db"),
                    ("UNSW", "results_v2/det_unsw_s20.db")]:
        if not os.path.exists(p):
            continue
        con = sqlite3.connect(p)
        d = pd.read_sql_query("SELECT route, cell, model, seed, macro_f1, test_pos_rate,"
                              " roc_auc FROM runs", con)
        con.close()
        d["dataset"] = name
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    d["route3"] = np.where(d.route == "time_window", "tw30s", d.route)
    d["excess"] = d.macro_f1 - floor_of(d.test_pos_rate)
    return d


def fig3(d, bdata, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    routes = ["tw30s", "host", "service"]
    mark = {"tw30s": "o", "host": "s", "service": "^"}
    fill = {"tw30s": "black", "host": "0.55", "service": "white"}
    pos = 0
    ticks, labels = [], []
    for dsname in ["CIC17", "CIC18", "UNSW"]:
        for rt in routes:
            v = d[(d.dataset == dsname) & (d.route3 == rt)]["roc_auc"].dropna().values
            if not len(v):
                continue
            rng = np.random.default_rng(1)
            ax.scatter(pos + rng.normal(0, 0.05, len(v)), v, s=2.5,
                       marker=mark[rt], facecolors=DS_COL[dsname], edgecolors="none",
                       linewidths=0.2, alpha=0.55, zorder=2)
            ax.hlines(np.median(v), pos - 0.28, pos + 0.28, color="black", lw=1.4, zorder=4)
            ticks.append(pos)
            labels.append(rt)
            pos += 1
        pos += 0.7
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, ha="center")
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("per-run ROC-AUC")
    import matplotlib.transforms as mtrans
    tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)

    for x, dsname in [(1.0, "CIC-\nIDS2017"), (4.7, "CSE-CIC-\nIDS2018"),
                      (8.4, "NF-UNSW-\nNB15")]:
        ax.text(x, 1.012, dsname, transform=tr, ha="center", va="bottom",
                fontsize=SIZE["note"], linespacing=0.95)
    for xs in [2.55, 5.45]:
        ax.axvline(xs, color="0.6", lw=0.6, ls=":")
    panel(ax, "A")

    ax2 = axes[1]


    groups = ["CIC17", "CIC18", "UNSW f0", "UNSW f1"]
    mark = {"CIC17": "o", "CIC18": "s", "UNSW f0": "^", "UNSW f1": "D"}
    facs = [("segment", "route"), ("padding", "padding"),
            ("model", "model"), ("seed_sd", "seed SD")]
    for i, (f, lab) in enumerate(facs):
        for k, g in enumerate(groups):
            if f == "seed_sd":
                r = bdata[(bdata.group == g) & (bdata.factor == "seed_sd")]
                if r.empty:
                    continue
                ax2.scatter(i + (k - 1.5) * 0.22, float(r.value.iloc[0]) * 100,
                            s=15, marker=mark[g], facecolors="white",
                            edgecolors=DS_COL[g], linewidths=0.9, zorder=3)
                continue
            for met, fc in [("excess", None), ("roc_auc", "white")]:
                r = bdata[(bdata.group == g) & (bdata.factor == f)
                          & (bdata.metric == met)]
                if r.empty:
                    continue
                dx = (k - 1.5) * 0.22 + (-0.055 if met == "excess" else 0.055)
                ax2.scatter(i + dx, float(r.value.iloc[0]) * 100,
                            s=15, marker=mark[g], facecolors=(DS_COL[g] if fc is None else fc),
                            edgecolors=DS_COL[g], linewidths=0.9, zorder=3)
    ax2.set_yscale("log")
    ax2.set_xticks(np.arange(len(facs)))
    ax2.set_xticklabels([lab for _, lab in facs])
    ax2.set_ylabel("spread (pp)")
    from matplotlib.lines import Line2D
    h_ds = [Line2D([0], [0], marker=mark[g], ls="", color=DS_COL[g], mfc=DS_COL[g],
                   mec=DS_COL[g], ms=3.2, label=g) for g in groups]
    h_met = [Line2D([0], [0], marker="o", ls="", color="black", mfc="black",
                    mec="black", ms=3.2, label="excess"),
             Line2D([0], [0], marker="o", ls="", color="black", mfc="white",
                    mec="black", ms=3.2, label="ROC-AUC")]

    ax2.legend(handles=h_ds + h_met, frameon=False, ncol=3, loc="lower left",
               bbox_to_anchor=(0.0, 1.02), handlelength=1.0,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0,
               labelspacing=0.3)
    panel(ax2, "B")
    fig.tight_layout()
    _save(fig, out)


RULES15 = ["fixed0.5", "val_thr", "val_rank", "qtrain", "qtrain_gt", "otsu", "otsu_logit",
           "hybrid", "hybrid50", "gmm_logit", "gmm_valley", "saerens", "saerens_cal",
           "bbse", "scale_val"]
VAL_RULES_FIG4 = {"val_thr", "val_rank", "scale_val", "bbse", "saerens_cal"}


def fig4(sm, clean, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.5),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    ax = axes[0]
    rules = [r for r in RULES15 if r in set(sm.rule)]
    y = np.arange(len(rules))[::-1]

    cfg = {"cic17": (BLUE, "o", "CIC-IDS2017"),
           "cic18": (ORANGE, "s", "CSE-CIC-IDS2018"),
           "unsw_pair": (AQUA, "^", "NF-UNSW-NB15")}
    for k, setting in enumerate(["cic17", "cic18", "unsw_pair"]):
        vals, los, his = [], [], []
        for r in rules:
            g = sm[(sm.rule == r) & (sm.setting == setting)]
            if g.empty or not np.isfinite(g.collapse.iloc[0]):
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            v = g.collapse.iloc[0] * 100
            lo = g.ci_lo.iloc[0] * 100 if np.isfinite(g.ci_lo.iloc[0]) else v
            hi = g.ci_hi.iloc[0] * 100 if np.isfinite(g.ci_hi.iloc[0]) else v
            vals.append(v)
            los.append(max(v - lo, 0.0))
            his.append(max(hi - v, 0.0))
        col, mk, lab = cfg[setting]
        if setting == "unsw_pair":
            special = np.array([r in VAL_RULES_FIG4 for r in rules])
            for mask, face, label in [(~special, col, lab), (special, "white", None)]:
                ax.errorbar(np.array(vals)[mask], (y + (k - 1) * 0.30)[mask],
                            xerr=[np.array(los)[mask], np.array(his)[mask]], fmt=mk, ms=2.9,
                            mfc=face, mec=col, mew=0.8, ecolor="0.35",
                            elinewidth=0.6, capsize=1.3, lw=0, label=label, zorder=3)
        else:
            ax.errorbar(vals, y + (k - 1) * 0.30, xerr=[los, his], fmt=mk, ms=2.9,
                        mfc=col, mec=col, mew=0.8, ecolor="0.35",
                        elinewidth=0.6, capsize=1.3, lw=0, label=lab, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r + "†" if r in VAL_RULES_FIG4 else r for r in rules],
                       fontsize=SIZE["tick"])
    ax.set_xlabel("collapse rate (%)")
    ax.set_xlim(0, 105)

    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), handlelength=1.2, columnspacing=1.1,
              handletextpad=0.4)
    ax.text(0.0, -0.235, "† NF-UNSW-NB15: fold 0 only (900 runs)",
            transform=ax.transAxes, ha="left", va="top", fontsize=SIZE["note"])
    panel(ax, "A")

    ax2 = axes[1]
    rules_fpr = [c[len("fpr_"):] for c in clean.columns if c.startswith("fpr_")]
    days = [("cic18", "CSE-CIC-IDS2018 (clean windows)"), ("unsw", "NF-UNSW-NB15 (clean windows)")]
    yy = np.arange(len(rules_fpr))[::-1]
    for k, (dsname, lab) in enumerate(days):
        sub = clean[clean.dataset == dsname]
        vals, los, his = [], [], []
        for r in rules_fpr:
            col = "fpr_" + r
            if col not in sub:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            v = pd.to_numeric(sub[col], errors="coerce").dropna()
            k_ = int((v > 0.05).sum())
            ph, lo, hi = wilson(k_, len(v))
            vals.append(ph * 100); los.append(lo * 100); his.append(hi * 100)
        col, ha = (ORANGE, "///") if k == 0 else (AQUA, "xx")
        ax2.errorbar(vals, yy + (k - 0.5) * 0.34, xerr=[np.array(vals) - np.array(los),
                     np.array(his) - np.array(vals)], fmt="o", ms=3.2,
                     mfc=col, mec=col, mew=0.5, ecolor="0.3", elinewidth=0.6,
                     capsize=1.4, lw=0, label=lab)
    base = "fpr_fixed0.5"
    if base in clean.columns:
        s = clean[clean.dataset == "cic18"]
        v = pd.to_numeric(s[base], errors="coerce").dropna()
        ph, _, _ = wilson(int((v > 0.05).sum()), len(v))
        ax2.axvline(ph * 100, color="black", lw=1.0, ls="--")
        ax2.text(ph * 100, 1.01, "fixed 0.5", transform=ax2.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=SIZE["note"])
    ax2.set_yticks(yy)
    ax2.set_yticklabels(rules_fpr, fontsize=SIZE["tick"])
    ax2.set_xlabel("runs with FPR > 5% (%)")
    ax2.set_xlim(-2, 105)
    ax2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               ncol=1, handlelength=1.2, handletextpad=0.4)
    panel(ax2, "B")
    fig.tight_layout()
    _save(fig, out)


def fig5(fam, out, stride=3):
    def pooled(dsname):
        files = sorted(glob.glob("scores_v2/%s/*.npz" % dsname))[::stride]
        vals = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            pos = z["label"] == 1
            if pos.any():
                vals.append(z["score"][pos])
        return np.concatenate(vals) if vals else np.array([])

    unseen = np.sort(pooled("cic18"))
    seen = np.sort(pooled("unsw"))
    fig, ax = plt.subplots(figsize=(5.8, 4.3))
    for v, lab, ls, col in [(unseen, "CSE-CIC-IDS2018 test day (all families unseen)",
                             "-", ORANGE),
                            (seen, "NF-UNSW-NB15 test days (all families seen)",
                             (0, (5, 2)), AQUA)]:
        if not len(v):
            continue
        y = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, y, ls=ls, color=col, lw=1.4, label=lab, drawstyle="steps-post")
    ax.vlines(0.5, 0, 1.0, color="black", lw=1.0, ls=":")
    ax.text(0.47, 0.17, "cut 0.5", ha="right", va="bottom",
            fontsize=SIZE["note"])
    ax.set_xlabel("score of positive sequences")
    ax.set_ylabel("cumulative share")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              fontsize=SIZE["note"], handlelength=2.0, ncol=1)


    axin = ax.inset_axes([0.105, 0.700, 0.335, 0.215])
    data, labs = [], []
    for dsname, lab in [("CIC18", "unseen"), ("UNSW", "seen")]:
        v = fam[(fam.dataset == dsname) & (fam.n_family >= 30)]["auc"].dropna().values
        if len(v):
            data.append(v)
            labs.append(lab)
    if data:
        bp = axin.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                          medianprops=dict(color="black", lw=0.9),
                          boxprops=dict(lw=0.5), whiskerprops=dict(lw=0.5),
                          capprops=dict(lw=0.5))
        for patch, fc, ha in zip(bp["boxes"], [ORANGE, AQUA], ["", ""]):
            patch.set_facecolor(fc)
            patch.set_hatch(ha)
        axin.set_xticks([1, 2])
        axin.set_xticklabels(labs, fontsize=8.0)
        axin.tick_params(labelsize=8.0, pad=1)
        axin.axhline(0.5, color="black", lw=0.7, ls=":")
        axin.set_ylabel("family AUC", fontsize=8.0, labelpad=1)
        axin.set_title("within-family", fontsize=8.0, pad=2)
        for sp in axin.spines.values():
            sp.set_linewidth(0.5)
    fig.tight_layout()
    _save(fig, out)


def main():
    global LETTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_v2/final/figures")
    ap.add_argument("--font", default="Liberation Sans")
    ap.add_argument("--stride", type=int, default=3, help="Run stride for Figure 5 sampling")
    ap.add_argument("--no-letter", action="store_true",
                    help="Omit Letter pages and produce only content-cropped PDFs for self-checks")
    args = ap.parse_args()
    LETTER = not args.no_letter
    os.makedirs(args.outdir, exist_ok=True)
    style(args.font)

    wr = pd.read_csv("results_v2/wp1_winrate_s20.csv")
    tau = pd.read_csv("results_v2/wp1_tau_s20.csv")
    fig1(wr, tau, os.path.join(args.outdir, "Fig1.pdf"))
    print("Fig1 ok")

    fig2(os.path.join(args.outdir, "Fig2.pdf"))
    print("Fig2 ok")

    d = load_runs()
    bdata = pd.read_csv("results_v2/r2/R2_fig3b_data.csv")
    print("  [Fig3B] Panel B data: results_v2/r2/R2_fig3b_data.csv, "
          f"{len(bdata)} rows, {bdata.group.nunique()} groups")
    fig3(d, bdata, os.path.join(args.outdir, "Fig3.pdf"))
    print("Fig3 ok")

    sm = pd.read_csv("results_v2/wp4d_summary.csv")
    sm20 = pd.read_csv("results_v2/wp4d_summary_unsw_s20_val.csv")
    sm = pd.concat([sm[sm.setting != "unsw_pair"], sm20], ignore_index=True)
    clean = pd.read_csv("results_v2/wp4_clean_run.csv")
    fig4(sm, clean, os.path.join(args.outdir, "Fig4.pdf"))
    print("Fig4 ok")

    fam = pd.read_csv("results_v2/wp2b_family.csv")
    if glob.glob("scores_v2/cic18/*.npz") and glob.glob("scores_v2/unsw/*.npz"):
        fig5(fam, os.path.join(args.outdir, "Fig5.pdf"), stride=args.stride)
        print("Fig5 ok")
    else:
        print("Fig5 skipped: scores_v2/cic18 and scores_v2/unsw NPZ files are required")

    rep = os.path.join(args.outdir, "COLLISION_REPORT.txt")
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print("\n--- Text collision/overflow and layout report (%s) ---" % rep)
    for line in REPORT:
        print("  " + line)


if __name__ == "__main__":
    main()
