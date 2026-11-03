#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp1_robustness as W

V2_DATASETS = [
    ("CIC17", "results_v2/det_v2_cic17.db", False),
    ("CIC18", "results_v2/det_v2_cic18.db", False),
    ("UNSW", "results_v2/det_unsw_s20.db", True),
]


def top1_disagree(Ma, Mb):

    n = len(Ma)
    diff = 0
    shift = []
    for i in range(n):
        a, b = Ma[i], Mb[i]
        if int(np.argmax(a)) != int(np.argmax(b)):
            diff += 1
        ra = stats.rankdata(-a, method="min")
        rb = stats.rankdata(-b, method="min")
        shift.append(np.abs(ra - rb).mean())
    return diff, n, float(np.mean(shift))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_v2")
    args = ap.parse_args()

    W.DATASETS = V2_DATASETS
    d = W.load_all()

    lines = ["# Recalculations following the Table 4 definition change (WP7 item 27)", ""]
    lines.append("Old definition `results/det_*.db` (UNSW 450 runs/5 seeds) → new definition "
                 "`det_v2_*` + `det_unsw_s20.db` (UNSW 1,800 runs/20 seeds).")
    lines.append("")


    lines.append("## 1. Pairwise wins and losses (two-sided exact sign test)")
    lines.append("")
    lines.append("| Dataset | Pair | Metric | W / L / T | p |")
    lines.append("|---|---|---|---|---|")
    pair_rows = []
    for label, _p, has_folds in V2_DATASETS:
        keys = [f"{label}:f0", f"{label}:f1"] if has_folds else [label]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            for met in ["excess", "roc_auc"]:
                M, _idx, _ = W.group_matrix(d, ds_key, "all", met)
                for ia, ib in [(1, 2), (3, 4), (0, 1)]:
                    a, b = M[:, ia], M[:, ib]
                    w = int((a > b).sum())
                    l = int((a < b).sum())
                    t = int((a == b).sum())
                    pv = stats.binomtest(w, w + l, 0.5).pvalue if (w + l) else float("nan")
                    lines.append("| %s %s | %s vs %s | `%s` | **%d / %d / %d** | %s |" % (
                        label, fold, W.MODELS[ia].upper(), W.MODELS[ib].upper(), met,
                        w, l, t, ("%.4f" % pv)))
                    pair_rows.append(dict(dataset=label, fold=fold, metric=met,
                                          model_a=W.MODELS[ia], model_b=W.MODELS[ib],
                                          wins=w, losses=l, ties=t,
                                          p_sign=round(float(pv), 8), n=w + l + t))
    pd.DataFrame(pair_rows).to_csv(
        os.path.join(args.outdir, "wp7_t4_pairs.csv"), index=False)
    lines.append("")


    lines.append("## 2. Top-1 disagreement share and mean rank shift")
    lines.append("")
    lines.append("| Dataset | Groups | Different top-1 | Mean rank shift |")
    lines.append("|---|---|---|---|")
    shift_rows = []
    for label, _p, has_folds in V2_DATASETS:
        keys = [f"{label}:f0", f"{label}:f1"] if has_folds else [label]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            Me, _, _ = W.group_matrix(d, ds_key, "all", "excess")
            Mo, _, _ = W.group_matrix(d, ds_key, "all", "roc_auc")
            diff, n, sh = top1_disagree(Me, Mo)
            lines.append("| %s | %d | %d/%d = %.1f%% | %.2f |" % (
                ds_key, n, diff, n, 100.0 * diff / n, sh))
            shift_rows.append(dict(dataset=label, fold=fold, n=n, diff=diff,
                                   frac=round(diff / n, 6), mean_shift=round(sh, 4)))
    pd.DataFrame(shift_rows).to_csv(
        os.path.join(args.outdir, "wp7_t4_shift.csv"), index=False)
    lines.append("")


    lines.append("## 3. Two shifts (metric change / dataset change)")
    lines.append("")
    wr = pd.read_csv(os.path.join(args.outdir, "wp1_winrate_s20.csv"))
    wr = wr[(wr.route_subset == "all") & (wr.eps == 0.0)]
    piv = wr.pivot_table(index=["dataset", "fold", "metric"],
                         columns="model", values="winrate")
    def g(ds, fold, met, m):
        try:
            return float(piv.loc[(ds, fold, met), m]) * 100
        except KeyError:
            return float("nan")

    rf_e, rf_a = g("CIC18", "-", "excess", "rf"), g("CIC18", "-", "roc_auc", "rf")
    tc17, tc18 = g("CIC17", "-", "excess", "tcn"), g("CIC18", "-", "excess", "tcn")
    lines.append("| Operation | Case | Magnitude |")
    lines.append("|---|---|---|")
    lines.append("| Metric change (same dataset) | RF on CIC18: `excess` %.1f%% → `ROC-AUC` %.1f%% | **+%.1fpp** |"
                 % (rf_e, rf_a, rf_a - rf_e))
    lines.append("| Dataset change (same metric) | TCN `excess`: CIC17 %.1f%% → CIC18 %.1f%% | **%.1fpp** |"
                 % (tc17, tc18, tc18 - tc17))
    lines.append("")
    lines.append("Difference between the two: **%.1fpp**." % abs((rf_a - rf_e) - (tc18 - tc17)))
    lines.append("")


    lines.append("## 4. UNSW median AUC by route (`tw30s` vs six leaky cells)")
    lines.append("")
    lines.append("| Route | LSTM | TCN | Transformer | RF | HGB |")
    lines.append("|---|---|---|---|---|---|")
    un = d[d["dataset"] == "UNSW"]
    for rs, lab in [("tw30s", "`tw30s` three cells"), ("host+service", "`host`/`service` six cells")]:
        x = un[un["route_subset"] == rs]
        cells = []
        for m in W.MODELS:
            v = x[x["model"] == m]["roc_auc"].median()
            cells.append("%.4f" % v)
        lines.append("| %s | %s |" % (lab, " | ".join(cells)))
    lines.append("")
    lines.append("(Runs: UNSW 1,800, %s with %d runs each)" % (
        "`tw30s`", int((un["route_subset"] == "tw30s").sum())))
    lines.append("")

    out = os.path.join(args.outdir, "wp7_t4_numbers.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()
