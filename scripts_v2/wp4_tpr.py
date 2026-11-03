#!/usr/bin/env python3


from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
S2 = os.path.join(ROOT, "scores_v2")

DATASETS = [("cic17", "cic17"), ("cic18", "cic18"), ("unsw", "unsw_pair")]
ALPHAS = [0.01, 0.001]


PAT = re.compile(
    r"^(?P<pre>[A-Za-z0-9]+?)_(?P<cell>.+?)_(?P<split>cross_day|cross_day_pair)"
    r"(?:_(?P<fold>f\d+))?_(?P<model>lstm|tcn|transformer|rf|hgb)_s(?P<seed>\d+)\.npz$"
)


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, alpha: float) -> float:

    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= alpha + 1e-15
    if not ok.any():
        return float("nan")
    return float(tpr[ok].max())


def main() -> int:
    rows = []
    miss = 0
    for ds, setting in DATASETS:
        d = os.path.join(S2, ds)
        if not os.path.isdir(d):
            print(f"[skip] {d} does not exist")
            continue
        files = sorted(os.listdir(d))
        n_ds = 0
        for fn in files:
            m = PAT.match(fn)
            if not m:
                miss += 1
                continue
            z = np.load(os.path.join(d, fn), allow_pickle=True)
            y = np.asarray(z["label"]).astype(int)
            s = np.asarray(z["score"], dtype=np.float64)
            if y.sum() == 0 or (y == 0).sum() == 0:

                row = dict(setting=setting, cell=m["cell"], model=m["model"],
                           seed=int(m["seed"]), fold=int(m["fold"][1:]) if m["fold"] else -1,
                           n_test=len(y), n_pos=int(y.sum()), test_pos_rate=float(y.mean()),
                           roc_auc=np.nan, **{f"tpr_at_fpr_{a:g}": np.nan for a in ALPHAS},
                           fpr_at_05=float("nan"))
                rows.append(row)
                n_ds += 1
                continue
            fpr, tpr, _ = roc_curve(y, s)
            auc = float(np.trapezoid(tpr, fpr))
            row = dict(
                setting=setting, cell=m["cell"], model=m["model"],
                seed=int(m["seed"]), fold=int(m["fold"][1:]) if m["fold"] else -1,
                n_test=len(y), n_pos=int(y.sum()), test_pos_rate=float(y.mean()),
                roc_auc=auc,
                fpr_at_05=float((y[s >= 0.5] == 0).sum() / max(1, (y == 0).sum())),
            )
            for a in ALPHAS:
                ok = fpr <= a + 1e-15
                row[f"tpr_at_fpr_{a:g}"] = float(tpr[ok].max()) if ok.any() else float("nan")
            rows.append(row)
            n_ds += 1
        print(f"[{setting}] {n_ds} runs")
    if miss:
        print(f"   ({miss} filenames did not follow the convention and were skipped)")

    d = pd.DataFrame(rows)
    out = os.path.join(R2, "wp4_tpr.csv")
    d.to_csv(out, index=False)
    print(f"→ {out}  {d.shape}")


    def med(g, col):
        v = g[col].dropna()
        return float(np.median(v)) if len(v) else float("nan")

    agg = []
    for (setting, cell), g in d.groupby(["setting", "cell"]):
        agg.append(dict(
            setting=setting, cell=cell, n_runs=len(g),
            pos_rate_test=med(g, "test_pos_rate"),
            auc=med(g, "roc_auc"),
            tpr_fpr1=med(g, "tpr_at_fpr_0.01"),
            tpr_fpr01=med(g, "tpr_at_fpr_0.001"),
            fpr_at_05=med(g, "fpr_at_05"),
        ))
    a = pd.DataFrame(agg)
    a.to_csv(os.path.join(R2, "wp4_tpr_by_cell.csv"), index=False)

    L = ["# WP4c: Operational operating points (TPR@FPR)", ""]
    L.append("Threshold-independent metric, computed per run and then median-aggregated. **This column is unaffected by prior correction**—"
             "it measures ranking, not a threshold.")
    L.append("")
    for setting, g in a.groupby("setting"):
        L.append(f"## {setting}")
        L.append("")
        L.append("| Cell | Runs | Test positive rate | AUC | TPR@FPR=1% | TPR@FPR=0.1% | FPR@0.5 |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            L.append(f"| `{r.cell}` | {int(r.n_runs)} | {r.pos_rate_test:.4f} | "
                     f"{r.auc:.4f} | {r.tpr_fpr1:.4f} | {r.tpr_fpr01:.4f} | {r.fpr_at_05:.4f} |")
        L.append("")
    path = os.path.join(R2, "WP4C_SUMMARY.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
