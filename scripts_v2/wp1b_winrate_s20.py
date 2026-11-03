#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp1_robustness as W

V2_DATASETS = [
    ("CIC17", "results_v2/det_v2_cic17.db", False),
    ("CIC18", "results_v2/det_v2_cic18.db", False),
    ("UNSW", "results_v2/det_unsw_s20.db", True),
]

Z = 1.959963984540054


def wilson(k: float, n: int):
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + Z * Z / n
    c = p + Z * Z / (2.0 * n)
    h = Z * np.sqrt(p * (1.0 - p) / n + Z * Z / (4.0 * n * n))
    return (c - h) / d, (c + h) / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_v2")
    args = ap.parse_args()

    W.DATASETS = V2_DATASETS
    d = W.load_all()
    print("Total runs %d" % len(d))
    print(d.groupby("ds_key").size())

    rows = []
    for label, _p, has_folds in V2_DATASETS:
        keys = [f"{label}:f0", f"{label}:f1"] if has_folds else [label]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            for sub in W.SUBSETS:
                for metric, eps_list in W.EPS_BY_METRIC.items():
                    M, idx, n_bad = W.group_matrix(d, ds_key, sub, metric)
                    n = len(M)
                    if n == 0:
                        continue
                    bi = W.boot_idx(n, W.NBOOT, W.SEED)
                    for eps in eps_list:
                        S = W.win_shares(M, eps)
                        wr = S.mean(axis=0)
                        boot = S[bi].mean(axis=1)
                        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
                        for j, m in enumerate(W.MODELS):
                            k = round(float(wr[j]) * n)
                            wl, wh = wilson(k, n)
                            rows.append(dict(
                                dataset=label, fold=fold, route_subset=sub,
                                metric=metric, eps=eps, model=m,
                                winrate=round(float(wr[j]), 6),
                                ci_lo=round(float(lo[j]), 6),
                                ci_hi=round(float(hi[j]), 6),
                                wilson_lo=round(float(wl), 6),
                                wilson_hi=round(float(wh), 6),
                                n_groups=int(n), n_groups_dropped_na=n_bad,
                            ))
    out = pd.DataFrame(rows)
    pth = os.path.join(args.outdir, "wp1_winrate_s20.csv")
    out.to_csv(pth, index=False)
    print("Wrote %s  (%d rows)" % (pth, len(out)))


    m = out[(out.route_subset == "all") & (out.eps == 0.0)]
    piv = m.pivot_table(index=["dataset", "fold", "metric"],
                        columns="model", values="winrate")
    piv = piv[[c for c in W.MODELS if c in piv.columns]]
    print("\n=== Main table (all, eps=0) ===")
    print((piv * 100).round(1).to_string())


    print("\n=== Comparison A: tw30s-only, eps=0 ===")
    a = out[(out.route_subset == "tw30s") & (out.eps == 0.0)]
    pa = a.pivot_table(index=["dataset", "fold", "metric"],
                       columns="model", values="winrate")
    print((pa[[c for c in W.MODELS if c in pa.columns]] * 100).round(1).to_string())

    print("\n=== Comparison B: all, roc_auc, eps=0.001 ===")
    b = out[(out.route_subset == "all") & (out.metric == "roc_auc") & (out.eps == 0.001)]
    pb = b.pivot_table(index=["dataset", "fold", "metric"],
                       columns="model", values="winrate")
    print((pb[[c for c in W.MODELS if c in pb.columns]] * 100).round(1).to_string())


    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    tau = W.wp1b(d, tmp)
    tp = os.path.join(args.outdir, "wp1_tau_s20.csv")
    tau.to_csv(tp, index=False)
    shutil.rmtree(tmp)
    print("\nWrote %s  (%d rows)" % (tp, len(tau)))
    tv = tau[(tau.route_subset == "all") & (tau.aggregation == "winrate")]
    print(tv[["dataset", "fold", "pair", "tau", "exact_perm_p",
              "boot_tau_ci_lo", "boot_tau_ci_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
