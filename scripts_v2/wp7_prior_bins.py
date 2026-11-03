#!/usr/bin/env python


from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
DBS = [("CIC17", "cic17", "det_v2_cic17.db"),
       ("CIC18", "cic18", "det_v2_cic18.db"),
       ("UNSW", "unsw", "det_v2_unsw.db")]
BINS = [(0.12, 0.25), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
DELTA = 0.02


def floor_of(p: float) -> float:
    return max((1 - p) / (2 - p), p / (1 + p))


def main() -> int:
    pr = pd.read_csv(os.path.join(R2, "wp4_prior.csv"))
    pk = {}
    for _, r in pr.iterrows():
        pk[(r.dataset, r.cell, int(r.fold))] = float(r.pos_rate_train)

    rows = []
    for name, ds, db in DBS:
        con = sqlite3.connect(os.path.join(R2, db))
        for rid, cell, fold, mf, ptr in con.execute(
                "select run_id, cell, fold, macro_f1, test_pos_rate from runs"):
            p_tr = pk.get((ds, cell, int(fold)))
            if p_tr is None or p_tr <= 0:
                continue
            rows.append(dict(dataset=name, ds=ds, cell=cell, fold=int(fold),
                             macro_f1=float(mf), pos_rate_test=float(ptr),
                             pos_rate_train=p_tr, mult=float(ptr) / p_tr))
        con.close()
    d = pd.DataFrame(rows)
    d["floor"] = [floor_of(p) for p in d["pos_rate_test"]]
    d["excess"] = d["macro_f1"] - d["floor"]
    d["collapse"] = (d["excess"] < DELTA).astype(int)
    print(f"Run-level data: {len(d)} runs\n")


    print("=== Table A: Prior ratios by dataset/fold (manuscript check) ===")
    t1 = (d.groupby(["dataset", "fold"])
          .agg(n=("mult", "size"), lo=("mult", "min"), hi=("mult", "max"),
               med=("mult", "median")).reset_index())
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()


    print("=== Table B: Bin → dataset/fold assignment ===")
    d["bin"] = "other"
    for lo, hi in BINS:
        m = (d["mult"] >= lo) & (d["mult"] < hi)
        d.loc[m, "bin"] = f"{lo:.2f}-{hi:.2f}x"
    ct = (d.groupby(["bin", "dataset", "fold"]).size()
          .rename("n").reset_index().sort_values(["bin", "dataset", "fold"]))
    print(ct.to_string(index=False))
    ct.to_csv(os.path.join(R2, "wp7_prior_bins_crosstab.csv"), index=False)
    print()


    print("=== Table C: Collapse rate by bin (manuscript check) ===")
    t3 = (d.groupby("bin").agg(n=("collapse", "size"), collapse=("collapse", "mean"))
          .reset_index())
    t3["attrib"] = [
        " + ".join(f"{r.dataset}{'' if r.fold < 0 else '/f%d' % r.fold}"
                   f"({r.n})" for r in ct[ct.bin == b].itertuples())
        for b in t3["bin"]]
    print(t3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    other = d[d["bin"] == "other"]
    print(f"Runs outside the four bins: {len(other)}")
    if len(other):
        print(other.groupby(["dataset", "fold"])["mult"]
              .agg(["size", "min", "max"]).to_string())

    d.drop(columns=["ds"]).to_csv(os.path.join(R2, "wp7_prior_bins.csv"), index=False)
    print(f"\n→ {R2}/wp7_prior_bins.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
