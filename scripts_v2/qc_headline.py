#!/usr/bin/env python


from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DELTA = 0.02


def floor_of(p):
    p = np.asarray(p, dtype=float)
    return np.maximum((1.0 - p) / (2.0 - p), p / (1.0 + p))


def load(name):
    con = sqlite3.connect(os.path.join(ROOT, "results", name))
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    d["floor"] = floor_of(d["test_pos_rate"].values)
    d["excess"] = d["macro_f1"] - d["floor"]
    d["collapsed"] = d["excess"] < DELTA
    return d


def main():
    ok = True
    print("=" * 76)
    print("Headline 1: CIC18 collapse rate 506/900 = 56.2%")
    print("=" * 76)
    d18 = load("det_cic18.db")
    n, c = len(d18), int(d18["collapsed"].sum())
    print("  Actual %d/%d = %.1f%%   (expected 506/900 = 56.2%%)" % (c, n, 100.0 * c / n))
    ok &= (c == 506 and n == 900)

    print("")
    print("=" * 76)
    print("Headline 2: ROC-AUC >= 0.95 in 78.9% of collapsed runs (399/506)")
    print("=" * 76)
    coll = d18[d18["collapsed"]]
    sel = coll[coll["roc_auc"] >= 0.95]
    print("  Actual %d/%d = %.1f%%   (expected 399/506 = 78.9%%)"
          % (len(sel), len(coll), 100.0 * len(sel) / len(coll)))
    print("  Median ROC-AUC for these runs: %.4f" % sel["roc_auc"].median())
    print("  All collapsed runs: median fpr %.4f, median f1_pos %.4f"
          % (coll["fpr"].median(), coll["f1_pos"].median()))
    ok &= (len(sel) == 399 and len(coll) == 506)

    print("")
    print("=" * 76)
    print("Headline 3: collapse rates across three datasets: CIC17 0.6% / CIC18 56.2% / UNSW 13.1%")
    print("=" * 76)
    exp = {"det_cic17.db": (5, 900, 0.6), "det_cic18.db": (506, 900, 56.2),
           "det_unsw.db": (59, 450, 13.1)}
    for f, (ec, en, ep) in exp.items():
        d = load(f)
        cc, nn = int(d["collapsed"].sum()), len(d)
        flag = "OK" if (cc == ec and nn == en) else "MISMATCH"
        print("  %-16s %4d/%4d = %5.1f%%   expected %4d/%4d = %5.1f%%   %s"
              % (f, cc, nn, 100.0 * cc / nn, ec, en, ep, flag))
        ok &= (cc == ec and nn == en)

    print("")
    print("=" * 76)
    print("Headline 4: 82.8% of collapsed runs detect no positives (f1_pos == 0, 419/506)")
    print("=" * 76)
    nz = int((coll["f1_pos"] == 0).sum())
    print("  Actual %d/%d = %.1f%%   (expected 419/506 = 82.8%%)"
          % (nz, len(coll), 100.0 * nz / len(coll)))
    print("  Comparison: only %.1f%% have fpr == 0 (FPR < 0.001 in %.1f%% of runs)"
          % (100.0 * (coll["fpr"] == 0).mean(), 100.0 * (coll["fpr"] < 1e-3).mean()))
    print("  → Collapsed runs produce almost no alerts rather than excessive false alarms")
    ok &= (nz == 419)

    print("")
    print("Overall: %s" % ("All consistent" if ok else "Inconsistencies found; stop and report"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
