#!/usr/bin/env python


from __future__ import annotations

import csv
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
FIN = os.path.join(R2, "final")
OUT = os.path.join(FIN, "T1_EXTRA.csv")

DBS = [("CIC17", "det_v2_cic17.db"), ("CIC18", "det_v2_cic18.db"), ("UNSW", "det_unsw_s20.db")]
DELTAS = [0.005, 0.02, 0.05]


KEYS = ["route", "padding", "model"]

ROWS = []


def emit(tag, value, source, note="", expect=None):
    ok = ""
    if expect is not None:
        try:
            g = float(str(value).replace("%", "").replace(",", ""))
            e = float(str(expect).replace("%", "").replace(",", ""))
            nd = len(str(expect).split(".")[1]) if "." in str(expect) else 0
            ok = 1 if abs(round(g, nd) - round(e, nd)) < 1e-9 else 0
        except ValueError:
            ok = 1 if str(value) == str(expect) else 0
        if ok == 0:
            print("  !! %-40s got %-12s expect %s" % (tag, value, expect))
    ROWS.append(dict(tag=tag, value=value, source=source,
                     command="scripts_v2/t1_fix_extra.py",
                     expect="" if expect is None else expect, expect_ok=ok, note=note))


def floor_of(p):
    p = np.asarray(p, float)
    return np.maximum((1.0 - p) / (2.0 - p), p / (1.0 + p))


def load(db):
    con = sqlite3.connect(os.path.join(R2, db))
    d = pd.read_sql_query("SELECT * FROM runs", con)
    con.close()
    d["floor"] = floor_of(d["test_pos_rate"].values)
    d["excess"] = d["macro_f1"] - d["floor"]
    return d


def main():
    data = {}
    for ds, db in DBS:
        data[ds] = load(db)
        print("%-6s %-22s %5d runs | route %s | padding %s | model %d | fold %s"
              % (ds, db, len(data[ds]), sorted(data[ds].route.unique()),
                 sorted(data[ds].padding.unique()), data[ds].model.nunique(),
                 sorted(data[ds].fold.unique())))


    print("\n[1] δ sensitivity (collapse rate)")
    exp_delta = {("CIC18", 0.005): "54.6%", ("CIC18", 0.05): "56.6%",
                 ("CIC17", 0.005): "0.1%", ("CIC17", 0.05): "2.3%"}
    for ds, db in DBS:
        d = data[ds]
        for dl in DELTAS:
            k = int((d["excess"] < dl).sum())
            rate = k / len(d)
            emit("DELTA.%s.r%s" % (ds, ("%g" % dl).replace(".", "")),
                 "%.1f%%" % (100 * rate), "%s excess < δ=%g" % (db, dl),
                 "%d/%d" % (k, len(d)), exp_delta.get((ds, dl)))
            print("   %-6s δ=%-5g %6.2f%%  (%d/%d)" % (ds, dl, 100 * rate, k, len(d)))


    print("\n[2] within-cell seed SD (cell = route × padding × model; fold and seed are replicates)")
    exp_sd = {"CIC17": "0.007", "CIC18": "0.048", "UNSW": "0.023"}
    for ds, db in DBS:
        d = data[ds]
        for metric, suf in (("roc_auc", "auc"), ("macro_f1", "f1")):
            g = d.groupby(KEYS)[metric].agg(["std", "count"])
            g = g[g["count"] >= 5]
            emit("SEEDSD.%s.%s" % (ds, suf), "%.3f" % float(g["std"].mean()),
                 "%s grouped by %s; arithmetic mean of within-group %s SD" % (db, "×".join(KEYS), metric),
                 "n_cell=%d, replicates per cell %s" % (len(g), sorted(set(g["count"].astype(int)))),
                 exp_sd.get(ds) if suf == "auc" else None)
            if suf == "auc":
                print("   %-6s AUC: mean SD %.6f → %.3f | median SD %.6f | n_cell %d | per cell %s"
                      % (ds, g["std"].mean(), round(g["std"].mean(), 3), g["std"].median(), len(g),
                         sorted(set(g["count"].astype(int)))))
            else:
                print("   %-6s F1 : mean SD %.6f" % (ds, g["std"].mean()))


    print("\n[3] Threshold ratio: manuscript states differ by a factor of about 50")
    d = data["CIC18"]
    col = d[d.excess >= 0.02].thr_oracle.median()
    thr = d[d.excess < 0.02].thr_oracle.values
    emit("CUT.cic18.factor_med", "%.1f" % (col / np.median(thr)),
         "det_v2_cic18.db: non-collapse median thr_oracle / collapse median thr_oracle (including +inf)",
         "Manuscript states about 50; excluding +inf gives CUT.cic18.factor = 64.2", "50")
    print("   %.4f / %.6f = %.2f" % (col, np.median(thr), col / np.median(thr)))


    old = []
    if os.path.exists(OUT):
        old = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
    newtags = {r["tag"] for r in ROWS}
    keep = [r for r in old if r["tag"] not in newtags]
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "value", "source", "command", "expect", "expect_ok", "note"])
        w.writeheader()
        for r in keep + ROWS:
            w.writerow(r)
    print("\n→ %s  wrote %d new rows, retained %d old rows" % (OUT, len(ROWS), len(keep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
