#!/usr/bin/env python


from __future__ import annotations

import csv
import glob
import json
import os
import sqlite3

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
FIN = os.path.join(R2, "final")


def cnt_db(path):
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p):
        return 0
    con = sqlite3.connect(p)
    try:
        n = con.execute("select count(*) from runs").fetchone()[0]
    except sqlite3.Error:
        n = 0
    con.close()
    return n


def cnt_nd():
    p = os.path.join(R2, "normal_day_fpr_v2_s20.json")
    return len(json.load(open(p))["runs"])


ROWS = [

    ("v2-core", "CIC-IDS2017", "results_v2/det_v2_cic17.db", "0-19", "no", "yes",
     "Tables 4, 5, 8, 9, 11, 12, 14; Figs 2-4"),
    ("v2-core", "CSE-CIC-IDS2018", "results_v2/det_v2_cic18.db", "0-19", "no", "yes",
     "Tables 4-9, 11, 12, 14; Figs 2-5"),
    ("v2-core", "NF-UNSW-NB15 (2 folds)", "results_v2/det_unsw_s20.db", "0-19", "no", "yes",
     "Tables 4, 5, 6, 9-12, 14; Fig 3"),
    ("v2-nocw", "CIC-IDS2017", "results_v2/det_nocw_cic17.db", "0-19", "no", "yes",
     "Table 6 (M1), Supplemental Table S7"),
    ("v2-nocw", "CSE-CIC-IDS2018", "results_v2/det_nocw_cic18.db", "0-19", "no", "yes",
     "Table 6 (M1), Supplemental Table S7"),
    ("v2-nocw", "NF-UNSW-NB15 (2 folds)", "results_v2/det_nocw_unsw.db", "0-19", "no", "yes",
     "Table 6 (M1)"),
    ("v2-splits", "CSE-CIC-IDS2018 (S1 block-stratified, S2 extended)", "results_v2/splits_cic18.db",
     "0-9", "no", "yes", "Table 6 (M2), Supplemental Table S11"),
    ("v2-val-dump", "CIC-IDS2017", "results_v2/det_cic17_val.db", "0-19", "no", "yes",
     "Table 12 (validation cut-off), Supplemental Table S8"),
    ("v2-val-dump", "CSE-CIC-IDS2018", "results_v2/det_cic18_val.db", "0-19", "no", "yes",
     "Table 12 (validation cut-off), Supplemental Table S8"),
    ("v2-val-dump", "NF-UNSW-NB15 (2 folds)", "results_v2/det_unsw_val.db", "0-4", "no", "yes",
     "Table 12 (validation cut-off row only; 20-seed val scores not dumped)"),
    ("v2-nd", "CIC-IDS2017 Monday (attack-free)", "results_v2/nd_cic17.db", "0-19", "no", "yes",
     "Table 13; Supplemental Table S9"),
    ("v2-nd", "NF-UNSW-NB15 23 Jan 2015 (attack-free)", ("nd_json",), "0-19", "no", "yes",
     "Table 13"),
    ("v2-e1", "CSE-CIC-IDS2018 (seen-family / unseen-family / prior-matched)", 
     "results_v2/e1/e1_runs.db", "0-19", "no", "yes", "Results (E1), Supplemental Table S12"),
    ("v1-unsupported", "CIC-IDS2017 + CSE-CIC-IDS2018 + NF-UNSW-NB15 (v1 sweeps, probes)",
     ("v1",), "0-19 / 0-9 / 0-4", "not fixed", "yes", "Does not support any conclusion (v1 did not fix cudnn.benchmark and is not bit-reproducible)"),
]


def main():
    os.makedirs(FIN, exist_ok=True)
    out = []
    tot_cite = tot_uns = 0
    for batch, ds, src, seeds, cudnn, retr, used in ROWS:
        if isinstance(src, tuple) and src[0] == "nd_json":
            n = cnt_nd()
        elif isinstance(src, tuple) and src[0] == "v1":
            n = 3120 + 2580 + 450 + 108 + 1 + 5 * 3
            n = 0
            for f in ("results/main.db", "results/cic18.db", "results/unsw.db",
                      "results/det_cic17.db", "results/det_cic18.db", "results/det_unsw.db",
                      "results/runs.db", "results/repro1.db",
                      "results_v2/_probe/probe_det_cic17.db",
                      "results_v2/_probe/probe_det_cic18.db",
                      "results_v2/_probe/probe_det_unsw.db"):
                n += cnt_db(f)
        else:
            n = cnt_db(src)
        if n == 0:
            print("!! %s / %s found 0 runs" % (batch, ds))
        out.append(dict(batch=batch, dataset=ds, n_runs=n, seeds=seeds,
                        cudnn_benchmark=cudnn, retrained=retr,
                        used_in_manuscript_section=used))
        if batch.startswith("v1"):
            tot_uns += n
        else:
            tot_cite += n

    p = os.path.join(FIN, "RUN_LEDGER.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["batch", "dataset", "n_runs", "seeds",
                                          "cudnn_benchmark", "retrained(yes/no)",
                                          "used_in_manuscript_section"])
        w.writeheader()
        for r in out:
            r["retrained(yes/no)"] = r.pop("retrained")
            w.writerow(r)
        w.writerow(dict(batch="TOTAL citable", dataset="(total)", n_runs=tot_cite, seeds="",
                        cudnn_benchmark="no", **{"retrained(yes/no)": "yes"},
                        used_in_manuscript_section="All manuscript numbers"))
        w.writerow(dict(batch="TOTAL non-supporting", dataset="(total)", n_runs=tot_uns, seeds="",
                        cudnn_benchmark="not fixed", **{"retrained(yes/no)": "yes"},
                        used_in_manuscript_section="Does not support any conclusion"))
    print("→ %s (%d rows; %d citable runs; %d non-supporting runs)" % (p, len(out), tot_cite, tot_uns))
    for r in out:
        print("   %-14s %-58s %5d" % (r["batch"], r["dataset"][:56], r["n_runs"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
