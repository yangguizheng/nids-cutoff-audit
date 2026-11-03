#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _c in (os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "src", "data"),
           os.path.join(_ROOT, "src", "models")):
    if os.path.isdir(_c):
        sys.path.insert(0, _c)

import seqdiff
from seqdiff import SeqConfig, add_group_ids, load_parquet_dir, segment_rows

CIC18_DIR = os.path.join(_ROOT, "data", "processed", "cic18")
OUT = os.path.join(_ROOT, "results_v2", "e1")


ROUTES = [
    ("tw30s", dict(segment="time_window", window_s=30.0, convention="by_start",
                   truncate="tail", padding="non_padded")),
    ("host", dict(segment="host", padding="non_padded")),
    ("service", dict(segment="service", padding="non_padded")),
]
T = 20


def label_strings(df: pd.DataFrame) -> np.ndarray:
    lab = df["Label"]
    if pd.api.types.is_numeric_dtype(lab):
        atk = pd.to_numeric(lab, errors="coerce").fillna(0).to_numpy() != 0
        return np.where(atk, "ATTACK", "BENIGN")
    raw = lab.astype(str).str.strip().str.upper().to_numpy()
    return np.where(raw != "BENIGN", raw, "BENIGN")


def seq_families(df: pd.DataFrame, route: str, kw: dict, code: np.ndarray) -> np.ndarray:

    cfg = SeqConfig(T=T, seed=0, **kw)
    segs = segment_rows(df, cfg)
    out = np.zeros(len(segs), dtype=np.int32)
    for s, rows in enumerate(segs):
        keep = rows[-T:] if len(rows) > T else rows
        ck = code[keep]
        ck = ck[ck > 0]
        if len(ck) == 0:
            out[s] = 0
            continue
        vals, cnts = np.unique(ck, return_counts=True)
        out[s] = int(vals[int(np.argmax(cnts))])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    inv_rows, day_rows = [], []
    t0 = time.time()

    for d in args.days:
        t1 = time.time()
        df = load_parquet_dir(CIC18_DIR, days=[d])
        df = add_group_ids(df)
        raw = label_strings(df)
        names = sorted({s for s in raw if s != "BENIGN"})
        lut = {s: i + 1 for i, s in enumerate(names)}
        code = np.zeros(len(raw), dtype=np.int32)
        for s, c in lut.items():
            code[raw == s] = c
        date = str(pd.to_datetime(df["Timestamp"].iloc[0]).date())
        n_flow_by_lab = {s: int((raw == s).sum()) for s in ["BENIGN"] + names}
        n_benign = n_flow_by_lab["BENIGN"]
        n_atk = len(raw) - n_benign

        route_stats = {}
        for route, kw in ROUTES:
            fam = seq_families(df, route, kw, code)
            n_seq = len(fam)
            pos = int((fam > 0).sum())
            route_stats[route] = dict(n_seq=n_seq, n_pos=pos,
                                      pos_rate=(pos / n_seq if n_seq else float("nan")))
            for s in ["BENIGN"] + names:
                c = 0 if s == "BENIGN" else lut[s]
                inv_rows.append(dict(day=d, date=date, route=route, label=s,
                                     n_flows=int(n_flow_by_lab[s]),
                                     n_seq_T20=int((fam == c).sum())))
        rec = dict(day=d, date=date, n_flows=len(df), n_benign_flows=n_benign,
                   n_attack_flows=n_atk,
                   pos_rate_flows=(n_atk / len(df) if len(df) else float("nan")),
                   labels=";".join(names))
        for route, _ in ROUTES:
            rec["n_seq_" + route] = route_stats[route]["n_seq"]
            rec["n_pos_seq_" + route] = route_stats[route]["n_pos"]
            rec["pos_rate_" + route] = route_stats[route]["pos_rate"]
        day_rows.append(rec)
        print("[d%d %s] flows %s (positive %s, %.4f)  " % (d, date, f"{len(df):,}",
                                                 f"{n_atk:,}", rec["pos_rate_flows"])
              + "  ".join("%s %d/%d" % (r, route_stats[r]["n_pos"], route_stats[r]["n_seq"])
                          for r, _ in ROUTES)
              + "  (%.1fs)" % (time.time() - t1), flush=True)

    pd.DataFrame(inv_rows).to_csv(os.path.join(args.out_dir, "inventory.csv"), index=False)
    day_df = pd.DataFrame(day_rows)
    day_df.to_csv(os.path.join(args.out_dir, "inventory_day.csv"), index=False)


    gate_rows = []
    for grp, te in (("Web", 6), ("Inf", 8)):
        row = day_df[day_df.day == te].iloc[0]
        for route, _ in ROUTES:
            n = int(row["n_pos_seq_" + route])
            gate_rows.append(dict(group=grp, test_day=te, route=route, n_pos_test_seq=n,
                                  pass_30=(n >= 30), n_test_seq=int(row["n_seq_" + route])))
    g = pd.DataFrame(gate_rows)
    g.to_csv(os.path.join(args.out_dir, "gates.csv"), index=False)
    print("\nFeasibility threshold (positive sequences ≥ 30)")
    print(g.to_string(index=False))
    for grp in ("Web", "Inf"):
        sub = g[g.group == grp]
        if not sub.pass_30.any():
            print("  !! %s group has all three routes < 30; cancel entire group under §1 threshold rule 2" % grp)
    print("\nElapsed %.1f minutes" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
