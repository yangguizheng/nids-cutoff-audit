#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "src", "data"), os.path.join(_ROOT, "src", "models")):
    sys.path.insert(0, _p)

from seqdiff import (
    SeqConfig, add_group_ids, build_sequences, load_days, load_parquet_dir, make_split,
)

ROOT_PARENT = ".."
ROUTES = ["host", "service"]
PADDINGS = ["non_padded", "repeat_last", "zero_mask"]
T = 20


def floor_of(p: float) -> float:
    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def best_model_auc(dbfile: str, cell: str, fold: str | None):

    if not os.path.exists(dbfile):
        return None, None
    con = sqlite3.connect(dbfile)
    q = "SELECT model, AVG(roc_auc) AS a FROM runs WHERE cell = ?"
    params = [cell]
    if fold is not None:
        q += " AND run_id LIKE ?"
        params.append(f"%|{fold}|%")
    q += " GROUP BY model ORDER BY a DESC"
    rows = con.execute(q, params).fetchall()
    con.close()
    if not rows:
        return None, None
    return float(rows[0][1]), str(rows[0][0])


def key_prior_scores(g_tr, y_tr, g_te, thr=0.5):

    glob = float(np.mean(y_tr)) if len(y_tr) else 0.0
    n_pos, n_all = {}, {}
    for h, y in zip(g_tr.tolist(), y_tr.tolist()):
        n_all[h] = n_all.get(h, 0) + 1
        n_pos[h] = n_pos.get(h, 0) + int(y)
    score = np.array([n_pos.get(h, 0) / n_all[h] if h in n_all else glob for h in g_te.tolist()],
                     dtype=float)
    return score, glob


def run_cell(d: pd.DataFrame, route: str, pad: str, split: str, max_segments: int,
             ds_label: str, dbfile: str):
    kw = dict(convention="by_start", truncate="tail") if route == "time_window" else {}
    cfg = SeqConfig(segment=route, T=T, padding=pad, seed=0,
                    max_segments=max_segments, **kw)
    b = build_sequences(d, cfg)
    folds = make_split(b, split, seed=0)
    g = np.asarray(b.group)
    out = []
    for fi, fold in enumerate(folds):
        tr, te = fold["train"], fold["test"]
        y_tr, y_te = b.y_bin[tr], b.y_bin[te]
        if len(te) == 0 or y_te.sum() == 0 or (y_te == 0).sum() == 0:
            continue
        score, glob = key_prior_scores(g[tr], y_tr, g[te])
        pred = (score > 0.5).astype(int)
        f1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
        auc = float(roc_auc_score(y_te, score))
        seen = set(g[tr].tolist())
        seen_rate = float(np.mean([h in seen for h in g[te].tolist()]))
        fold_id = f"f{fi}" if split == "cross_day_pair" else "-"
        bm_auc, bm_name = best_model_auc(dbfile, cfg.tag(), fold_id if split == "cross_day_pair" else None)
        out.append(dict(
            dataset=ds_label, fold=fold_id, route=route, padding=pad, cell=cfg.tag(),
            split=split, n_keys_train=int(len(seen)), n_keys_test=int(len(set(g[te].tolist()))),
            n_test=int(len(te)), test_pos_rate=round(float(y_te.mean()), 6),
            floor=round(floor_of(float(y_te.mean())), 6),
            excess=round(f1 - floor_of(float(y_te.mean())), 6),
            j1_macro_f1=round(f1, 6), j1_roc_auc=round(auc, 6),
            key_seen_rate=round(seen_rate, 6),
            global_train_atk_rate=round(glob, 6),
            best_model_auc=(None if bm_auc is None else round(bm_auc, 6)),
            best_model_name=bm_name,
            auc_gap_to_best_model=(None if bm_auc is None else round(bm_auc - auc, 6)),
        ))
        print(f"  {ds_label}{(':'+fold_id) if fold_id!='-' else ''} {cfg.tag():<40} "
              f"F1={f1:.4f} AUC={auc:.4f} seen={seen_rate:.3f} best={bm_name}:{bm_auc}", flush=True)
    del b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_v2")
    ap.add_argument("--datasets", default="cic17,cic18,unsw")
    ap.add_argument("--max-segments", type=int, default=200000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows = []

    if "cic17" in want:
        print("=== CIC17 (nids_audit engelen, five days) ===", flush=True)
        df = add_group_ids(load_days(ROOT_PARENT, "engelen"))
        for route in ROUTES:
            for pad in PADDINGS:
                rows += run_cell(df, route, pad, "cross_day", args.max_segments,
                                 "CIC17", f"{_ROOT}/results/det_cic17.db")
        del df

    if "cic18" in want:
        print("=== CIC18 (five days) ===", flush=True)
        df = add_group_ids(load_parquet_dir(f"{_ROOT}/data/processed/cic18",
                                            days=[0, 1, 2, 3, 4]))
        for route in ROUTES:
            for pad in PADDINGS:
                rows += run_cell(df, route, pad, "cross_day", args.max_segments,
                                 "CIC18", f"{_ROOT}/results/det_cic18.db")
        del df

    if "unsw" in want:
        print("=== UNSW (three days, two cross_day_pair folds) ===", flush=True)
        df = add_group_ids(load_parquet_dir(f"{_ROOT}/data/processed/unsw",
                                            ts_guard=("2015-01-01", "2015-03-01"),
                                            days=[0, 1, 2]))
        keep = set(json.load(open(f"{_ROOT}/data/features_unsw_noTTL.json", encoding="utf-8")))
        keep |= {"Label", "Timestamp", "Src IP", "Src Port", "Dst IP", "Dst Port",
                 "Protocol", "Flow Duration"}
        drop = [c for c in df.columns if c not in keep and not c.startswith("_")]
        if drop:
            df = df.drop(columns=drop)
        for route in ROUTES:
            for pad in PADDINGS:
                rows += run_cell(df, route, pad, "cross_day_pair", args.max_segments,
                                 "UNSW", f"{_ROOT}/results/det_unsw.db")
        del df

    t = pd.DataFrame(rows)
    dst = os.path.join(args.outdir, "wp5_j1.csv")
    t.to_csv(dst, index=False)
    print(f"\n{len(t)} rows -> {dst}")
    if len(t):
        hit = t.loc[(t.j1_macro_f1 - 0.9834).abs().idxmin()]
        print(f"Cell closest to 0.9834: {hit.dataset} {hit.cell} F1={hit.j1_macro_f1:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
