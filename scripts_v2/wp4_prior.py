#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "data"))
sys.path.insert(0, os.path.join(ROOT, "scripts_v2"))

from seqdiff import (
    SeqBatch, SeqConfig, add_group_ids, load_days, load_parquet_dir,
    make_split, segment_rows,
)
import splits_v2 as SP

R2 = os.path.join(ROOT, "results_v2")
DB = {"cic17": "det_v2_cic17.db", "cic18": "det_v2_cic18.db", "unsw": "det_v2_unsw.db",
      "cic17nd": "nd_cic17.db"}


BUILD = {
    "cic17": dict(max_segments=0, max_seq=0, split="cross_day"),
    "cic18": dict(max_segments=200_000, max_seq=0, split="cross_day"),
    "unsw": dict(max_segments=200_000, max_seq=80_000, split="cross_day_pair"),

    "cic17nd": dict(max_segments=0, max_seq=0, split="cross_day_nd"),
}


def attack_flags(df: pd.DataFrame) -> np.ndarray:

    lab = df["Label"]
    if pd.api.types.is_numeric_dtype(lab):
        return pd.to_numeric(lab, errors="coerce").fillna(0).to_numpy() != 0
    return lab.astype(str).str.upper().to_numpy() != "BENIGN"


def cells_of(ds: str) -> pd.DataFrame:
    con = sqlite3.connect(os.path.join(R2, DB[ds]))
    d = pd.read_sql_query("select distinct cell, route, padding from runs order by cell", con)
    con.close()
    return d


def load_df(ds: str, args) -> pd.DataFrame:
    if ds in ("unsw",):
        df = load_parquet_dir(
            f"{args.root}/protocol_robust_nids_20261028/data/processed/unsw",
            ts_guard=("2015-01-01", "2015-03-01"), days=args.unsw_days)
    elif ds == "cic18":
        df = load_parquet_dir(
            f"{args.root}/protocol_robust_nids_20261028/data/processed/cic18",
            days=args.cic18_days)
    else:
        df = load_days(args.root, "engelen", args.days)
    return add_group_ids(df)


def cfg_of(route: str, padding: str, cell: str, max_segments: int) -> SeqConfig:
    if route == "time_window":
        ws = float(cell.split("/")[0].replace("tw", "").replace("s", ""))
        kw = dict(segment="time_window", window_s=ws, convention="by_start", truncate="tail")
    else:
        kw = dict(segment=route)
    return SeqConfig(T=20, padding=padding, label_mode="any", seed=0,
                     max_segments=max_segments, **kw)


def strat_idx_of(y: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:

    n = len(y)
    if cap <= 0 or n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    k_pos = min(len(pos), int(round(cap * y.mean())))
    k_neg = min(len(neg), cap - k_pos)
    return np.sort(np.concatenate([
        rng.choice(pos, k_pos, replace=False) if len(pos) > k_pos else pos,
        rng.choice(neg, k_neg, replace=False) if len(neg) > k_neg else neg]))


def seg_arrays(df: pd.DataFrame, cfg: SeqConfig):

    atk = attack_flags(df)
    segs = segment_rows(df, cfg)
    n_seg_all = len(segs)
    if cfg.max_segments and n_seg_all > cfg.max_segments:
        rng = np.random.default_rng(cfg.seed)
        pick = np.sort(rng.choice(n_seg_all, size=cfg.max_segments, replace=False))
        segs = [segs[i] for i in pick]
    tsec = (df["_t_start"] - df["_t_start"].min()).dt.total_seconds().to_numpy()
    day_all = df["_day"].to_numpy() if "_day" in df.columns else None
    key = {"session": "_sid", "service": "_svid", "host": "_hid",
           "time_window": "_sid"}[cfg.segment]
    grp = df[key].to_numpy()
    n = len(segs)
    y = np.empty(n, np.int8)
    wid = np.empty(n, np.int64)
    day = np.empty(n, np.int64)
    ln = np.empty(n, np.int64)
    g = np.empty(n, np.int64)
    for i, rows in enumerate(segs):
        keep = rows[-cfg.T:] if len(rows) > cfg.T else rows
        y[i] = 1 if atk[keep].any() else 0
        wid[i] = int(tsec[keep].min())
        day[i] = int(day_all[keep].min()) if day_all is not None else -1
        ln[i] = len(rows)
        v, c = np.unique(grp[keep], return_counts=True)
        g[i] = int(v[int(np.argmax(c))])
    return y, wid, day, ln, g, n_seg_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--datasets", nargs="+", default=["cic17", "unsw"],
                    choices=["cic17", "cic18", "unsw", "cic17nd"])
    ap.add_argument("--days", nargs="+",
                    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    ap.add_argument("--cic18-days", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--unsw-days", type=int, nargs="+", default=[0, 2])
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        p = BUILD[ds]
        cells = cells_of(ds)
        print(f"\n=== {ds} (split {p['split']}, max_segments={p['max_segments']}, "
              f"max_seq={p['max_seq']}, {len(cells)} cells) ===", flush=True)
        df = load_df(ds, args)
        print(f"  Flows {len(df):,}  days {df['_day'].nunique()}", flush=True)
        for _, c in cells.iterrows():
            cfg = cfg_of(c.route, c.padding, c.cell, p["max_segments"])
            y, wid, day, ln, g, n_seg_all = seg_arrays(df, cfg)
            if c.padding == "non_padded":
                keep = ln >= cfg.T
                y, wid, day, ln, g = y[keep], wid[keep], day[keep], ln[keep], g[keep]
            n_after = len(y)
            si = strat_idx_of(y, p["max_seq"])
            y, wid, day, g = y[si], wid[si], day[si], g[si]
            b = SeqBatch(X=np.zeros((len(y), 1, 1), np.float32),
                         mask=np.ones((len(y), 1), bool), length=np.ones(len(y), np.int64),
                         y_bin=y, y_multi=None, group=g, wid=wid, day=day, meta={})


            if p["split"] in ("block_strat", "cross_day_ext", "cross_day_nd"):
                folds = SP.make_split_v2(b, p["split"], seed=0)
            else:
                folds = make_split(b, p["split"], seed=0)
            for k, f in enumerate(folds):
                tr, va, te = f["train"], f["val"], f["test"]
                rows.append(dict(
                    dataset=ds, setting=("unsw_pair" if ds == "unsw" else ds),
                    split=p["split"], cell=c.cell, fold=(k if len(folds) > 1 else -1),
                    n_seg_all=n_seg_all, n_after_sampling=n_after, n_total=len(y),
                    n_train=len(tr), n_val=len(va), n_test=len(te),
                    n_pos_train=int(y[tr].sum()), n_pos_val=int(y[va].sum()),
                    n_pos_test=int(y[te].sum()),
                    pos_rate_train=float(y[tr].mean()) if len(tr) else float("nan"),
                    pos_rate_val=float(y[va].mean()) if len(va) else float("nan"),
                    pos_rate_test=float(y[te].mean()) if len(te) else float("nan"),
                ))
            print(f"  [{c.cell}] seg={n_seg_all:,}→{n_after:,}→{len(y):,}  {len(folds)} folds  "
                  f"π_tr={rows[-1]['pos_rate_train']:.4f}  π_te={rows[-1]['pos_rate_test']:.4f}",
                  flush=True)
        del df

    d = pd.DataFrame(rows)
    out = os.path.join(R2, "wp4_prior.csv")
    d.to_csv(out, index=False)
    print(f"\n→ {out}  {d.shape}")


    print("\n=== Gate: compare n_test and test_pos_rate for every cell and fold ===")
    bad = 0
    for ds in args.datasets:
        con = sqlite3.connect(os.path.join(R2, DB[ds]))
        q = pd.read_sql_query(
            "select cell, fold, n_test, test_pos_rate from runs group by cell, fold, n_test, "
            "test_pos_rate", con)
        con.close()
        sub = d[d.dataset == ds]
        for _, r in q.iterrows():
            m = sub[(sub.cell == r.cell) & (sub.fold == (r.fold if r.fold is not None else -1))]
            if m.empty:
                print(f"  [missing] {ds} {r.cell} fold={r.fold}")
                bad += 1
                continue
            v, nn = float(m.pos_rate_test.iloc[0]), int(m.n_test.iloc[0])
            if nn != int(r.n_test) or abs(v - float(r.test_pos_rate)) > 1e-9:
                print(f"  [mismatch] {ds} {r.cell} f{r.fold}: this script n={nn} π={v:.10f} "
                      f"vs DB n={int(r.n_test)} π={float(r.test_pos_rate):.10f}")
                bad += 1
    print("Gate passed: segmentation, sampling, and DB construction are bit-identical" if bad == 0 else f"Gate failed: {bad} mismatches")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
