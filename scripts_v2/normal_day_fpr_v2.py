#!/usr/bin/env python


from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_pilot as RP
from data.seqdiff import (
    SeqConfig, add_group_ids, build_sequences, load_parquet_dir,
)
from sklearn.metrics import f1_score, roc_auc_score

ROOT = ".."

ANALYSIS_PLAN_COMMIT = "e0dc336"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def floor_of(p: float) -> float:
    return max((1 - p) / (2 - p), p / (1 + p))


def otsu_thr(x) -> float:

    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 4:
        return 0.5
    cs = np.cumsum(x)
    i = np.arange(1, n, dtype=np.float64)
    w0 = i / n
    w1 = 1.0 - w0
    m0 = cs[:-1] / i
    m1 = (cs[-1] - cs[:-1]) / (n - i)
    v = w0 * w1 * (m0 - m1) ** 2
    return float((x[int(np.argmax(v))] + x[int(np.argmax(v)) + 1]) / 2.0)


def build_fold(b, te_day: int, norm_day: int):

    d = np.asarray(b.day)
    days = sorted(set(d.tolist()))
    atk = [x for x in days if x != norm_day]
    tr_days = [x for x in atk if x != te_day]
    tr_all = np.where(np.isin(d, tr_days))[0]
    wid = np.asarray(b.wid)
    order = tr_all[np.argsort(wid[tr_all], kind="mergesort")]
    n_va = max(1, int(len(order) * 0.2))
    return dict(train=order[:-n_va], val=order[-n_va:],
                test=np.where(d == te_day)[0], norm=np.where(d == norm_day)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--days", default="0,1,2", help="Three UNSW days")
    ap.add_argument("--cells", default="", help="Comma-separated substring filters")
    ap.add_argument("--models", default="lstm,tcn,transformer,rf,hgb")
    ap.add_argument("--seeds", default="0 1 2")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--max-train", type=int, default=80000,
                    help="Stratified downsampling only on training; keep test and norm complete")
    ap.add_argument("--max-segments", type=int, default=200000)
    ap.add_argument("--feature-subset", default="data/features_unsw_noTTL.json")
    ap.add_argument("--out", default="results/normal_day_fpr.json")
    ap.add_argument("--probe", action="store_true", help="Report sequence counts for three days without training")
    ap.add_argument("--cudnn-benchmark", type=int, default=0)
    args = ap.parse_args()

    RP.CUDNN_BENCH = bool(args.cudnn_benchmark)

    days = [int(x) for x in args.days.split(",")]
    df = add_group_ids(load_parquet_dir(
        f"{args.root}/protocol_robust_nids_20261028/data/processed/unsw",
        ts_guard=("2015-01-01", "2015-03-01"), days=days))
    if args.feature_subset:
        keep = set(json.load(open(args.feature_subset, encoding="utf-8")))
        keep |= set(RP.IDENTITY_COLS) | set(RP.LABEL_COLS)
        drop = [c for c in df.columns if c not in keep and not c.startswith("_")]
        df = df.drop(columns=drop)
    pm = RP._pos_mask(df["Label"])
    print(f"UNSW: {len(df):,} flows, attack share {pm.mean():.4f}")
    for dd in sorted(df["_day"].unique()):
        m = df["_day"] == dd
        print(f"   day{dd}: {int(m.sum()):,} flows, attack share {pm[m].mean():.6f}")

    norm_day = min((dd for dd in df["_day"].unique() if pm[df["_day"] == dd].sum() == 0),
                   default=None)
    if norm_day is None:
        raise SystemExit("No attack-free day found")
    print(f"→ attack-free day = day{norm_day}")

    cells = list(RP.CELLS)
    if args.cells:
        subs = [s for s in args.cells.split(",") if s]
        cells = [c for c in cells if any(s in f"{c[0]}/{c[1]['padding']}" for s in subs)]
    models = [m for m in args.models.split(",") if m]
    seeds = [int(x) for x in args.seeds.split()]

    runs, t_all = [], time.time()
    for route, kw in cells:
        cfg = SeqConfig(segment=route, T=args.T, seed=0,
                        max_segments=args.max_segments, **kw)
        t0 = time.time()
        b = build_sequences(df, cfg)
        cell = cfg.tag()
        d = np.asarray(b.day)
        cnt = {int(x): int((d == x).sum()) for x in sorted(set(d.tolist()))}
        print(f"\n[{cell}] Sequences {len(b.wid):,}  by day {cnt}  "
              f"positive rate {b.meta['pos_rate']:.4f}  built in {time.time()-t0:.0f}s")
        if args.probe:
            continue

        for te_day in sorted(x for x in set(d.tolist()) if x != norm_day):
            fold = build_fold(b, int(te_day), int(norm_day))
            tr, va, te, nd = fold["train"], fold["val"], fold["test"], fold["norm"]
            if int(b.y_bin[tr].sum()) < 5 or int(b.y_bin[te].sum()) < 5:
                print(f"  [skip] te=day{te_day}: insufficient classes")
                continue
            tr = RP.cap_train_indices(b.y_bin, tr, args.max_train, seed=0)

            ev = np.concatenate([va, te, nd])
            y_ev = np.concatenate([b.y_bin[va], b.y_bin[te], b.y_bin[nd]])
            n_va, n_te, n_nd = len(va), len(te), len(nd)
            print(f"  fold te=day{te_day}: train {len(tr):,}(pos {int(b.y_bin[tr].sum()):,}) "
                  f"val {len(va):,} test {n_te:,}(pos {int(b.y_bin[te].sum()):,}) "
                  f"norm {n_nd:,}(pos 0)")

            mu, sd = RP.fit_scaler(b.X[tr], b.mask[tr])
            Xn = RP.apply_scaler(b.X, mu, sd)
            prior_tr = float(b.y_bin[tr].mean())

            for model in models:
                for seed in seeds:
                    t1 = time.time()
                    try:
                        if model == "rf":
                            p, npar, eps = RP.train_rf(Xn[tr], b.mask[tr], b.y_bin[tr],
                                                       Xn[ev], b.mask[ev], seed)
                        elif model == "hgb":
                            p, npar, eps = RP.train_hgb(Xn[tr], b.mask[tr], b.y_bin[tr],
                                                        Xn[ev], b.mask[ev], seed)
                        else:
                            p, npar, eps = RP.train_seq(
                                model, Xn[tr], b.mask[tr], b.y_bin[tr],
                                Xn[va], b.mask[va], b.y_bin[va],
                                Xn[ev], b.mask[ev], y_ev, seed, DEV)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        continue

                    pva, pt, pn = p[:n_va], p[n_va:n_va + n_te], p[n_va + n_te:]
                    yt = b.y_bin[te]

                    f1 = f1_score(yt, (pt >= 0.5).astype(int), average="macro", zero_division=0)
                    auc = float(roc_auc_score(yt, pt)) if len(np.unique(yt)) > 1 else float("nan")
                    neg = yt == 0
                    fpr_te = float((pt[neg] >= 0.5).mean()) if neg.any() else float("nan")


                    thr_q = float(np.clip(1.0 - prior_tr, 0.001, 0.999))
                    thr_q = float(np.quantile(pt, thr_q))
                    thr_o = otsu_thr(pt)


                    yva = b.y_bin[va]
                    cand = np.unique(np.concatenate(
                        [np.quantile(pva, np.linspace(0.5, 0.9995, 60)), [0.5]]))
                    thr_v = max(cand, key=lambda t: f1_score(
                        yva, (pva >= t).astype(int), average="macro",
                        zero_division=0))
                    fpr_n = {k: float((pn >= t).mean()) for k, t in
                             [("@0.5", 0.5), ("@qtrain", thr_q), ("@otsu", thr_o), ("@val", thr_v)]}

                    rec = dict(cell=cell, model=model, seed=seed, te_day=int(te_day),
                               cudnn_benchmark=bool(RP.CUDNN_BENCH),
                               analysis_plan_commit=ANALYSIS_PLAN_COMMIT,
                               ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               prior_tr=prior_tr,
                               n_norm=n_nd, n_test=n_te,
                               f1_te=float(f1), auc_te=auc, fpr_te=fpr_te,
                               excess_te=float(f1 - floor_of(float(yt.mean()))),
                               thr=dict(qtrain=thr_q, otsu=thr_o, val=thr_v),
                               **{f"fpr_norm{k}": v for k, v in fpr_n.items()},
                               epochs=eps, sec=round(time.time() - t1, 1))
                    runs.append(rec)
                    print(f"    {model:<12} s{seed}  F1={f1:.4f} AUC={auc:.4f} "
                          f"FPR_te={fpr_te:.4f} | attack-free-day FPR @0.5={fpr_n['@0.5']:.4f} "
                          f"@qtrain={fpr_n['@qtrain']:.4f} @otsu={fpr_n['@otsu']:.4f}")
        del b
        gc.collect()

    if args.probe:
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(runs=runs, norm_day=int(norm_day),
                   args=dict(models=models, seeds=seeds, T=args.T,
                             max_train=args.max_train,
                             cudnn_benchmark=bool(RP.CUDNN_BENCH),
                             analysis_plan_commit=ANALYSIS_PLAN_COMMIT,
                             script="scripts_v2/normal_day_fpr_v2.py",
                             upstream="scripts/normal_day_fpr.py")),
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nWrote {len(runs)} runs to {args.out}  elapsed {(time.time()-t_all)/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
