#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _c in (os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "src", "data"),
           os.path.join(_ROOT, "src", "models")):
    if os.path.isdir(_c):
        sys.path.insert(0, _c)

import wp2e_dump as W
import seqdiff
import run_pilot as RP
import seqmodels_v2 as SM

PLAN_COMMIT = "c1ced61"
E1_DIR = os.path.join(_ROOT, "results_v2", "e1")
SCORE_DIR = os.path.join(_ROOT, "scores_v2", "e1")


ARMS = {
    "inf_U":     dict(days=[0, 1, 2, 3, 7, 8], train=[0, 1, 2, 3], test=8, group="Inf"),
    "inf_S":     dict(days=[0, 1, 2, 3, 7, 8], train=[0, 1, 2, 7], test=8, group="Inf"),
    "inf_P":     dict(days=[0, 1, 2, 3, 7, 8], train=[0, 1, 2, 7], test=8, group="Inf",
                      prior_match="inf_U"),
    "selfcheck": dict(days=[0, 1, 2, 3, 4], train=[0, 1, 2], test=4, group="selfcheck",
                      legacy=True),
}

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY, ts TEXT, dataset TEXT, split TEXT, fold INTEGER,
    cell TEXT, route TEXT, padding TEXT, model TEXT, seed INTEGER,
    class_weight TEXT, cudnn_benchmark INTEGER, "commit" TEXT, analysis_plan_commit TEXT,
    n_seq INTEGER, n_train INTEGER, n_test INTEGER, test_pos_rate REAL,
    macro_f1 REAL, f1_pos REAL, fpr REAL, pr_auc REAL, mcc REAL, roc_auc REAL,
    params INTEGER, epochs INTEGER, elapsed_s REAL,
    pad_fraction REAL, label_mismatch REAL, filtered_out_rate REAL,
    pos_q10 REAL, pos_q50 REAL, pos_q90 REAL,
    neg_q10 REAL, neg_q50 REAL, neg_q90 REAL,
    val_pos_q50 REAL, val_neg_q50 REAL,
    frac_pos_above_05 REAL, pred_true_ratio REAL,
    thr_oracle REAL, macro_f1_oracle REAL,
    -- E1 additions --
    "group" TEXT, arm TEXT, train_days TEXT, test_day INTEGER,
    train_prior REAL, test_prior REAL,
    n_train_seq INTEGER, n_test_seq INTEGER, n_pos_test_seq INTEGER,
    tpr_at_fpr_1pct REAL, tpr_at_fpr_0p1pct REAL,
    prior_match_side TEXT, prior_match_note TEXT,
    prior_match_keep INTEGER, score_file TEXT
)"""

FAMILY_COLS = ["run_id", "group", "arm", "family", "seen_in_train", "n_pos_seq",
               "family_auc", "family_score_q50", "recall_at_05"]


def tpr_at_fpr(y, p, target: float) -> float:
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    P = int(y.sum())
    N = len(y) - P
    if P <= 0 or N <= 0:
        return float("nan")
    o = np.argsort(-p, kind="mergesort")
    ys = y[o]
    tp = np.cumsum(ys)
    fp = np.arange(1, len(y) + 1) - tp
    ok = np.where(fp / N <= target)[0]
    return float((tp[ok] / P).max()) if len(ok) else 0.0


def base_split(b, train_days, test_day, val_frac=0.2):

    d = np.asarray(b.day)
    tr_all = np.where(np.isin(d, np.asarray(train_days)))[0]
    order = tr_all[np.argsort(np.asarray(b.wid)[tr_all], kind="mergesort")]
    n_va = max(1, int(len(order) * val_frac))
    return dict(train=order[:-n_va], val=order[-n_va:],
                test=np.where(d == int(test_day))[0])


def downsample_extra_day(b, tr, extra_day, target_prior, rng):

    d = np.asarray(b.day)
    y = np.asarray(b.y_bin)
    is_extra_atk = (d[tr] == int(extra_day)) & (y[tr] == 1)
    rest = tr[~is_extra_atk]
    extra = tr[is_extra_atk]
    n_rest, pos_rest = len(rest), int(y[rest].sum())
    if not len(extra):
        return np.sort(rest), 0, "no_extra_attack"
    if target_prior >= 1.0:
        return np.sort(tr), len(extra), "target_ge_1"
    k = (target_prior * n_rest - pos_rest) / (1.0 - target_prior)
    k = int(round(k))
    note = ""
    if k < 0:
        k, note = 0, "target_below_rest"
    if k > len(extra):
        k, note = len(extra), "capped"
    sel = rng.choice(extra, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)
    return np.sort(np.concatenate([rest, sel])), k, note


def family_rows(rid, group, arm, fam_te, y_te, p, seen_names):


    out = []
    benign = y_te == 0
    for c in sorted(set(fam_te.tolist())):
        if c == "BENIGN":
            continue
        m = fam_te == c
        if m.sum() == 0:
            continue
        yy = np.r_[np.ones(int(m.sum())), np.zeros(int(benign.sum()))]
        pp = np.r_[p[m], p[benign]]
        try:
            auc = float(roc_auc_score(yy, pp))
        except Exception:
            auc = float("nan")
        out.append(dict(run_id=rid, group=group, arm=arm, family=str(c),
                        seen_in_train=bool(c in seen_names),
                        n_pos_seq=int(m.sum()), family_auc=auc,
                        family_score_q50=float(np.median(p[m])),
                        recall_at_05=float((p[m] >= 0.5).mean())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="inf_U,inf_S,inf_P")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--models", default="lstm,tcn,transformer,rf,hgb")
    ap.add_argument("--cells", default="", help="Run only cells matching the tag substring; default: all nine cells")
    ap.add_argument("--max-segments", type=int, default=200000)
    ap.add_argument("--db", default=os.path.join(E1_DIR, "e1_runs.db"))
    ap.add_argument("--family-csv", default=os.path.join(E1_DIR, "e1_family.csv"))
    ap.add_argument("--verify-db", default="")
    ap.add_argument("--verify-mode", default="strict", choices=["strict", "record"])
    ap.add_argument("--dry-run", action="store_true", help="Build splits and print sizes without training")
    args = ap.parse_args()

    RP.CUDNN_BENCH = False
    import torch
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {dev}  cudnn.benchmark={RP.CUDNN_BENCH}")

    try:
        import subprocess
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=_ROOT) != 0
        code_commit = head + ("+dirty" if dirty else "")
    except Exception:
        code_commit = "unknown"

    os.makedirs(E1_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)
    con = sqlite3.connect(args.db)
    con.execute(DB_SCHEMA)
    con.commit()
    fam_head = not os.path.exists(args.family_csv)

    old = {}
    if args.verify_db:
        ocon = sqlite3.connect(args.verify_db)
        q = "select run_id,%s from runs" % ",".join(W.MATCH_COLS)
        old = {r[0]: r[1:] for r in ocon.execute(q)}
        ocon.close()
        print(f"Consistency gate: loaded {len(old)} rows from old DB {args.verify_db}")

    models = tuple(args.models.split(","))
    from run_pilot import CELLS as RP_CELLS

    for arm_name in args.arms.split(","):
        arm_name = arm_name.strip()
        spec = ARMS[arm_name]
        print(f"\n===== Arm {arm_name} ({spec['group']}) days {spec['days']} "
              f"train {spec['train']} test d{spec['test']} =====")
        df = W.add_group_ids(W.load_parquet_dir(
            os.path.join(_ROOT, "data", "processed", "cic18"), days=spec["days"]))
        code, fam_names = W.label_codes(df, "cic18")
        print(f"  Flows {len(df):,}  attack families {len(fam_names)-1}")
        y_flow = np.asarray(code)
        day_flow = np.asarray(df["_day"])

        for route, kw in RP_CELLS:
            for T in (20,):
                cfg = seqdiff.SeqConfig(segment=route, T=T, seed=0,
                                        max_segments=args.max_segments, **kw)
                cell = cfg.tag()
                if args.cells and args.cells not in cell:
                    continue
                t0 = time.time()
                b = seqdiff.build_sequences(df, cfg)
                si, _ = W.strat_idx_of(b.y_bin, 0, seed=0)
                lengths, fam_keep, fam_full, yb = W.cell_meta(df, cfg, code, fam_names, si)
                assert np.array_equal(yb, b.y_bin), "Recomputed family labels do not match y_bin"
                y = np.asarray(b.y_bin)
                d = np.asarray(b.day)


                side = ""
                if spec.get("legacy"):
                    from seqdiff import make_split
                    sp = make_split(b, "cross_day", seed=0)[0]
                    train, val, test = sp["train"], sp["val"], sp["test"]
                    rid_split = "cross_day"
                else:
                    base = base_split(b, spec["train"], spec["test"])
                    train, val, test = base["train"], base["val"], base["test"]
                    rid_split = arm_name
                    if spec.get("prior_match"):
                        u = base_split(b, ARMS[spec["prior_match"]]["train"], spec["test"])
                        pU = float(y[u["train"]].mean())
                        pS = float(y[train].mean())
                        if pS > pU:
                            side, target, extra = "S", pU, 7
                            btrain, bval = train, val
                        else:
                            side, target, extra = "U", pS, 3
                            btrain, bval = u["train"], u["val"]
                        print(f"  [inf_P] pU={pU:.4f} pS={pS:.4f} → downsampling {side} side d{extra} "
                              f"to {target:.4f}")
                        train = btrain
                        val = bval


                fams_tr_base = set(np.unique(fam_keep[train]).tolist()) - {"BENIGN"}
                print(f"  [{cell}] Sequences {len(b.wid):,} train {len(train):,} "
                      f"validation {len(val):,} test {len(test):,}  "
                      f"train prior {y[train].mean():.4f} test prior {y[test].mean():.4f}  "
                      f"test positives {int(y[test].sum()):,}  train families {len(fams_tr_base)}  "
                      f"({time.time()-t0:.0f}s)")

                if args.dry_run:
                    continue
                if int(y[test].sum()) < 5:
                    print("    [skip] test positives < 5")
                    continue

                mu, sd = RP.fit_scaler(b.X[train], b.mask[train])
                Xn = RP.apply_scaler(b.X, mu, sd)

                for model in models:
                    for seed in args.seeds:
                        if spec.get("prior_match"):
                            rng = np.random.default_rng(seed)
                            tr_s, k_used, note = downsample_extra_day(
                                b, train, 3 if side == "U" else 7, target, rng)
                        else:
                            tr_s, k_used, note = train, -1, ""
                        if len(np.unique(y[tr_s])) < 2:
                            print(f"    [skip] {model} s{seed}: training set has only one class")
                            continue
                        t1 = time.time()
                        try:
                            if model == "rf":
                                p, pv, npar, eps = SM.train_rf_v2(
                                    Xn[tr_s], b.mask[tr_s], y[tr_s],
                                    Xn[val], b.mask[val], Xn[test], b.mask[test], seed,
                                    class_weight="balanced_subsample")
                            elif model == "hgb":
                                p, pv, npar, eps = SM.train_hgb_v2(
                                    Xn[tr_s], b.mask[tr_s], y[tr_s],
                                    Xn[val], b.mask[val], Xn[test], b.mask[test], seed,
                                    class_weight="balanced")
                            else:


                                p, pv, npar, eps = SM.train_seq_v2(
                                    model, Xn[tr_s], b.mask[tr_s], y[tr_s],
                                    Xn[val], b.mask[val], y[val],
                                    Xn[test], b.mask[test], y[test], seed, dev,
                                    max_epochs=20, batch=512, patience_max=5,
                                    lr=1e-3, warmup_frac=0.0,
                                    class_weight="balanced")
                            m = RP.metrics_of(y[test], p)
                        except Exception as e:
                            print(f"    [failed] {model} s{seed}: {type(e).__name__}: {e}")
                            continue
                        el = time.time() - t1
                        rid = (f"c18e1|{cell}|{rid_split}|{model}|s{seed}"
                               if arm_name != "selfcheck"
                               else f"c18det|{cell}|cross_day|{model}|s{seed}")

                        if old:
                            if rid not in old:
                                print(f"[stop] {rid} is absent from old DB")
                                return 2
                            ov = dict(zip(W.MATCH_COLS, old[rid]))
                            nv = {c: m[c] for c in W.MATCH_COLS if c != "epochs"}
                            nv["epochs"] = eps
                            bad = [(c, ov[c], nv[c]) for c in W.EXACT_COLS if ov[c] != nv[c]]
                            for c in W.TOL_COLS:
                                if abs(ov[c] - nv[c]) > W.TOL:
                                    bad.append((c, ov[c], nv[c]))
                            if bad:
                                print("[stop] self-check mismatch: " + "; ".join(
                                    "%s old=%r new=%r" % x for x in bad))
                                if args.verify_mode == "strict":
                                    return 2
                            else:
                                print(f"    ✓ bit-identical to primary DB (epochs={eps})")

                        safe = W.safe_name(rid)
                        sfile = os.path.join(SCORE_DIR, safe + ".npz")
                        np.savez_compressed(
                            sfile,
                            score=np.asarray(p, dtype=np.float64),
                            label=np.asarray(y[test], dtype=np.int8),
                            family=np.asarray(fam_keep[test], dtype=str),
                            family_full=np.asarray(fam_full[test], dtype=str),
                            segment_key=np.asarray(b.group[test], dtype=np.int64),
                            t_start=np.asarray(b.wid[test], dtype=np.int64),
                            run_id=np.asarray([rid]))

                        sp_ = np.asarray(p)
                        yt = y[test]
                        pq = W.quant3(sp_[yt == 1])
                        nq = W.quant3(sp_[yt == 0])
                        vq = W.quant3(np.asarray(pv)[y[val] == 1])
                        vn = W.quant3(np.asarray(pv)[y[val] == 0])
                        orc_f1, orc_thr = W.thr_oracle_sweep(yt, sp_)
                        rec = dict(
                            run_id=rid, ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            dataset="CIC18-6d" if not spec.get("legacy") else "CIC18-5d",
                            split=rid_split, fold=-1, cell=cell, route=route,
                            padding=kw["padding"], model=model, seed=seed,
                            class_weight="balanced", cudnn_benchmark=int(RP.CUDNN_BENCH),
                            commit=code_commit, analysis_plan_commit=PLAN_COMMIT,
                            n_seq=len(b.wid), n_train=len(tr_s), n_test=len(test),
                            test_pos_rate=float(yt.mean()),
                            params=int(npar), epochs=int(eps), elapsed_s=round(el, 2),
                            pad_fraction=float(b.meta["pad_fraction"]),
                            label_mismatch=float(b.meta["label_input_mismatch"]),
                            filtered_out_rate=float(b.meta["filtered_out_rate"]),
                            pos_q10=pq[0], pos_q50=pq[1], pos_q90=pq[2],
                            neg_q10=nq[0], neg_q50=nq[1], neg_q90=nq[2],
                            val_pos_q50=vq[1], val_neg_q50=vn[1],
                            frac_pos_above_05=float((sp_[yt == 1] >= 0.5).mean()),
                            pred_true_ratio=float((sp_ >= 0.5).sum() / max(int(yt.sum()), 1)),
                            thr_oracle=orc_thr, macro_f1_oracle=orc_f1,
                            group=spec["group"], arm=arm_name,
                            train_days="+".join(str(x) for x in spec["train"]),
                            test_day=int(spec["test"]),
                            train_prior=float(y[tr_s].mean()),
                            test_prior=float(yt.mean()),
                            n_train_seq=int(len(tr_s)), n_test_seq=int(len(test)),
                            n_pos_test_seq=int(yt.sum()),
                            tpr_at_fpr_1pct=tpr_at_fpr(yt, sp_, 0.01),
                            tpr_at_fpr_0p1pct=tpr_at_fpr(yt, sp_, 0.001),
                            prior_match_side=(side if spec.get("prior_match") else ""),
                            prior_match_note=note,
                            prior_match_keep=(int(k_used) if spec.get("prior_match") else -1),
                            score_file=sfile, **m)
                        con.execute(
                            "INSERT OR REPLACE INTO runs(%s) VALUES(%s)"
                            % (",".join('"%s"' % k for k in rec), ",".join("?" * len(rec))),
                            tuple(rec.values()))
                        con.commit()


                        seen_s = set(np.unique(fam_keep[tr_s]).tolist()) - {"BENIGN"}
                        rows = family_rows(rid, spec["group"], arm_name, fam_keep[test],
                                           yt, sp_, seen_s)
                        with open(args.family_csv, "a", newline="", encoding="utf-8") as f:
                            w = csv.DictWriter(f, fieldnames=FAMILY_COLS)
                            if fam_head:
                                w.writeheader()
                                fam_head = False
                            w.writerows(rows)

                        print(f"    {model:<12} s{seed:<2} F1 {m['macro_f1']:.4f} "
                              f"AUC {m['roc_auc']:.4f} pos_q50 {pq[1]:.4f} "
                              f"thr* {orc_thr:.3f} ({el:.1f}s"
                              + (f", keep {k_used} {note}" if note else "") + ")")

    con.close()
    print("\nDone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
