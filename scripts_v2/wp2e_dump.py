


from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
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
import run_pilot as RP
import seqmodels_v2 as SM
import splits_v2 as SP
from seqdiff import (
    IDENTITY_COLS, LABEL_COLS, SeqConfig, add_group_ids, build_sequences,
    load_days, load_parquet_dir, make_split,
)

PLAN_COMMIT = "e0dc336"


EXACT_COLS = ["macro_f1", "f1_pos", "fpr", "mcc", "epochs"]


TOL_COLS = ["pr_auc", "roc_auc"]
TOL = 1e-4
MATCH_COLS = EXACT_COLS + TOL_COLS
TOL_MAX = {c: 0.0 for c in TOL_COLS}


_SEGS: dict = {}
_ORIG_SEGMENT_ROWS = seqdiff.segment_rows


def _recording_segment_rows(df, cfg):
    segs = _ORIG_SEGMENT_ROWS(df, cfg)
    _SEGS["segs"] = segs
    return segs


seqdiff.segment_rows = _recording_segment_rows


def safe_name(rid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", rid)


def quant3(x):
    if len(x) == 0:
        return (float("nan"),) * 3
    v = np.quantile(x, [0.10, 0.50, 0.90])
    return float(v[0]), float(v[1]), float(v[2])


def thr_oracle_sweep(y, p):


    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n = len(y)
    P = int(y.sum())
    N = n - P
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, n + 1)
    fp = k - tp
    fn = P - tp
    tn = N - fp
    f1p = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1n = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    macro = (f1p + f1n) / 2.0

    f1n0 = 2 * N / max(2 * N + P, 1)
    best_f1, best_thr = f1n0 / 2.0, float("inf")

    ends = np.where(np.r_[ps[1:] != ps[:-1], True])[0]
    for j in ends:
        if macro[j] > best_f1:
            best_f1, best_thr = float(macro[j]), float(ps[j])
    return best_f1, best_thr


def label_codes(df, dataset):


    if dataset == "unsw" and "_Attack" in df.columns:
        raw = df["_Attack"].astype(str).str.strip().str.upper().to_numpy()
        atk = df["Label"].astype(np.float64).to_numpy() != 0.0
        raw = np.where(atk, raw, "BENIGN")
    else:
        lab = df["Label"]
        if pd.api.types.is_numeric_dtype(lab):
            atk = pd.to_numeric(lab, errors="coerce").fillna(0).to_numpy() != 0
            raw = np.where(atk, "ATTACK", "BENIGN")
        else:
            raw = lab.astype(str).str.upper().to_numpy()
            atk = raw != "BENIGN"
    raw = np.where(atk, raw, "BENIGN")
    names = sorted({s for s in raw if s != "BENIGN"})
    lut = {s: i + 1 for i, s in enumerate(names)}
    code = np.zeros(len(raw), dtype=np.int16)
    for s, c in lut.items():
        code[raw == s] = c
    return code, ["BENIGN"] + names


def _mode_family(codes_sub):

    if len(codes_sub) == 0:
        return 0
    vals, cnts = np.unique(codes_sub, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def cell_meta(df, cfg, code, fam_names, strat_idx):


    segs = _SEGS["segs"]
    if cfg.segment == "time_window" and cfg.truncate == "random":
        raise SystemExit("truncate=random requires shared RNG state with build_sequences, which this script does not implement; "
                         "the nine cells in this database do not use that setting")
    n_all = len(segs)
    rng = np.random.default_rng(cfg.seed)
    if cfg.max_segments and n_all > cfg.max_segments:
        keep = np.sort(rng.choice(n_all, size=cfg.max_segments, replace=False))
        segs = [segs[i] for i in keep]

    T = cfg.T
    lengths = np.empty(len(segs), dtype=np.int64)
    fam_keep = np.zeros(len(segs), dtype=np.int16)
    fam_full = np.zeros(len(segs), dtype=np.int16)
    yb = np.zeros(len(segs), dtype=np.int8)
    for s, rows in enumerate(segs):
        n_full = len(rows)
        lengths[s] = n_full
        if n_full > T:
            keep_rows = rows[-T:] if cfg.truncate == "tail" else rows[:T]
        else:
            keep_rows = rows
        ck = code[keep_rows]
        yb[s] = int((ck > 0).any())
        fam_keep[s] = _mode_family(ck[ck > 0])
        cf = code[rows]
        fam_full[s] = _mode_family(cf[cf > 0])

    if cfg.padding == "non_padded":
        m = lengths >= T
        lengths, fam_keep, fam_full, yb = (lengths[m], fam_keep[m], fam_full[m], yb[m])


    strat_idx = np.asarray(strat_idx)
    if len(strat_idx) and int(strat_idx.max()) >= len(lengths):
        raise SystemExit(
            "Index out of bounds: strat_idx.max()=%d ≥ meta length %d — segment sampling recomputation differs from actual data; stopping"
            % (int(strat_idx.max()), len(lengths)))
    lengths = lengths[strat_idx]
    fam_keep = fam_keep[strat_idx]
    fam_full = fam_full[strat_idx]
    yb = yb[strat_idx]
    fam_keep_s = np.array([fam_names[c] for c in fam_keep], dtype=object).astype(str)
    fam_full_s = np.array([fam_names[c] for c in fam_full], dtype=object).astype(str)
    return lengths, fam_keep_s, fam_full_s, yb


def strat_idx_of(y, cap, seed=0):

    n = len(y)
    if cap <= 0 or n <= cap:
        return np.arange(n), 1.0
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    k_pos = min(len(pos), int(round(cap * y.mean())))
    k_neg = min(len(neg), cap - k_pos)
    idx = np.sort(np.concatenate([
        rng.choice(pos, k_pos, replace=False) if len(pos) > k_pos else pos,
        rng.choice(neg, k_neg, replace=False) if len(neg) > k_neg else neg]))
    return idx, len(idx) / n


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
    score_file TEXT
)"""


def load_dataset(args):
    if args.dataset == "unsw":
        df = add_group_ids(load_parquet_dir(
            f"{args.root}/protocol_robust_nids_20261028/data/processed/unsw",
            ts_guard=("2015-01-01", "2015-03-01"), days=args.unsw_days))
        ds_tag = "UNSW-" + str(len(args.unsw_days)) + "d"
        rp = args.rid_prefix or "unsw"
    elif args.dataset == "cic18":
        df = add_group_ids(load_parquet_dir(
            f"{args.root}/protocol_robust_nids_20261028/data/processed/cic18",
            days=args.cic18_days))
        ds_tag = "CIC18-" + str(len(args.cic18_days)) + "d"
        rp = args.rid_prefix or "cic18"
    else:
        df = add_group_ids(load_days(args.root, "engelen", args.days))
        ds_tag = "CIC17-" + "+".join(d[:3] for d in args.days)
        rp = args.rid_prefix or "+".join(d[:3] for d in args.days)

    if args.feature_subset:
        keep = set(json.load(open(args.feature_subset, encoding="utf-8")))
        keep |= set(IDENTITY_COLS) | set(LABEL_COLS)
        drop = [c for c in df.columns if c not in keep and not c.startswith("_")]
        df = df.drop(columns=drop)
        print(f"  Feature subset pruning: retained {df.shape[1]} columns, dropped {len(drop)} columns")

    print(f"Data {ds_tag}: {len(df):,} flows, days {df['_day'].nunique() if '_day' in df else '?'}")
    return df, ds_tag, rp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--dataset", default="cic17", choices=["cic17", "cic18", "unsw"])
    ap.add_argument("--days", nargs="+",
                    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    ap.add_argument("--cic18-days", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--unsw-days", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", default="lstm,tcn,transformer,rf,hgb")
    ap.add_argument("--t-values", type=int, nargs="+", default=[20])
    ap.add_argument("--split", default="cross_day",
                    choices=["cross_day", "random", "temporal", "cross_day_pair",
                             "block_strat", "cross_day_ext", "cross_day_nd"])
    ap.add_argument("--block-minutes", type=int, default=10,
                    help="Time-block width in minutes for WP2c-S1")
    ap.add_argument("--zero-days", type=int, nargs="+", default=None,
                    help="Zero-attack days (inference only, no training). Use with --split cross_day_nd; "
                         "scores are stored in score_zero/label_zero in the npz file")
    ap.add_argument("--dump-val", action="store_true",
                    help="Also save validation-fold scores (required for WP2d BBSE and calibration)")
    ap.add_argument("--prior-csv", default="",
                    help="WP2c: append each cell's training/validation positive rates to this CSV (required for S3 downsampling)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cell-filter", default="")
    ap.add_argument("--max-seq", type=int, default=0)
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--feature-subset", default="")
    ap.add_argument("--rid-prefix", default="")
    ap.add_argument("--db", required=True)
    ap.add_argument("--scores-dir", default="scores_v2")
    ap.add_argument("--verify-db", default="",
                    help="Path to old database; compare seven fields immediately after each run")
    ap.add_argument("--verify-mode", default="strict", choices=["strict", "record"],
                    help="strict=exit(2) immediately on tolerance violations (specification default); "
                         "record=only log violations to wp2e_gate_log.csv and continue, for measuring deviation magnitude")
    ap.add_argument("--class-weight", default="balanced",
                    help="For records: balanced=main-database setting; none=WP2a unweighted setting")
    ap.add_argument("--hgb-class-weight", default="balanced", choices=["balanced", "none"])
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.0)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    RP.CUDNN_BENCH = False
    import torch
    torch.backends.cudnn.benchmark = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {dev}  cudnn.benchmark={RP.CUDNN_BENCH} (deterministic kernels)")


    try:
        import subprocess
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=_ROOT) != 0
        code_commit = head + ("+dirty" if dirty else "")
    except Exception:
        code_commit = "unknown"

    df, ds_tag, rid_prefix = load_dataset(args)
    code, fam_names = label_codes(df, args.dataset)
    print(f"Attack families: {len(fam_names)-1}: {fam_names[1:]}")

    cells = [(r, dict(kw)) for r, kw in RP.CELLS]
    if args.cell_filter:
        cells = [c for c in cells if args.cell_filter in f"{c[0]}/{c[1]['padding']}"]
        if not cells:
            raise SystemExit("cell-filter matched nothing")
    models = tuple(args.models.split(","))
    cells = [(r, dict(kw), T) for T in args.t_values for r, kw in cells]

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    con = sqlite3.connect(args.db)
    con.execute(DB_SCHEMA)
    con.commit()

    old = {}
    if args.verify_db:
        ocon = sqlite3.connect(args.verify_db)
        q = "select run_id,%s from runs" % ",".join(MATCH_COLS)
        old = {r[0]: r[1:] for r in ocon.execute(q)}
        ocon.close()
        print(f"Consistency gate: loaded {len(old)} rows from old database {args.verify_db}")

    existing = set()
    if args.resume:
        existing = {r[0] for r in con.execute("select run_id from runs")}

    t_all = time.time()
    n_ok = n_bad = n_skip = 0
    for route, kw, T in cells:
        cfg = SeqConfig(segment=route, T=T, seed=0,
                        max_segments=args.max_segments, **kw)
        b = build_sequences(df, cfg)
        si, _ = strat_idx_of(b.y_bin, args.max_seq, seed=0)
        if len(si) != len(b.wid):
            b = seqdiff.SeqBatch(
                X=b.X[si], mask=b.mask[si], length=b.length[si],
                y_bin=b.y_bin[si], y_multi=b.y_multi[si], group=b.group[si],
                wid=b.wid[si], day=(None if b.day is None else b.day[si]),
                meta=b.meta)
        if len(b.wid) != len(si):
            raise SystemExit("stratified_cap alignment failed")
        lengths, fam_keep, fam_full, yb_meta = cell_meta(df, cfg, code, fam_names, si)
        if not np.array_equal(yb_meta, b.y_bin):
            bad = int((yb_meta != b.y_bin).sum())
            raise SystemExit(
                "Recomputed family-label y_bin differs from build_sequences at %d positions — "
                "recomputed keep set differs from actual data; stopping" % bad)
        if not np.array_equal(lengths, b.length):
            raise SystemExit(
                "Recomputed sequence lengths differ from build_sequences (%d/%d positions) — stopping"
                % (int((lengths != b.length).sum()), len(lengths)))
        cell = cfg.tag()
        print(f"\n[{cell}] sequences {len(b.wid):,}  positive rate {b.meta['pos_rate']:.4f}  "
              f"families {len(set(fam_keep))}")

        if args.split in ("block_strat", "cross_day_ext", "cross_day_nd"):
            folds = SP.make_split_v2(b, args.split,
                                     block_minutes=args.block_minutes, seed=0)
        else:
            folds = make_split(b, args.split, seed=0)
        if args.fold >= len(folds):
            raise SystemExit(f"--fold {args.fold} is out of range ({len(folds)} folds total)")
        fold = folds[args.fold]
        fold_tag = f"|f{args.fold}" if len(folds) > 1 else ""
        fold_id = args.fold if len(folds) > 1 else -1
        tr, va, te = fold["train"], fold["val"], fold["test"]
        if args.prior_csv:
            s0p = SP.s0_train_pos_rate(b) if b.day is not None else float("nan")
            _log_prior(args.prior_csv, cell, args.split, b, tr, va, te, s0p)
        if args.max_train:
            tr = RP.cap_train_indices(b.y_bin, tr, args.max_train, seed=0)
        pos_tr = int(b.y_bin[tr].sum())
        pos_te = int(b.y_bin[te].sum())
        neg_te = int(len(te) - pos_te)
        if pos_tr < 5 or pos_te < 5 or neg_te < 5:
            print(f"  [skip] insufficient classes train_pos={pos_tr} test_pos={pos_te} test_neg={neg_te}")
            continue

        mu, sd = RP.fit_scaler(b.X[tr], b.mask[tr])
        Xn = RP.apply_scaler(b.X, mu, sd)


        zi = fold.get("zero") if args.zero_days is not None else None
        if args.zero_days is not None and zi is None:
            raise SystemExit("--zero-days requires a split with a zero key (use --split cross_day_nd)")
        Xz = Xn[zi] if zi is not None else None
        Mz = b.mask[zi] if zi is not None else None

        for model in models:
            for seed in args.seeds:
                rid = f"{rid_prefix}|{cell}|{args.split}{fold_tag}|{model}|s{seed}"
                if args.resume and rid in existing:
                    n_skip += 1
                    continue
                t1 = time.time()
                zout = {}
                cw = ("none" if args.class_weight == "none" else "balanced")
                try:
                    if model == "rf":
                        _rf_cw = (None if args.class_weight == "none"
                                  else "balanced_subsample")
                        p, pv, npar, eps = SM.train_rf_v2(
                            Xn[tr], b.mask[tr], b.y_bin[tr],
                            Xn[va], b.mask[va], Xn[te], b.mask[te], seed,
                            class_weight=_rf_cw, Xz=Xz, Mz=Mz, zout=zout)
                    elif model == "hgb":
                        _cw = None if (args.class_weight == "none"
                                       or args.hgb_class_weight == "none") else "balanced"
                        p, pv, npar, eps = SM.train_hgb_v2(
                            Xn[tr], b.mask[tr], b.y_bin[tr],
                            Xn[va], b.mask[va], Xn[te], b.mask[te], seed,
                            class_weight=_cw, Xz=Xz, Mz=Mz, zout=zout)
                    else:
                        p, pv, npar, eps = SM.train_seq_v2(
                            model, Xn[tr], b.mask[tr], b.y_bin[tr],
                            Xn[va], b.mask[va], b.y_bin[va],
                            Xn[te], b.mask[te], b.y_bin[te], seed, dev,
                            max_epochs=args.max_epochs, batch=args.batch,
                            patience_max=args.patience, lr=args.lr,
                            warmup_frac=args.warmup_frac,
                            class_weight=args.class_weight,
                            Xz=Xz, Mz=Mz, zout=zout)
                    m = RP.metrics_of(b.y_bin[te], p)
                except Exception as e:
                    print(f"  [failed] {model} s{seed}: {type(e).__name__}: {e}")
                    continue
                el = time.time() - t1


                if old:
                    if rid not in old:
                        msg = f"{rid} is absent from the old database and cannot be verified"
                        _raise_verification_error(msg, args)
                        print("[stop]", msg)
                        return 2
                    ov = dict(zip(MATCH_COLS, old[rid]))
                    nv = {c: m[c] for c in MATCH_COLS if c != "epochs"}
                    nv["epochs"] = eps
                    bad = [(c, ov[c], nv[c]) for c in EXACT_COLS if ov[c] != nv[c]]
                    for c in TOL_COLS:
                        d = abs(ov[c] - nv[c])
                        TOL_MAX[c] = max(TOL_MAX[c], d)
                        if d > TOL:
                            bad.append((c, ov[c], nv[c]))
                    if bad:
                        msg = ("Rerun differs from old database: %s → %s" % (
                            rid, "; ".join("%s old=%r new=%r" % x for x in bad)))
                        if args.verify_mode == "strict":
                            _raise_verification_error(msg, args)
                            print("[stop]", msg)
                            return 2
                        _log_gate(bad, rid)
                        n_bad += 1
                    else:
                        n_ok += 1


                sdir = os.path.join(args.scores_dir, args.dataset)
                os.makedirs(sdir, exist_ok=True)
                sfile = os.path.join(sdir, safe_name(rid) + ".npz")


                np.savez_compressed(
                    sfile,
                    score=np.asarray(p, dtype=np.float64),
                    label=np.asarray(b.y_bin[te], dtype=np.int8),
                    family=np.asarray(fam_keep[te], dtype=str),
                    family_full=np.asarray(fam_full[te], dtype=str),
                    day=np.asarray(b.day[te] if b.day is not None else np.full(len(te), -1),
                                   dtype=np.int32),
                    segment_key=np.asarray(b.group[te], dtype=np.int64),
                    t_start=np.asarray(b.wid[te], dtype=np.int64),
                    length=np.asarray(lengths[te], dtype=np.int32),
                    run_id=np.asarray([rid]),
                    **({"score_val": np.asarray(pv, dtype=np.float64),
                        "label_val": np.asarray(b.y_bin[va], dtype=np.int8)}
                       if args.dump_val else {}),
                    **({"score_zero": np.asarray(zout["p"], dtype=np.float64),
                        "label_zero": np.asarray(b.y_bin[zi], dtype=np.int8),
                        "t_start_zero": np.asarray(b.wid[zi], dtype=np.int64),
                        "day_zero": np.asarray(b.day[zi], dtype=np.int32)}
                       if (Xz is not None and "p" in zout) else {}),
                )

                sp = np.asarray(p)
                yt = np.asarray(b.y_bin[te])
                pq = quant3(sp[yt == 1])
                nq = quant3(sp[yt == 0])
                yv = np.asarray(b.y_bin[va])
                vq = quant3(np.asarray(pv)[yv == 1])
                vn = quant3(np.asarray(pv)[yv == 0])
                pred_pos = int((sp >= 0.5).sum())
                orc_f1, orc_thr = thr_oracle_sweep(yt, sp)

                rec = dict(
                    run_id=rid, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), dataset=ds_tag,
                    split=args.split, fold=fold_id, cell=cell, route=route,
                    padding=kw["padding"], model=model, seed=seed,
                    class_weight=cw, cudnn_benchmark=int(RP.CUDNN_BENCH),
                    commit=code_commit, analysis_plan_commit=PLAN_COMMIT,
                    n_seq=len(b.wid), n_train=len(tr), n_test=len(te),
                    test_pos_rate=float(yt.mean()),
                    params=int(npar), epochs=int(eps), elapsed_s=round(el, 2),
                    pad_fraction=float(b.meta["pad_fraction"]),
                    label_mismatch=float(b.meta["label_input_mismatch"]),
                    filtered_out_rate=float(b.meta["filtered_out_rate"]),
                    pos_q10=pq[0], pos_q50=pq[1], pos_q90=pq[2],
                    neg_q10=nq[0], neg_q50=nq[1], neg_q90=nq[2],
                    val_pos_q50=vq[1], val_neg_q50=vn[1],
                    frac_pos_above_05=float((sp[yt == 1] >= 0.5).mean()) if pos_te else float("nan"),
                    pred_true_ratio=float(pred_pos / max(pos_te, 1)),
                    thr_oracle=orc_thr, macro_f1_oracle=orc_f1,
                    score_file=sfile,
                    **m)
                con.execute(
                    "INSERT OR REPLACE INTO runs(%s) VALUES(%s)"
                    % (",".join('"%s"' % k for k in rec), ",".join("?" * len(rec))),
                    tuple(rec.values()))
                con.commit()
                print(f"  {model:<12} s{seed:<2} F1 {m['macro_f1']:.4f}  AUC {m['roc_auc']:.4f}  "
                      f"pos_q50 {pq[1]:.4f}  val_pos_q50 {vq[1]:.4f}  "
                      f"ptr {rec['pred_true_ratio']:.4f}  thr* {orc_thr:.3f}  ({el:.1f}s)")

    con.close()
    print(f"\nComplete: {n_ok} runs matched, {n_bad} runs out of tolerance, {n_skip} runs skipped; "
          f"elapsed {(time.time()-t_all)/60:.1f} minutes")
    if old:
        print("  Maximum deviation of ranking metrics from old database: " +
              "  ".join("%s %.3e (tolerance %.0e)" % (c, TOL_MAX[c], TOL) for c in TOL_COLS))
        rep = os.path.join(_ROOT, "results_v2", "wp2e_identity_report.csv")
        head = not os.path.exists(rep)
        with open(rep, "a", encoding="utf-8") as f:
            if head:
                f.write("dataset,n_checked,tol,max_abs_diff_pr_auc,max_abs_diff_roc_auc,ts\n")
            f.write("%s,%d,%.0e,%.6e,%.6e,%s\n" % (
                args.dataset, n_ok, TOL, TOL_MAX["pr_auc"], TOL_MAX["roc_auc"],
                time.strftime("%Y-%m-%dT%H:%M:%S")))
    return 0


def _log_gate(bad, rid) -> None:
    path = os.path.join(_ROOT, "results_v2", "wp2e_gate_log.csv")
    head = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if head:
            f.write("run_id,column,old,new,abs_diff,ts\n")
        for c, o, n in bad:
            f.write("%s,%s,%.17g,%.17g,%.6e,%s\n" % (
                rid, c, o, n, abs(o - n), time.strftime("%Y-%m-%dT%H:%M:%S")))


def _log_prior(path, cell, split, b, tr, va, te, s0_pos) -> None:


    y = np.asarray(b.y_bin)
    head = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if head:
            f.write("cell,split,n_total,n_train,n_val,n_test,"
                    "n_pos_val,n_pos_test,"
                    "pos_rate_all,pos_rate_train,pos_rate_val,pos_rate_test,"
                    "pos_rate_train_s0,ts\n")
        f.write("%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%s\n" % (
            cell, split, len(y), len(tr), len(va), len(te),
            int(y[va].sum()) if len(va) else 0, int(y[te].sum()),
            float(y.mean()), float(y[tr].mean()),
            float(y[va].mean()) if len(va) else float("nan"),
            float(y[te].mean()), s0_pos,
            time.strftime("%Y-%m-%dT%H:%M:%S")))


def _raise_verification_error(msg: str, args) -> None:
    raise RuntimeError(
        f"{msg}\nCommand: --dataset {args.dataset} --verify-db {args.verify_db}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
