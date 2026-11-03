


from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.abspath(os.path.join(_HERE, "..", "src", "data")),
           os.path.abspath(os.path.join(_HERE, "..", "src", "models")),
           os.path.abspath(os.path.join(_HERE, "src", "data")),
           os.path.abspath(os.path.join(_HERE, "src", "models"))):
    if os.path.isdir(_c):
        sys.path.insert(0, _c)

import torch
import torch.nn as nn
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve,
)

from seqdiff import (
    IDENTITY_COLS, LABEL_COLS,
    SeqConfig, add_group_ids, build_sequences, load_days, load_parquet_dir, make_split,
)
from zoo import build_seq_model, n_params, pool_table


CELLS = []
for _pad in ("repeat_last", "zero_mask", "non_padded"):
    CELLS.append(("time_window", dict(convention="by_start", truncate="tail", padding=_pad)))
for _pad in ("repeat_last", "zero_mask", "non_padded"):
    CELLS.append(("service", dict(padding=_pad)))
for _pad in ("repeat_last", "zero_mask", "non_padded"):
    CELLS.append(("host", dict(padding=_pad)))

MODELS = ("lstm", "tcn", "transformer", "rf")


def _pos_mask(labels) -> np.ndarray:


    v = np.asarray(labels)
    if v.dtype.kind in "iufb":
        return v.astype(np.float64) != 0.0
    return np.array([str(x).strip().upper() != "BENIGN" for x in v], dtype=bool)


CUDNN_BENCH = True

def seed_everything(s: int) -> None:


    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = bool(CUDNN_BENCH)


def stratified_cap(b, cap: int, seed: int = 0):

    from seqdiff import SeqBatch
    n = len(b.wid)
    if cap <= 0 or n <= cap:
        return b, 1.0
    rng = np.random.default_rng(seed)
    pos = np.where(b.y_bin == 1)[0]
    neg = np.where(b.y_bin == 0)[0]
    k_pos = min(len(pos), int(round(cap * b.y_bin.mean())))
    k_neg = min(len(neg), cap - k_pos)
    idx = np.sort(np.concatenate([
        rng.choice(pos, k_pos, replace=False) if len(pos) > k_pos else pos,
        rng.choice(neg, k_neg, replace=False) if len(neg) > k_neg else neg]))
    return SeqBatch(X=b.X[idx], mask=b.mask[idx], length=b.length[idx],
                    y_bin=b.y_bin[idx], y_multi=b.y_multi[idx],
                    group=b.group[idx], wid=b.wid[idx],
                    day=(None if b.day is None else b.day[idx]), meta=b.meta), len(idx) / n


def metrics_of(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fpr = fp / max(tn + fp, 1)
    return dict(
        macro_f1=float(f1_score(y, yhat, average="macro", zero_division=0)),
        f1_pos=float(f1_score(y, yhat, pos_label=1, zero_division=0)),
        fpr=float(fpr),
        pr_auc=float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),


        roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        mcc=float(matthews_corrcoef(y, yhat)),
    )


def fit_scaler(X, mask):
    v = X[mask]
    mu = v.mean(0)
    sd = v.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(X, mu, sd):
    return np.clip((X - mu) / sd, -10.0, 10.0).astype(np.float32)


def train_seq(name, Xtr, Mtr, ytr, Xva, Mva, yva, Xte, Mte, yte, seed, dev,
              max_epochs=20, batch=256, patience_max=5,
              lr=1e-3, warmup_frac=0.0):
    seed_everything(seed)
    model = build_seq_model(name, Xtr.shape[2], Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cnt = np.bincount(ytr, minlength=2).astype(np.float64)
    w = torch.tensor((cnt.sum() / (2 * np.maximum(cnt, 1))), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)

    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev)
    Xtr_t, Mtr_t, ytr_t = t(Xtr), t(Mtr, torch.bool), t(ytr, torch.long)
    Xva_t, Mva_t = t(Xva), t(Mva, torch.bool)
    Xte_t, Mte_t = t(Xte), t(Mte, torch.bool)

    bs = batch
    n = len(ytr)


    _spe = max(1, int(np.ceil(n / bs)))
    _total = max(1, max_epochs * _spe)
    _warm = max(1, int(_total * warmup_frac)) if warmup_frac > 0 else 0

    def _lr_lambda(step):
        if _warm and step < _warm:
            return float(step + 1) / float(_warm)
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = np.random.default_rng(seed * 1000 + ep).permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            out = model(Xtr_t[idx], Mtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(Xva_t, Mva_t), -1)[:, 1].cpu().numpy()
        f1 = f1_score(yva, (pv >= 0.5).astype(int), average="macro", zero_division=0)
        if f1 > best_f1 + 1e-6:
            best_f1, best_state, patience = float(f1), {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= patience_max:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(Xte_t, Mte_t), -1)[:, 1].cpu().numpy()
    return p, n_params(model), ep + 1


def train_rf(Xtr, Mtr, ytr, Xte, Mte, seed):
    Ftr = pool_table(Xtr, Mtr)
    Fte = pool_table(Xte, Mte)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=16,
                                 class_weight="balanced_subsample", random_state=seed)
    clf.fit(Ftr, ytr)
    p = clf.predict_proba(Fte)[:, 1]
    return p, 0, 0


def train_hgb(Xtr, Mtr, ytr, Xte, Mte, seed, class_weight="balanced"):


    Ftr = pool_table(Xtr, Mtr)
    Fte = pool_table(Xte, Mte)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.15,
        class_weight=class_weight, random_state=seed)
    clf.fit(Ftr, ytr)
    p = clf.predict_proba(Fte)[:, 1]
    return p, 0, int(clf.n_iter_)


def cap_train_indices(y, idx, cap: int, seed: int = 0):

    idx = np.asarray(idx)
    if cap <= 0 or len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    pos = idx[np.asarray(y)[idx] == 1]
    neg = idx[np.asarray(y)[idx] == 0]
    frac = cap / float(len(idx))
    k_pos = int(round(len(pos) * frac))
    k_neg = int(round(len(neg) * frac))
    keep_p = rng.choice(pos, k_pos, replace=False) if len(pos) > k_pos else pos
    keep_n = rng.choice(neg, k_neg, replace=False) if len(neg) > k_neg else neg
    return np.sort(np.concatenate([keep_p, keep_n]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/external")
    ap.add_argument("--days", nargs="+", default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--db", default="results/runs.db")
    ap.add_argument("--out", default="results/pilot_summary.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--max-seq", type=int, default=0, help="0=no sampling; >0 downsamples each route")
    ap.add_argument("--max-segments", type=int, default=0,
                    help="Segment-level sampling cap applied before constructing X. 0=unlimited. "
                         "The CIC18 host route has about 3 million segments; set a cap or X will use 20GB+")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate (default 1e-3, consistent with 5000+ historical runs). "
                         "Changing lr requires --rid-prefix; otherwise --resume skips run_id values shared with old arms")
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="Fraction of total steps for linear warmup. 0=no scheduler (original behavior)")
    ap.add_argument("--hgb-class-weight", default="balanced",
                    choices=["balanced", "none"],
                    help="HGB class_weight. Use none for ablation to test whether "
                         "balanced overweights positives on nearly balanced data (changing it requires --rid-prefix)")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="Skip run_id values already present in runs.db")
    ap.add_argument("--models", default="lstm,tcn,transformer,rf")
    ap.add_argument("--cell-filter", default="", help="Run only cells whose tag contains this substring")
    ap.add_argument("--t-values", type=int, nargs="+", default=[20],
                    help="T ablation: run the full grid for each T in turn (run_id includes T to avoid conflicts)")
    ap.add_argument("--cic18-days", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="CIC18 days to use (chronological _day indices). All 10 days contain 63 million rows and exhaust memory")
    ap.add_argument("--unsw-days", type=int, nargs="+", default=[0, 2],
                    help="UNSW days to use: 0=2015-01-22(p=1.6%%), 1=01-23(all benign), "
                         "2=02-18(p=9.0%%). Day 1 has no positives and cannot be used in cross_day")
    ap.add_argument("--fold", type=int, default=0,
                    help="Fold index from cross_day_pair (0/1). Appends |f{i} to run_id when folds>1")
    ap.add_argument("--max-train", type=int, default=0,
                    help="Stratified downsampling only on training (test unchanged). For the common-sample-budget comparison: "
                         "--max-seq also downsamples the test set, altering it and making it unsuitable for this comparison")
    ap.add_argument("--split", default="cross_day",
                    choices=["cross_day", "random", "temporal", "cross_day_pair"],
                    help="Split strategy. Use random only for the protocol-hardness comparison: "
                         "it allows the same host across splits, inflating scores; not valid for primary results")
    ap.add_argument("--feature-subset", default="",
                    help="Keep only feature columns listed in this JSON. Required for cross-dataset comparisons; "
                         "otherwise CIC18 includes 5 features absent from CIC17 (see P5 §6)")
    ap.add_argument("--rid-prefix", default="",
                    help="Override run_id prefix. Required for extended windows such as 8 days; "
                         "otherwise --resume skips all run_id values shared with existing arms")
    ap.add_argument("--dataset", default="cic17", choices=["cic17", "cic18", "unsw"],
                    help="cic17=corrected engelen 5 days; cic18=CSE-CIC-IDS2018 10 days")
    ap.add_argument("--cudnn-benchmark", type=int, default=1,
                    help="1=runtime convolution algorithm autotuning (setting for all 6150 historical runs, "
                         "but **not reproducible across processes**); 0=fixed algorithm, reproducible, about 13%% slower per round. "
                         "Old DBs all use 1; explicitly pass 0 for new DBs and record it in --db to avoid mixing")
    args = ap.parse_args()

    global CUDNN_BENCH
    CUDNN_BENCH = bool(args.cudnn_benchmark)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device {dev}  torch {torch.__version__}")
    print(f"cudnn.benchmark={CUDNN_BENCH}  "
          f"({'autotuned, not reproducible across processes' if CUDNN_BENCH else 'fixed algorithm, reproducible'})")
    if dev == "cuda":
        print(f"GPU {torch.cuda.get_device_name(0)}")

    if args.dataset == "unsw":


        df = add_group_ids(load_parquet_dir(
            f"{args.root}/protocol_robust_nids_20261028/data/processed/unsw",
            ts_guard=("2015-01-01", "2015-03-01"),
            days=args.unsw_days))
        ds_tag = "UNSW-" + str(len(args.unsw_days)) + "d"
        rid_prefix = args.rid_prefix or "unsw"
        _pm = _pos_mask(df["Label"])
        print(f"Data {ds_tag}: {len(df):,} flows, days {df['_day'].nunique()}, "
              f"attack-flow share {_pm.mean():.4f}")
        print("  Attack-flow share by day: " + "  ".join(
            f"d{i}={_pos_mask(df.loc[df['_day']==i, 'Label']).mean()*100:.1f}%"
            for i in sorted(df["_day"].unique())))
    elif args.dataset == "cic18":


        df = add_group_ids(load_parquet_dir(
            f"{args.root}/protocol_robust_nids_20261028/data/processed/cic18",
            days=args.cic18_days))
        ds_tag = "CIC18-" + str(len(args.cic18_days)) + "d"
        rid_prefix = args.rid_prefix or "cic18"
        print(f"Data {ds_tag}: {len(df):,} flows, days {df['_day'].nunique()}, attack-flow share "
              f"{(df['Label'].astype(str).str.upper()!='BENIGN').mean():.4f}")
        print("  Attack-flow share by day: " + "  ".join(
            f"d{i}={(df.loc[df['_day']==i,'Label'].astype(str).str.upper()!='BENIGN').mean()*100:.1f}%"
            for i in sorted(df["_day"].unique())))
    else:
        df = add_group_ids(load_days(args.root, "engelen", args.days))
        ds_tag = "CIC17-" + "+".join(d[:3] for d in args.days)

        rid_prefix = args.rid_prefix or "+".join(d[:3] for d in args.days)
        print(f"Data {', '.join(args.days)}: {len(df):,} flows, attack-flow share "
              f"{(df['Label'].astype(str).str.upper()!='BENIGN').mean():.4f}")
        print("  Attack-flow share by day: " + "  ".join(
            f"{d[:3]}={(df.loc[df['_day']==i,'Label'].astype(str).str.upper()!='BENIGN').mean()*100:.1f}%"
            for i, d in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday"])
            if (df["_day"] == i).any()))

    if args.feature_subset:
        import json as _json
        _keep = set(_json.load(open(args.feature_subset, encoding="utf-8")))
        _keep |= set(IDENTITY_COLS) | set(LABEL_COLS)
        _drop = [c for c in df.columns
                 if c not in _keep and not c.startswith("_")]
        if _drop:
            df = df.drop(columns=_drop)
        print(f"  Feature subset: retained {df.shape[1]} columns, dropped {len(_drop)} columns {_drop}")

    cells = CELLS[:1] if args.smoke else CELLS
    if args.cell_filter:
        cells = [c for c in cells if args.cell_filter in f"{c[0]}/{c[1]['padding']}"]
        if not cells:
            print(f"No cells match {args.cell_filter!r}")
            return 1
    models = ("lstm",) if args.smoke else tuple(args.models.split(","))
    seeds = args.seeds[:1] if args.smoke else args.seeds


    cells = [(r, dict(kw), T) for T in args.t_values for r, kw in cells]

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    con = sqlite3.connect(args.db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY, ts TEXT, dataset TEXT, route TEXT, cell TEXT,
            convention TEXT, truncate TEXT, padding TEXT, T INTEGER, split TEXT,
            model TEXT, seed INTEGER, n_seq INTEGER, n_train INTEGER, n_test INTEGER,
            test_pos_rate REAL, macro_f1 REAL, f1_pos REAL, fpr REAL, pr_auc REAL, mcc REAL,
            roc_auc REAL,
            params INTEGER, epochs INTEGER, elapsed_s REAL,
            pad_fraction REAL, label_mismatch REAL, filtered_out_rate REAL
        )""")
    con.commit()

    rows, skipped = [], []
    t_all = time.time()
    total = len(cells) * len(models) * len(seeds)
    k = 0

    for route, kw, T in cells:
        cfg = SeqConfig(segment=route, T=T, seed=0,
                        max_segments=args.max_segments, **kw)
        t0 = time.time()
        b = build_sequences(df, cfg)
        n_before = len(b.wid)
        b, kept = stratified_cap(b, args.max_seq, seed=0)
        build_s = time.time() - t0
        if kept < 1.0:
            print(f"  Sequences {n_before:,} → downsampled to {len(b.wid):,} (retained {kept:.2f})")
        cell = cfg.tag()
        print(f"\n[{cell}] Sequences {len(b.wid):,}  positive rate {b.meta['pos_rate']:.4f}  "
              f"padding {b.meta['pad_fraction']:.4f}  label mismatch {b.meta['label_input_mismatch']:.4f}  "
              f"filtered out {b.meta['filtered_out_rate']:.3f}  built in {build_s:.1f}s")

        folds = make_split(b, args.split, seed=0)
        if args.fold >= len(folds):
            raise SystemExit(f"--fold {args.fold} out of range ({args.split} has {len(folds)} folds)")
        fold = folds[args.fold]

        fold_tag = f"|f{args.fold}" if len(folds) > 1 else ""
        tr, va, te = fold["train"], fold["val"], fold["test"]
        if args.max_train:
            n_tr0 = len(tr)
            tr = cap_train_indices(b.y_bin, tr, args.max_train, seed=0)
            print(f"  Stratified training downsampling {n_tr0:,} → {len(tr):,}"
                  f" (test set unchanged at {len(te):,})")
        pos_te = int(b.y_bin[te].sum())
        neg_te = int(len(te) - pos_te)
        pos_tr = int(b.y_bin[tr].sum())
        if pos_tr < 5 or pos_te < 5 or neg_te < 5:
            print(f"  [skip] Insufficient classes: train_pos={pos_tr} test_pos={pos_te} "
                  f"test_neg={neg_te} → cell is not comparable under this split")
            skipped.append(dict(cell=cell, reason="single-class split",
                                train_pos=pos_tr, test_pos=pos_te, test_neg=neg_te))
            continue
        _tr = set(np.asarray(b.group)[tr].tolist())
        _lk = float(np.mean([x in _tr for x in np.asarray(b.group)[te]]))

        print(f"  Fraction of test group keys seen in training {_lk:.4f} (descriptive only for cross_day)")

        mu, sd = fit_scaler(b.X[tr], b.mask[tr])
        Xn = apply_scaler(b.X, mu, sd)

        existing = set()
        if args.resume:
            existing = {r[0] for r in con.execute("select run_id from runs").fetchall()}

        for model in models:
            for seed in seeds:
                k += 1
                rid_pre = f"{rid_prefix}|{cell}|{args.split}{fold_tag}|{model}|s{seed}"
                if args.resume and rid_pre in existing:
                    print(f"  [{k}/{total}] {model:<12} s{seed} already exists; skipping")
                    continue
                t1 = time.time()
                try:
                    if model == "rf":
                        p, npar, eps = train_rf(Xn[tr], b.mask[tr], b.y_bin[tr],
                                                Xn[te], b.mask[te], seed)
                    elif model == "hgb":
                        _cw = None if args.hgb_class_weight == "none" else "balanced"
                        p, npar, eps = train_hgb(Xn[tr], b.mask[tr], b.y_bin[tr],
                                                 Xn[te], b.mask[te], seed,
                                                 class_weight=_cw)
                    else:
                        p, npar, eps = train_seq(model, Xn[tr], b.mask[tr], b.y_bin[tr],
                                                 Xn[va], b.mask[va], b.y_bin[va],
                                                 Xn[te], b.mask[te], b.y_bin[te], seed, dev,
                                                 max_epochs=args.max_epochs, batch=args.batch,
                                                 patience_max=args.patience,
                                                 lr=args.lr, warmup_frac=args.warmup_frac)
                    m = metrics_of(b.y_bin[te], p)
                except Exception as e:
                    print(f"  [{k}/{total}] {model} seed{seed} failed: {type(e).__name__}: {e}")
                    continue
                el = time.time() - t1
                rid = f"{rid_prefix}|{cell}|{args.split}{fold_tag}|{model}|s{seed}"
                rec = dict(
                    run_id=rid, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), dataset=ds_tag,
                    route=route, cell=cell, convention=kw.get("convention", "-"),
                    truncate=kw.get("truncate", "-"), padding=kw["padding"], T=T,
                    split=args.split, model=model, seed=seed, n_seq=len(b.wid),
                    n_train=len(tr), n_test=len(te),
                    test_pos_rate=float(b.y_bin[te].mean()),
                    params=int(npar), epochs=int(eps), elapsed_s=round(el, 2),
                    pad_fraction=float(b.meta["pad_fraction"]),
                    label_mismatch=float(b.meta["label_input_mismatch"]),
                    filtered_out_rate=float(b.meta["filtered_out_rate"]),
                    **m)
                rows.append(rec)
                con.execute(
                    f"INSERT OR REPLACE INTO runs({','.join(rec)}) VALUES({','.join('?'*len(rec))})",
                    tuple(rec.values()))
                con.commit()
                print(f"  [{k}/{total}] {model:<12} s{seed}  macro-F1 {m['macro_f1']:.4f}  "
                      f"FPR {m['fpr']:.4f}  PR-AUC {m['pr_auc']:.4f}  MCC {m['mcc']:.4f}  "
                      f"({el:.1f}s, {eps}ep)")

    con.close()


    if not rows:
        print("No valid results")
        return 1

    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["cell"], r["model"])].append(r["macro_f1"])


    cells_seen = list(dict.fromkeys(r["cell"] for r in rows))
    models_seen = [m for m in MODELS if any(r["model"] == m for r in rows)]

    print("\n" + "=" * 96)
    print("Pilot results: macro-F1 (mean±std over seeds)")
    print("=" * 96)
    head = f"{'Operator cell':<44}" + "".join(f"{m:>13}" for m in models_seen)
    print(head)
    print("-" * len(head))
    for c in cells_seen:
        line = f"{c:<44}"
        for m in models_seen:
            v = agg.get((c, m), [])
            line += f"{np.mean(v):>8.3f}±{np.std(v):.3f}" if v else f"{'-':>13}"
        print(line)

    print("\nCross-operator swing vs cross-model spread:")
    summary = {}
    for c in cells_seen:
        vals = {m: float(np.mean(agg[(c, m)])) for m in models_seen if agg.get((c, m))}
        if len(vals) < 2:
            continue
        best = max(vals, key=vals.get)
        spread_model = max(vals.values()) - min(vals.values())
        summary[c] = dict(by_model=vals, best=best, model_spread=spread_model)
        print(f"  {c:<44} best {best:<12} cross-model spread {spread_model:.4f}")

    print()
    for m in models_seen:
        v = [np.mean(agg[(c, m)]) for c in cells_seen if agg.get((c, m))]
        if len(v) >= 2:
            print(f"  {m:<12} across operators {min(v):.3f} — {max(v):.3f}  swing {max(v)-min(v):.3f}")

    spreads = [s["model_spread"] for s in summary.values()]
    print(f"\n  Median cross-model spread {np.median(spreads):.4f}")
    rank_changes = len({s["best"] for s in summary.values()})
    print(f"  Distinct best models across operators: {rank_changes} → "
          f"{'Unstable ranking ✅ claim supported' if rank_changes > 1 else 'Stable ranking; use swing evidence'}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, skipped=skipped, summary=summary, n_runs=len(rows),
                       elapsed_s=round(time.time() - t_all, 1)), f,
                  ensure_ascii=False, indent=2)
    print(f"\nWrote {args.out}  total elapsed {(time.time()-t_all)/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
