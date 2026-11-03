#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "data"))
sys.path.insert(0, os.path.join(ROOT, "scripts_v2"))

from seqdiff import add_group_ids, load_days, load_parquet_dir

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_rules as ER

R2 = os.path.join(ROOT, "results_v2")
S2 = os.path.join(ROOT, "scores_v2")
PAT = re.compile(
    r"^(?P<pre>[A-Za-z0-9]+?)_(?P<cell>.+?)_(?P<split>cross_day|cross_day_pair)"
    r"(?:_(?P<fold>f\d+))?_(?P<model>lstm|tcn|transformer|rf|hgb)_s(?P<seed>\d+)\.npz$")


def load_prior() -> dict:


    out = {}
    p = os.path.join(R2, "wp4_prior.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            out[(r.setting, r.cell, int(r.fold))] = float(r.pos_rate_train)
    q = os.path.join(R2, "wp2c_prior.csv")
    if os.path.exists(q):
        d = pd.read_csv(q)
        d = d[d.pos_rate_train_s0.notna()].drop_duplicates("cell", keep="last")
        for _, r in d.iterrows():
            out.setdefault(("cic18", r.cell, -1), float(r.pos_rate_train_s0))
    return out


def adjust(p, pi: float, pi_src: float):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    if pi <= 0.0:
        return np.zeros_like(p)
    if pi >= 1.0:
        return np.ones_like(p)
    r = (pi / pi_src) / ((1.0 - pi) / (1.0 - pi_src))
    return r * p / (r * p + (1.0 - p))


def saerens(s, pi_src: float, tol=1e-8, max_iter=200) -> np.ndarray:
    pi = float(pi_src)
    for _ in range(max_iter):
        w = adjust(s, pi, pi_src)
        new = float(w.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return adjust(s, pi, pi_src)


def otsu(s: np.ndarray, nb: int = 256) -> float:
    v = np.clip(np.asarray(s, float), 0.0, 1.0)
    hist, edges = np.histogram(v, bins=nb, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    m1 = m0[-1] - m0
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = m1 / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
    var[~np.isfinite(var)] = -1.0
    return float(centers[int(np.argmax(var))])


def _qgrid(x: np.ndarray, n: int = 999) -> np.ndarray:


    x = np.sort(np.asarray(x, float))
    if len(x) == 0:
        return x
    if len(x) <= n:
        return x
    return np.unique(np.quantile(x, np.linspace(0.0, 1.0, n)))


def make_run(s: np.ndarray, y: np.ndarray, pi_tr: float) -> dict:


    pos = s[y == 1]
    neg = s[y == 0]
    qp, qn = _qgrid(pos), _qgrid(neg)
    npos, nneg = len(pos), len(neg)
    v, w = ER._pooled(qp, qn, npos, nneg)
    return dict(v=v, w=w, qp=qp, qn=qn, npos=npos, nneg=nneg, prior=pi_tr)


RULES_NOVAL = [
    ("fixed0.5", ER.thr_fixed),
    ("qtrain", ER.thr_qtrain),
    ("qtrain_gt", ER.thr_qtrain_gt),
    ("otsu", ER.thr_otsu),
    ("otsu_logit", ER.thr_otsu_logit),
    ("hybrid", ER.thr_hybrid),
    ("hybrid50", ER.thr_hybrid50),
    ("gmm_logit", ER.thr_gmm),
    ("gmm_valley", ER.thr_gmm_valley),
]
RULES_NEED_VAL = ["val_thr", "val_rank", "scale_val", "bbse", "saerens_cal"]


def rule_thresholds(run: dict) -> dict:

    out = {}
    for name, fn in RULES_NOVAL:
        if name.startswith("qtrain") and not (np.isfinite(run["prior"])
                                              and 0.0 < run["prior"] < 1.0):
            continue
        try:
            out[name] = float(fn(run))
        except Exception:
            continue
    if np.isfinite(run["prior"]) and 0.0 < run["prior"] < 1.0:
        o = np.argsort(run["v"])
        vv = np.asarray(run["v"], float)[o]
        adj = np.asarray(saerens(vv, run["prior"]), float)

        idx = np.where(adj >= 0.5)[0]
        out["saerens"] = float(vv[idx[0]]) if len(idx) else float(np.inf)
    return out


def attack_intervals(ts: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = np.argsort(ts, kind="mergesort")
    ts, te = ts[o], te[o]
    lo, hi = [ts[0]], [te[0]]
    for a, b in zip(ts[1:], te[1:]):
        if a <= hi[-1]:
            hi[-1] = max(hi[-1], b)
        else:
            lo.append(a)
            hi.append(b)
    return np.asarray(lo), np.asarray(hi)


def clean_windows(lo, hi, day_lo, day_hi, min_s):

    gaps = []
    cur = day_lo
    for a, b in zip(lo, hi):
        if a > cur:
            gaps.append((cur, min(a, day_hi)))
        cur = max(cur, b)
    if cur < day_hi:
        gaps.append((cur, day_hi))
    return [(a, b) for a, b in gaps if (b - a) >= min_s]


def win_slug(cell_db: str) -> str:
    return cell_db.replace("/", "_")


def window_s(cell_slug: str) -> float:
    m = re.match(r"tw(\d+)s", cell_slug)
    if not m:
        raise ValueError(f"Not a time_window cell: {cell_slug}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--datasets", nargs="+", default=["cic17", "unsw"],
                    choices=["cic17", "cic18", "unsw"])
    ap.add_argument("--min-minutes", type=float, default=60.0)
    ap.add_argument("--min-seqs", type=int, default=20,
                    help="Exclude windows with fewer sequences than this value (FPR is not meaningful for very small samples)")
    ap.add_argument("--days", nargs="+",
                    default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    ap.add_argument("--cic18-days", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--unsw-days", type=int, nargs="+", default=[0, 2])
    args = ap.parse_args()

    min_s = args.min_minutes * 60.0
    priors = load_prior()
    priors_cells = {}
    _p = os.path.join(R2, "wp4_prior.csv")
    if os.path.exists(_p):
        for (dsx, cx), _ in pd.read_csv(_p).groupby(["dataset", "cell"]):
            priors_cells.setdefault(dsx, set()).add(cx)
    _q = os.path.join(R2, "wp2c_prior.csv")
    if os.path.exists(_q):
        for cx in pd.read_csv(_q).cell.unique():
            priors_cells.setdefault("cic18", set()).add(cx)
    win_rows, run_rows, gap_rows = [], [], []
    bad = 0

    for ds in args.datasets:
        setting = "unsw_pair" if ds == "unsw" else ds
        if ds == "unsw":
            df = load_parquet_dir(
                f"{args.root}/protocol_robust_nids_20261028/data/processed/unsw",
                ts_guard=("2015-01-01", "2015-03-01"), days=args.unsw_days)
        elif ds == "cic18":
            df = load_parquet_dir(
                f"{args.root}/protocol_robust_nids_20261028/data/processed/cic18",
                days=args.cic18_days)
        else:
            df = load_days(args.root, "engelen", args.days)
        df = add_group_ids(df)
        print(f"\n=== {ds} flows {len(df):,} days {df['_day'].nunique()} ===", flush=True)

        lab = df["Label"]
        atk = (pd.to_numeric(lab, errors="coerce").fillna(0).to_numpy() != 0
               if pd.api.types.is_numeric_dtype(lab)
               else lab.astype(str).str.upper().to_numpy() != "BENIGN")
        ts_all = (df["_t_start"] - df["_t_start"].min()).dt.total_seconds().to_numpy()
        te_all = (df["_t_end"] - df["_t_start"].min()).dt.total_seconds().to_numpy()
        day_all = df["_day"].to_numpy()


        base = os.path.join(S2, ds)
        files = sorted(os.listdir(base))
        tw = [f for f in files if PAT.match(f) and PAT.match(f)["cell"].startswith("tw")]
        test_days = sorted({int(np.load(os.path.join(base, f), allow_pickle=True)["day"][0]) for f in tw})
        print(f"  Test days {test_days}, tw files {len(tw)}", flush=True)

        cw = {}
        for d in test_days:
            m = (day_all == d) & atk
            if m.sum() == 0:
                print(f"  [day {d}] no attack flows; skipping (this is itself a zero-attack day)")
                continue
            lo, hi = attack_intervals(ts_all[m], te_all[m])
            sel = day_all == d
            dl, dh = float(ts_all[sel].min()), float(te_all[sel].max())
            allg = clean_windows(lo, hi, dl, dh, 0.0)
            gl = np.asarray([(b - a) / 60.0 for a, b in allg]) if allg else np.zeros(0)
            gap_rows.append(dict(
                dataset=ds, day=d, span_min=(dh - dl) / 60.0, n_attack_intervals=len(lo),
                n_gaps=len(gl),
                max_gap_min=float(gl.max()) if len(gl) else 0.0,
                n_ge_5min=int((gl >= 5).sum()), n_ge_15min=int((gl >= 15).sum()),
                n_ge_30min=int((gl >= 30).sum()), n_ge_60min=int((gl >= 60).sum()),
                clean_cover_frac=float(gl[gl >= args.min_minutes].sum() / max(1e-9, (dh - dl) / 60.0)),
            ))
            gaps = clean_windows(lo, hi, dl, dh, min_s)
            cw[d] = gaps
            span_min = (dh - dl) / 60.0
            cover = sum(b - a for a, b in gaps) / max(1.0, dh - dl)
            mg = float(gl.max()) if len(gl) else 0.0
            print(f"  [day {d}] attack intervals {len(lo)}, day span {span_min:.0f} min, "
                  f"longest attack-free gap {mg:.1f} min, ≥60min windows {len(gaps)}, "
                  f"time coverage {cover*100:.1f}%", flush=True)
            for k, (a, b) in enumerate(gaps):
                win_rows.append(dict(dataset=ds, day=d, win_id=f"d{d}w{k}",
                                     t0_s=a, t1_s=b, minutes=(b - a) / 60.0))


        cells = sorted({PAT.match(f)["cell"] for f in files if PAT.match(f)})
        slug2db = {win_slug(c): c for c in priors_cells.get(ds, [])}
        for cell in [c for c in cells if c.startswith("tw")]:
            w = window_s(cell)
            cell_db = slug2db.get(cell, cell)
            for f in [x for x in files if PAT.match(x) and PAT.match(x)["cell"] == cell]:
                mm = PAT.match(f)
                z = np.load(os.path.join(base, f), allow_pickle=True)
                t0 = np.asarray(z["t_start"], dtype=np.float64)
                y = np.asarray(z["label"]).astype(int)
                s = np.asarray(z["score"], dtype=np.float64)
                d = int(np.asarray(z["day"])[0])
                if d not in cw:
                    continue
                t1 = t0 + w
                keep = np.zeros(len(t0), bool)
                for a, b in cw[d]:
                    keep |= (t0 >= a) & (t1 <= b)
                if keep.sum() == 0:
                    continue
                if y[keep].sum() > 0:
                    print(f"  [gate failed] {f}: {int(y[keep].sum())} positives inside clean windows")
                    bad += 1
                fold = int(mm["fold"][1:]) if mm["fold"] else -1
                pi_tr = priors.get((setting, cell_db, fold), np.nan)
                run = make_run(s, y, pi_tr)
                thrs = rule_thresholds(run)
                row = dict(dataset=ds, cell=cell_db, model=mm["model"], seed=int(mm["seed"]),
                           fold=fold, n_clean_seq=int(keep.sum()),
                           n_clean_windows=len(cw[d]),
                           clean_minutes=sum(b - a for a, b in cw[d]) / 60.0,
                           prior=float(pi_tr) if np.isfinite(pi_tr) else np.nan)
                for k, t in thrs.items():
                    if not np.isfinite(t):
                        f = 1.0 if t > 0 else 0.0
                        row[f"fpr_{k}"] = f
                        row[f"fprfull_{k}"] = f
                        row[f"thr_{k}"] = t
                        continue
                    row[f"thr_{k}"] = t
                    row[f"fpr_{k}"] = float((s[keep] >= t).mean())
                    row[f"fprfull_{k}"] = float((s >= t).mean())
                run_rows.append(row)
            med = np.median([r["n_clean_seq"] for r in run_rows if r["cell"] == cell_db])
            print(f"  [{cell_db}] median sequences inside clean windows {med:.0f}", flush=True)
        del df

    run = pd.DataFrame(run_rows)
    run.to_csv(os.path.join(R2, "wp4_clean_run.csv"), index=False)
    pd.DataFrame(gap_rows).to_csv(os.path.join(R2, "wp4_gap_stats.csv"), index=False)
    if run.empty:
        print("\n[conclusion] No test day in the processed datasets has a clean window meeting the threshold. "
              "See the longest attack-free gaps in wp4_gap_stats.csv.")

    if run.empty:
        L = ["# WP4a: Clean windows within test days", "",
             f"Threshold: at least {args.min_minutes:g} consecutive minutes without attack flows.", "",
             "**Result: no test day in the three datasets has a clean window meeting the threshold.**", "",
             "| Dataset | Test day | Day span (min) | Attack intervals | Longest attack-free gap (min) | ≥30min gaps | ≥60min windows |",
             "|---|---|---|---|---|---|---|"]
        for _, r in pd.DataFrame(gap_rows).iterrows():
            L.append(f"| {r.dataset} | d{r.day} | {r.span_min:.0f} | {r.n_attack_intervals} | "
                     f"{r.max_gap_min:.1f} | {r.n_ge_30min} | {r.n_ge_60min} |")
        L += ["", "Manuscript wording: do not claim 'zero false positives in clean windows'; state that 'no continuous 60-minute "
              "attack-free period exists within the test days, so same-day data cannot provide an operational false-positive probe', "
              "and use this as direct evidence that 'test-day negatives do not imply attack-free periods'."]
        path = os.path.join(R2, "WP4A_SUMMARY.md")
        open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
        print(f"→ {path}")
        return 0

    def wilson(k, n, z=1.959963984540054):
        if n == 0:
            return np.nan, np.nan
        ph = k / n
        dd = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / dd
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / dd
        return max(0.0, c - h), min(1.0, c + h)

    agg = []
    rules = sorted({c[4:] for c in run.columns if c.startswith("fpr_")})
    for (ds, cell), g in run.groupby(["dataset", "cell"]):
        for rule in rules:
            v = g[f"fpr_{rule}"].dropna()
            k = int((v > 0.05).sum())
            lo, hi = wilson(k, len(v))
            vf = g[f"fprfull_{rule}"].dropna() if f"fprfull_{rule}" in g else pd.Series(dtype=float)
            agg.append(dict(dataset=ds, cell=cell, rule=rule, n_runs=len(v),
                            n_clean_seq=int(g.n_clean_seq.iloc[0]),
                            n_windows=int(g.n_clean_windows.iloc[0]),
                            clean_minutes=float(g.clean_minutes.iloc[0]),
                            fpr_median=float(np.median(v)) if len(v) else np.nan,
                            fprfull_median=float(np.median(vf)) if len(vf) else np.nan,
                            n_fpr_gt5=k, frac_fpr_gt5=(k / len(v)) if len(v) else np.nan,
                            frac_full_gt5=float((vf > 0.05).mean()) if len(vf) else np.nan,
                            ci_lo=lo, ci_hi=hi))
    a = pd.DataFrame(agg)
    a.to_csv(os.path.join(R2, "wp4_clean_windows.csv"), index=False)
    pd.DataFrame(win_rows).to_csv(os.path.join(R2, "wp4_clean_window_list.csv"), index=False)

    L = ["# WP4a: Clean windows within test days", "",
         f"Clean-window definition: complements of attack-flow intervals on the test-day flow timeline, with length ≥ {args.min_minutes:g} minutes.",
         "**These windows occur on the same days as attack periods and are not independent clean days**—the manuscript must include this qualification.",
         "Computed only for the three `time_window` cells: session/service/host sequences are not time windows "
         "and cannot be wholly contained within a 60-minute window.", "",
         "**Threshold convention (corrected 2026-11-02)**: each rule's cutoff is determined from scores for **all sequences on that day**,",
         "then applied to the clean-window subset only for positive/negative classification. This matches the main table in §6.5.",
         "The old version also determined cutoffs from the clean-window subset, making `qtrain` clean-window FPR equal to π_tr by construction;",
         "that row measured nothing. Results before and after the correction must not be mixed.", "",
         "**Only 9 rules can be computed**: `val_thr`/`val_rank`/`scale_val`/`bbse`/`saerens_cal`",
         "require labels or scores from the previous day, while clean-window analysis uses only test-day scores; these 5 rules are therefore omitted here.",
         "The 'full day' column is the FPR of the same threshold on all sequences that day, used to compare distributions inside and outside windows.", ""]
    for ds, g in a.groupby("dataset"):
        L.append(f"## {ds}")
        L.append("")
        L.append("| Cell | Rule | Clean sequences | Windows | Median clean-window FPR | Median full-day FPR | "
                 "Clean-window runs >5% | Full-day runs >5% | Wilson 95% CI (clean window) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            L.append(f"| `{r.cell}` | {r.rule} | {int(r.n_clean_seq)} | {int(r.n_windows)} | "
                     f"{r.fpr_median:.4f} | {r.fprfull_median:.4f} | "
                     f"{r.frac_fpr_gt5*100:.1f}% | {r.frac_full_gt5*100:.1f}% | "
                     f"[{r.ci_lo*100:.1f}, {r.ci_hi*100:.1f}] |")
        L.append("")
    path = os.path.join(R2, "WP4A_SUMMARY.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\n→ {path}")
    print(f"→ {os.path.join(R2, 'wp4_clean_windows.csv')}")
    if bad:
        print(f"Gate failures: {bad}; positives appeared inside clean windows")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
