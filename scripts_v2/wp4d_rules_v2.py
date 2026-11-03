#!/usr/bin/env python


from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import sqlite3
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_rules as ER

DELTA = 0.02
THR = 0.5
Z = 1.959963984540054


NEED_VAL = {"val_thr", "val_rank", "scale_val", "bbse", "saerens_cal"}

SETTINGS = [
    dict(name="cic18", pds="cic18", te="scores_v2/cic18", va="scores_v2_val/cic18",
         db="det_v2_cic18.db", split="cross_day"),
    dict(name="unsw_pair", pds="unsw", te="scores_v2/unsw", va="scores_v2_val/unsw",
         db="det_v2_unsw.db", split="cross_day_pair"),
    dict(name="cic17", pds="cic17", te="scores_v2/cic17", va="scores_v2_val/cic17",
         db="det_v2_cic17.db", split="cross_day"),
    dict(name="cic17nd", pds="cic17nd", te="scores_v2_nd/cic17", va=None,
         db="nd_cic17.db", split="cross_day_nd"),
]

PAT = re.compile(
    r"^(?P<pre>[A-Za-z0-9]+?)_(?P<cell>.+?)_(?P<split>cross_day|cross_day_pair|cross_day_nd)"
    r"(?:_(?P<fold>f\d+))?_(?P<model>lstm|tcn|transformer|rf|hgb)_s(?P<seed>\d+)\.npz$")


def wilson(k: int, n: int, z: float = Z):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def fpr_of(y, pred) -> float:
    neg = y == 0
    return float(pred[neg].mean()) if neg.any() else float("nan")


def adjust(p, pi: float, pi_src: float):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    if pi <= 0.0:
        return np.zeros_like(p)
    if pi >= 1.0:
        return np.ones_like(p)
    r = (pi / pi_src) / ((1.0 - pi) / (1.0 - pi_src))
    return r * p / (r * p + (1.0 - p))


def saerens_em(p, pi_src: float, tol=1e-8, max_iter=200):
    pi = float(pi_src)
    for _ in range(max_iter):
        w = adjust(p, pi, pi_src)
        new = float(w.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return adjust(p, pi, pi_src), pi


def bbse_prior(pred_val, y_val, pred_te):
    q1v, q0v = pred_val[y_val == 1], pred_val[y_val == 0]
    if len(q1v) == 0 or len(q0v) == 0:
        return float("nan")
    q1, q0 = float(q1v.mean()), float(q0v.mean())
    if abs(q1 - q0) < 1e-9:
        return float("nan")
    return float(np.clip((float(pred_te.mean()) - q0) / (q1 - q0), 0.0, 1.0))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-npz", type=int, default=0, help="Run only the first N npz files per setting (for smoke tests)")
    ap.add_argument("--settings", nargs="+", default=None)
    ap.add_argument("--unsw-s20", action="store_true",
                    help="Use the 20-seed database for unsw_pair (det_unsw_s20.db + scores_v2_unsw_s20);"
                         "validation-day scores are unavailable, so rules requiring them remain blank")
    ap.add_argument("--out-suffix", default="",
                    help="Output filename suffix, e.g. _unsw_s20; default empty = overwrite original files (legacy behavior)")
    args = ap.parse_args()

    if args.unsw_s20:
        for S in SETTINGS:
            if S["name"] == "unsw_pair":
                S["te"] = "scores_v2_unsw_s20/unsw"
                S["va"] = None
                S["db"] = "det_unsw_s20.db"
                print("unsw_pair → 20-seed database (no validation-day scores; NEED_VAL rules remain blank)")

    pri = pd.read_csv(os.path.join(R2, "wp4_prior.csv"))
    pri_key = {}
    for _, r in pri.iterrows():
        f = int(r.fold)
        pri_key[(r.dataset, r.cell, f)] = float(r.pos_rate_train)
        pri_key.setdefault((r.dataset, r.cell, -1), float(r.pos_rate_train))
    print(f"π_tr: {len(pri_key)} (dataset,cell,fold) combinations")

    zb = pd.read_csv(os.path.join(R2, "wp4_zeroday.csv"))
    zb_key = {(r.cell, r.model, int(r.seed)): (float(r.fpr_05), int(r.n_zero_eff))
              for _, r in zb.iterrows()}

    rules = [n for n, _fn, _need in ER.RULES if n != "oracle"]
    rules += ["saerens", "bbse", "saerens_cal"]
    print(f"Rule count {len(rules)}: {rules}")

    rows, gate = [], dict(checked=0, bad=[], miss_val={}, miss_te=0, drop_zero=0)

    for S in SETTINGS:
        if args.settings and S["name"] not in args.settings:
            continue
        te_dir = os.path.join(ROOT, S["te"])
        va_dir = os.path.join(ROOT, S["va"]) if S["va"] else None
        fs = sorted(glob.glob(os.path.join(te_dir, "*.npz")))
        if args.max_npz:
            fs = fs[:args.max_npz]
        con = sqlite3.connect(os.path.join(R2, S["db"]))
        db = {}
        for rid, f1_, fpr_, ptr in con.execute(
                "select run_id, macro_f1, fpr, test_pos_rate from runs"):
            parts = rid.split("|")

            fold_db = (int(parts[-3][1:])
                       if len(parts) >= 5 and re.fullmatch(r"f\d+", parts[-3]) else -1)
            db[(parts[1], parts[-2], int(parts[-1][1:]), fold_db)] = (f1_, fpr_, ptr)
        con.close()
        print(f"\n[{S['name']}] {len(fs)} npz files, {len(db)} database runs")

        for f in fs:
            m = PAT.match(os.path.basename(f))
            if not m:
                continue
            cell, model, seed = m.group("cell"), m.group("model"), int(m.group("seed"))
            fold = int(m.group("fold")[1:]) if m.group("fold") else -1
            slug = cell
            cell_db = slug.replace("_", "/")

            cand = [k for k in pri_key
                    if k[0] == S["pds"] and k[1].replace("/", "_") == slug]
            if not cand:
                continue
            cell_db = cand[0][1]
            f_key = fold if (S["pds"], cell_db, fold) in pri_key else -1
            pi_tr = pri_key[(S["pds"], cell_db, f_key)]

            d = np.load(f, allow_pickle=True)
            s = np.asarray(d["score"], float)
            y = np.asarray(d["label"]).astype(int)
            if S["name"] == "cic17nd":
                s_zd = np.asarray(d["score_zero"], float)
                y_zd = np.asarray(d["label_zero"]).astype(int)
            else:
                s_zd = y_zd = None

            sv = yv = None
            if va_dir:
                fv = os.path.join(va_dir, os.path.basename(f))
                if os.path.exists(fv):
                    dv = np.load(fv, allow_pickle=True)
                    if "score_val" in dv.files:
                        sv = np.asarray(dv["score_val"], float)
                        yv = np.asarray(dv["label_val"]).astype(int)
                        if yv.min() == yv.max():
                            sv = yv = None


            qp, qn = np.sort(s[y == 1]), np.sort(s[y == 0])
            npos, nneg = len(qp), len(qn)
            if npos < 2 or nneg < 2:
                continue
            v, w = ER._pooled(qp, qn, npos, nneg)
            has_val = sv is not None
            if has_val:
                qpv, qnv = np.sort(sv[yv == 1]), np.sort(sv[yv == 0])
                npos_v, nneg_v = len(qpv), len(qnv)
                thr_val = float(ER._best_thr(qpv, qnv, npos_v, nneg_v,
                                             ER._cands(qpv, qnv)))
            else:
                qpv = qnv = np.array([], dtype=float)
                npos_v = nneg_v = 0
                thr_val = float("nan")
            run = dict(cell=cell_db, model=model, seed=seed, fold=fold,
                       prior=pi_tr, qp=qp, qn=qn, npos=npos, nneg=nneg, v=v, w=w,
                       qpv=qpv, qnv=qnv, npos_v=npos_v, nneg_v=nneg_v,
                       thr_val=thr_val,
                       thr_oracle=float(ER._best_thr(qp, qn, npos, nneg,
                                                     ER._cands(qp, qn))))

            rec = dict(setting=S["name"], cell=cell_db, model=model, seed=seed,
                       fold=fold, pi_tr=pi_tr, n_test=npos + nneg)


            for name, fn, _need in ER.RULES:
                if name == "oracle":
                    continue
                if name in NEED_VAL and not has_val:
                    rec["excess_" + name] = np.nan
                    rec["fpr_" + name] = np.nan
                    gate["miss_val"][name] = gate["miss_val"].get(name, 0) + 1
                    continue
                t = float(fn(run))
                f1v, fprv = ER._f1_vec(qp, qn, npos, nneg, t)
                rec["excess_" + name] = float(f1v[0]) - ER.floor_of(npos / (npos + nneg))
                rec["fpr_" + name] = float(fprv[0])


            adj, pi_h = saerens_em(s, pi_tr)
            rec["excess_saerens"] = macro_f1(y, (adj >= THR).astype(int)) \
                - ER.floor_of(npos / (npos + nneg))
            rec["fpr_saerens"] = fpr_of(y, (adj >= THR).astype(int))

            if has_val:
                pi_b = bbse_prior((sv >= THR).astype(int), yv, (s >= THR).astype(int))
                if np.isfinite(pi_b):
                    ab = adjust(s, pi_b, pi_tr)
                    rec["excess_bbse"] = macro_f1(y, (ab >= THR).astype(int)) \
                        - ER.floor_of(npos / (npos + nneg))
                    rec["fpr_bbse"] = fpr_of(y, (ab >= THR).astype(int))
                else:
                    rec["excess_bbse"] = rec["fpr_bbse"] = np.nan

                pi_val = float(yv.mean())
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(sv, yv)
                s_cal = np.clip(iso.predict(s), 1e-6, 1 - 1e-6)
                adj_c, _ = saerens_em(s_cal, pi_val)
                rec["excess_saerens_cal"] = macro_f1(y, (adj_c >= THR).astype(int)) \
                    - ER.floor_of(npos / (npos + nneg))
                rec["fpr_saerens_cal"] = fpr_of(y, (adj_c >= THR).astype(int))
            else:
                for n in ("bbse", "saerens_cal"):
                    rec["excess_" + n] = np.nan
                    rec["fpr_" + n] = np.nan
                    gate["miss_val"][n] = gate["miss_val"].get(n, 0) + 1


            if S["name"] == "cic17nd":
                k = (cell_db, model, seed)
                if k in zb_key:
                    rec["zero_day_fpr"], rec["zero_day_n_seq"] = zb_key[k]
                else:
                    rec["zero_day_fpr"] = rec["zero_day_n_seq"] = np.nan
                    gate["drop_zero"] += 1
            else:
                rec["zero_day_fpr"] = rec["zero_day_n_seq"] = np.nan


            key = (cell_db, model, seed, fold)
            if key in db and np.isfinite(rec.get("excess_fixed0.5", np.nan)):
                f1_db, fpr_db, ptr_db = db[key]
                got_f1 = rec["excess_fixed0.5"] + ER.floor_of(npos / (npos + nneg))
                if abs(got_f1 - float(f1_db)) > 1e-9 or abs(
                        rec["fpr_fixed0.5"] - float(fpr_db)) > 1e-9:
                    gate["bad"].append((S["name"], cell_db, model, seed,
                                        got_f1, float(f1_db),
                                        rec["fpr_fixed0.5"], float(fpr_db)))
                gate["checked"] += 1

            rows.append(rec)

        print(f"  Cumulative rows: {len(rows)}")

    df = pd.DataFrame(rows)
    print(f"\nTotal rows {len(df)}; gate compared {gate['checked']} runs, {len(gate['bad'])} mismatches")
    if gate["bad"]:
        print("Gate failed; first 5 entries:")
        for b in gate["bad"][:5]:
            print("  ", b)
        print("→ No files written; fix the definitions first")
        return 1
    if gate["miss_val"]:
        print("Blank counts for rules requiring validation-day scores:", gate["miss_val"])

    cols = (["setting", "cell", "model", "seed", "fold", "pi_tr", "n_test"]
            + ["excess_" + r for r in rules] + ["fpr_" + r for r in rules]
            + ["zero_day_fpr", "zero_day_n_seq"])
    df = df[[c for c in cols if c in df.columns]]
    sfx = args.out_suffix
    p1 = os.path.join(R2, "wp4d_rules%s.csv" % sfx)
    df.to_csv(p1, index=False)
    print(f"→ {p1}（{df.shape[0]}×{df.shape[1]}）")


    agg = []
    for setting, g in df.groupby("setting"):
        base = g["excess_fixed0.5"].to_numpy(float)
        for r in rules:
            e = g["excess_" + r].to_numpy(float)
            ok = np.isfinite(e)
            if not ok.any():
                agg.append(dict(setting=setting, rule=r, n=0, collapse=np.nan,
                                ci_lo=np.nan, ci_hi=np.nan, excess_median=np.nan,
                                rescue=np.nan, harm=np.nan, fpr_median=np.nan,
                                zero_day_fpr_median=np.nan))
                continue
            c = (e[ok] < DELTA)
            bb = base[ok] < DELTA
            k = int(c.sum())
            lo, hi = wilson(k, int(ok.sum()))
            agg.append(dict(
                setting=setting, rule=r, n=int(ok.sum()), collapse=k / ok.sum(),
                ci_lo=lo, ci_hi=hi, excess_median=float(np.median(e[ok])),
                rescue=float((bb & ~c).sum() / max(bb.sum(), 1)),
                harm=float((~bb & c).sum() / max((~bb).sum(), 1)),
                fpr_median=float(np.nanmedian(g["fpr_" + r].to_numpy(float))),
                zero_day_fpr_median=float(np.nanmedian(
                    g["zero_day_fpr"].to_numpy(float))
                    if np.isfinite(g["zero_day_fpr"].to_numpy(float)).any() else np.nan)))
    ag = pd.DataFrame(agg)
    p2 = os.path.join(R2, "wp4d_summary%s.csv" % sfx)
    ag.to_csv(p2, index=False)
    print(f"→ {p2}")


    L = ["# WP4d — Rule table v2 (15 rules × 4 settings, two axes side by side)", "",
         "All thresholds are computed by the **original functions** in `scripts/eval_rules.py` (imported here, not reimplemented),"
         "with per-sequence scores as input. Gate: macro-F1 and FPR in the `0.5` column match the det database exactly;"
         f"**{gate['checked']}** runs compared, **0** mismatches.", "",
         "`zero_day_fpr` is populated only for settings with zero-attack days (`cic17nd`);"
         "the other three remain **blank rather than 0** (Q16: 0 would be read as “zero false positives”).", "",
         "## 1. Collapse rate by setting (excess < 0.02)", ""]
    for setting in df["setting"].unique():
        sg = ag[ag.setting == setting].sort_values("collapse")
        L += [f"### {setting} (n={int(sg['n'].max())} runs)", "",
              "| Rule | Collapse rate | Wilson 95% CI | Median excess | Rescued | Harmed | Median test-day FPR |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in sg.iterrows():
            if r.n == 0:
                L.append(f"| `{r.rule}` | — | — | — | — | — | — |")
                continue
            L.append(f"| `{r.rule}` | {100*r.collapse:.1f}% | "
                     f"[{100*r.ci_lo:.1f}, {100*r.ci_hi:.1f}] | "
                     f"{r.excess_median:+.4f} | {100*r.rescue:.1f}% | "
                     f"{100*r.harm:.1f}% | {r.fpr_median:.4f} |")
        L.append("")
    mdl = "WP4D_SUMMARY%s.md" % sfx
    open(os.path.join(R2, mdl), "w", encoding="utf-8").write("\n".join(L))
    print("→ %s" % mdl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
