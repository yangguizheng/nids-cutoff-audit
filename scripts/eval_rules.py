#!/usr/bin/env python


from __future__ import annotations

import json
import math
import os

import numpy as np

MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
SHORT = {"lstm": "LSTM", "tcn": "TCN", "transformer": "TF", "rf": "RF", "hgb": "HGB"}
EPS = 1e-7
NCAND = 3000
DIP_MAX = 0.60
SEP_MIN = 2.0


def floor_of(p):
    return max((1 - p) / (2 - p), p / (1 + p))


def _cands(qp, qn, n=NCAND):
    v = np.unique(np.concatenate([qp, qn]).astype(np.float64))
    if len(v) > n:
        v = np.unique(np.quantile(v, np.linspace(0, 1, n)))
    return np.unique(np.concatenate([v, [0.5]]))


def _f1_vec(qp_s, qn_s, npos, nneg, t):

    t = np.atleast_1d(np.asarray(t, dtype=np.float64))

    tpr = (len(qp_s) - np.searchsorted(qp_s, t, side="left")) / len(qp_s)
    fpr = (len(qn_s) - np.searchsorted(qn_s, t, side="left")) / len(qn_s)
    tp = tpr * npos
    fp = fpr * nneg
    fn = npos - tp
    tn = nneg - fp
    d1 = 2 * tp + fp + fn
    d2 = 2 * tn + fn + fp
    f1p = np.divide(2 * tp, d1, out=np.zeros_like(tp), where=d1 > 0)
    f1n = np.divide(2 * tn, d2, out=np.zeros_like(tn), where=d2 > 0)
    return 0.5 * (f1p + f1n), np.divide(fp, nneg, out=np.full_like(fp, np.nan),
                                        where=nneg > 0)


def _best_thr(qp_s, qn_s, npos, nneg, cands):
    f1, _ = _f1_vec(qp_s, qn_s, npos, nneg, cands)
    return float(cands[int(np.argmax(f1))])


def _pooled(qp, qn, npos, nneg):
    v = np.concatenate([qp, qn]).astype(np.float64)
    w = np.concatenate([np.full(len(qp), npos / max(len(qp), 1)),
                        np.full(len(qn), nneg / max(len(qn), 1))])
    return v, w


def _wquant(v, w, level):

    o = np.argsort(v)
    c = np.cumsum(np.asarray(w)[o]) / np.sum(w)
    k = int(np.searchsorted(c, level))
    return float(np.asarray(v)[o][min(k, len(v) - 1)])


def _logit(v):
    p = np.clip(np.asarray(v, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sig(x):
    return float(1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0))))


def _em1d(x, wt, k=2, iters=200, tol=1e-9, init=None):

    o = np.argsort(x)
    x, wt = x[o], wt[o]
    if init is None:
        c = np.cumsum(wt) / wt.sum()
        mu = np.array([np.interp((i + 0.5) / k, c, x) for i in range(k)], dtype=float)
        sd = np.full(k, max(float(x.std()), 1e-3))
        pi = np.full(k, 1.0 / k)
    else:
        mu, sd, pi = (np.array(z, dtype=float) for z in init)
    ll_old, ll = -np.inf, -np.inf
    for _ in range(iters):
        z = (x[:, None] - mu[None, :]) / sd[None, :]
        r = pi[None, :] * np.exp(-0.5 * z * z) / np.maximum(sd[None, :], 1e-12)
        s = r.sum(1, keepdims=True)
        r = r / np.maximum(s, 1e-300)
        rw = r * wt[:, None]
        nk = rw.sum(0) + 1e-12
        pi = nk / nk.sum()
        mu = (rw * x[:, None]).sum(0) / nk
        sd = np.maximum(np.sqrt((rw * (x[:, None] - mu[None, :]) ** 2).sum(0) / nk), 1e-6)
        ll = float((wt * np.log(np.maximum(s[:, 0], 1e-300))).sum())
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            break
        ll_old = ll
    return mu, sd, pi, ll


def _fit2(x, wt, nstart=9, iters=150):


    o = np.argsort(x)
    xs, ws = x[o], wt[o]
    cw = np.cumsum(ws) / ws.sum()
    best = None
    for q in np.linspace(0.30, 0.985, nstart):
        k = int(np.searchsorted(cw, q))
        k = min(max(k, 1), len(xs) - 1)
        w0, w1 = ws[:k].sum(), ws[k:].sum()
        if w0 <= 0 or w1 <= 0:
            continue
        m0 = float((ws[:k] * xs[:k]).sum() / w0)
        m1 = float((ws[k:] * xs[k:]).sum() / w1)
        s0 = max(float(np.sqrt((ws[:k] * (xs[:k] - m0) ** 2).sum() / w0)), 1e-3)
        s1 = max(float(np.sqrt((ws[k:] * (xs[k:] - m1) ** 2).sum() / w1)), 1e-3)
        out = _em1d(x, wt, 2, iters=iters,
                    init=([m0, m1], [s0, s1], [w0, w1] / (w0 + w1)))
        if best is None or out[3] > best[3]:
            best = out
    return best


def _gmm_cut(mu, sd, pi):

    i, j = np.argsort(mu)
    m1, s1, p1 = mu[i], sd[i], pi[i]
    m2, s2, p2 = mu[j], sd[j], pi[j]
    A = 1 / (2 * s2 ** 2) - 1 / (2 * s1 ** 2)
    B = m1 / s1 ** 2 - m2 / s2 ** 2
    C = (m2 ** 2 / (2 * s2 ** 2) - m1 ** 2 / (2 * s1 ** 2)
         + math.log(max(p1 * s2, 1e-300) / max(p2 * s1, 1e-300)))
    roots = []
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            roots = [-C / B]
    else:
        disc = B * B - 4 * A * C
        if disc >= 0:
            sq = math.sqrt(disc)
            roots = [(-B - sq) / (2 * A), (-B + sq) / (2 * A)]
    inb = [r for r in roots if m1 <= r <= m2]
    if inb:
        return float(inb[0])
    if roots:
        mid = 0.5 * (m1 + m2)
        return float(min(roots, key=lambda r: abs(r - mid)))
    return float(0.5 * (m1 + m2))


def _dens(mu, sd, pi, t):
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    z = (t[:, None] - mu[None, :]) / sd[None, :]
    return (pi[None, :] * np.exp(-0.5 * z * z)
            / (sd[None, :] * math.sqrt(2 * math.pi))).sum(1)


def _fit2_cached(run):
    if "fit2" not in run:
        run["fit2"] = _fit2(_logit(run["v"]), run["w"])
    return run["fit2"]


def _valley(run):

    mu, sd, pi, _ = _fit2_cached(run)
    cut = _gmm_cut(mu, sd, pi)
    dcut = float(_dens(mu, sd, pi, [cut])[0])
    dmin = float(_dens(mu, sd, pi, mu).min())
    dip = dcut / dmin if dmin > 0 else 1.0
    sep = abs(float(mu.max() - mu.min())) / max(float(sd.mean()), 1e-9)
    return cut, dip, sep


def thr_fixed(run, **kw):
    return 0.5


def _otsu(x, wt):
    o = np.argsort(x)
    x, w = np.asarray(x, dtype=np.float64)[o], np.asarray(wt)[o]
    cw = np.cumsum(w)
    cx = np.cumsum(w * x)
    tw, tx = cw[-1], cx[-1]
    m0 = cx[:-1] / np.maximum(cw[:-1], 1e-12)
    m1 = (tx - cx[:-1]) / np.maximum(tw - cw[:-1], 1e-12)
    var = cw[:-1] * (tw - cw[:-1]) * (m0 - m1) ** 2
    k = int(np.argmax(var))
    return float((x[k] + x[k + 1]) / 2.0)


def thr_otsu(run, **kw):
    return _otsu(run["v"], run["w"])


def thr_otsu_logit(run, **kw):
    return _sig(_otsu(_logit(run["v"]), run["w"]))


def thr_qtrain(run, **kw):
    return _wquant(run["v"], run["w"], 1.0 - run["prior"])


def thr_qtrain_gt(run, **kw):

    t = thr_qtrain(run)
    return float(np.nextafter(t, np.inf))


def _val_run(run):

    v, w = _pooled(run["qpv"], run["qnv"], run["npos_v"], run["nneg_v"])
    return dict(v=v, w=w, qp=run["qpv"], qn=run["qnv"],
                npos=run["npos_v"], nneg=run["nneg_v"])


def thr_scale_val(run, **kw):


    vr = _val_run(run)
    scores = {}
    for name, fn in (("lin", thr_otsu), ("log", thr_otsu_logit)):
        f1, _ = _f1_vec(vr["qp"], vr["qn"], vr["npos"], vr["nneg"], fn(vr))
        scores[name] = float(f1[0])
    return thr_otsu(run) if scores["lin"] >= scores["log"] else thr_otsu_logit(run)


def thr_hybrid(run, cut=0.01, **kw):


    tl = thr_otsu_logit(run)
    return tl if tl < cut else thr_otsu(run)


def thr_hybrid50(run, **kw):
    return thr_hybrid(run, cut=0.05)


def thr_val(run, **kw):
    return run["thr_val"]


def thr_val_rank(run, **kw):

    t = run["thr_val"]
    tvp = (len(run["qpv"]) - np.searchsorted(run["qpv"], t, side="left")) / len(run["qpv"])
    tvn = (len(run["qnv"]) - np.searchsorted(run["qnv"], t, side="left")) / len(run["qnv"])
    nv = run["npos_v"] + run["nneg_v"]
    F = (tvp * run["npos_v"] + tvn * run["nneg_v"]) / max(nv, 1)
    return _wquant(run["v"], run["w"], 1.0 - F)


def thr_gmm(run, **kw):
    return _sig(_valley(run)[0])


def thr_gmm_valley(run, **kw):
    cut, dip, sep = _valley(run)
    if dip < DIP_MAX and sep > SEP_MIN:
        return _sig(cut)
    return 0.5


def thr_oracle(run, **kw):
    return run["thr_oracle"]


RULES = [
    ("fixed0.5", thr_fixed, "None"),
    ("val_thr", thr_val, "Previous-day labels"),
    ("val_rank", thr_val_rank, "Previous-day labels (quantile)"),
    ("qtrain", thr_qtrain, "Historical attack rate"),
    ("qtrain_gt", thr_qtrain_gt, "Historical attack rate + tie correction"),
    ("otsu", thr_otsu, "Same-day scores (linear)"),
    ("otsu_logit", thr_otsu_logit, "Same-day scores (log-odds)"),
    ("scale_val", thr_scale_val, "Previous-day labels (scale selection)"),
    ("hybrid", thr_hybrid, "Same-day scores (scale selection, 0.01)"),
    ("hybrid50", thr_hybrid50, "Same-day scores (scale selection, 0.05)"),
    ("gmm_logit", thr_gmm, "Same-day scores (two clusters)"),
    ("gmm_valley", thr_gmm_valley, "Same-day scores (two clusters + valley)"),
    ("oracle", thr_oracle, "Same-day labels (upper bound)"),
]


CANDIDATES = {
    "cic18": ["results/scores_cic18.npz"],
    "unsw_pair": ["results/scores_unsw_pair.npz"],
    "unsw_nd": ["results/scores_unsw_nd_full.npz", "results/scores_unsw_nd.npz"],
}


def prepare(setting):
    f = next((x for x in CANDIDATES.get(setting, [f"results/scores_{setting}.npz"])
              if os.path.exists(x)), None)
    if f is None:
        return None, -1
    print(f"  [data] {setting} ← {f}")
    d = np.load(f, allow_pickle=True)
    raw = json.loads(str(d["recs"]))
    out = []
    for r in raw:
        qp = np.sort(np.asarray(r["test_q_pos"], dtype=np.float64))
        qn = np.sort(np.asarray(r["test_q_neg"], dtype=np.float64))
        npos, nneg = r["test_n_pos"], r["test_n_neg"]
        v, w = _pooled(qp, qn, npos, nneg)
        qpv = np.sort(np.asarray(r["q_pos"], dtype=np.float64))
        qnv = np.sort(np.asarray(r["q_neg"], dtype=np.float64))
        out.append(dict(
            cell=r["cell"], model=r["model"], seed=r["seed"],
            fold=r.get("fold", ""), prior=r["prior_train"],
            qp=qp, qn=qn, npos=npos, nneg=nneg, v=v, w=w,
            qpv=qpv, qnv=qnv, npos_v=r["n_pos"], nneg_v=r["n_neg"],
            thr_val=float(_best_thr(qpv, qnv, r["n_pos"], r["n_neg"],
                                    _cands(qpv, qnv))),
            thr_oracle=float(_best_thr(qp, qn, npos, nneg, _cands(qp, qn))),
            qn_norm=(np.asarray(r["q_norm"], dtype=np.float64)
                     if "q_norm" in r else None)))
    return out, int(d["norm_day"])


def eval_ranking(runs, setting):
    print(f"\n{'='*106}\nRanking axis  {setting}  n={len(runs)} runs\n{'='*106}")
    print(f"{'Rule':15s}{'Information required':24s}{'Collapse rate':>9s}{'Rescued':>8s}{'Harmed':>8s}"
          f"{'Median excess':>11s}{'Median threshold':>10s}")
    ex, th = {}, {}
    for name, fn, need in RULES:
        vals, ts = [], []
        for r in runs:
            t = fn(r)
            ts.append(t)
            f1, _ = _f1_vec(r["qp"], r["qn"], r["npos"], r["nneg"], t)
            pos = r["npos"] / (r["npos"] + r["nneg"])
            vals.append(float(f1[0]) - floor_of(pos))
        ex[name] = np.array(vals)
        th[name] = np.array(ts)

    bc = ex["fixed0.5"] < 0.02
    for name, fn, need in RULES:
        e = ex[name]
        c = e < 0.02
        resc = float((bc & ~c).sum() / max(bc.sum(), 1))
        harm = float((~bc & c).sum() / max((~bc).sum(), 1))
        print(f"{name:15s}{need:24s}{100*c.mean():8.1f}%{100*resc:7.1f}%"
              f"{100*harm:7.1f}%{np.median(e):+11.4f}{np.median(th[name]):10.4f}")
    return ex


def eval_operational(runs, setting, norm_day):
    rr = [r for r in runs if r["qn_norm"] is not None]
    if not rr:
        return {}
    print(f"\n{'='*106}\nOperational axis  attack-free-day FPR ({setting}, day{norm_day}, n={len(rr)} runs)\n{'='*106}")
    print(f"{'Rule':15s}{'Median FPR':>10s}{'Mean':>9s}{'P90':>9s}{'Max':>9s}"
          f"{'Runs >5%':>9s}{'Runs >50%':>9s}{'Median threshold':>10s}")
    store = {}
    for name, fn, need in RULES:
        if name == "oracle":
            continue
        fpr, thr = [], []
        for r in rr:
            t = fn(r)
            thr.append(t)
            fpr.append(float(np.mean(r["qn_norm"] >= t)))
        fpr = np.array(fpr)
        store[name] = (fpr, np.array(thr))
        print(f"{name:15s}{np.median(fpr):10.4f}{fpr.mean():9.4f}"
              f"{np.quantile(fpr, .9):9.4f}{fpr.max():9.4f}"
              f"{100*(fpr > 0.05).mean():8.1f}%{100*(fpr > 0.5).mean():8.1f}%"
              f"{np.median(thr):10.4f}")

    print("\n  Median FPR · by segment")
    routes = ["tw30s", "service", "host"]
    print(f"{'Rule':15s}" + "".join(f"{x:>10s}" for x in routes))
    for name in store:
        row = f"{name:15s}"
        for rt in routes:
            sub = [i for i, r in enumerate(rr) if r["cell"].split("/")[0] == rt]
            row += f"{np.median(store[name][0][sub]) if sub else float('nan'):10.4f}"
        print(row)

    print("\n  Median FPR · by model")
    print(f"{'Rule':15s}" + "".join(f"{SHORT[m]:>10s}" for m in MODELS))
    for name in store:
        row = f"{name:15s}"
        for m in MODELS:
            sub = [i for i, r in enumerate(rr) if r["model"] == m]
            row += f"{np.median(store[name][0][sub]) if sub else float('nan'):10.4f}"
        print(row)
    return store


if __name__ == "__main__":
    import csv
    import sys

    rows = []
    for s in ["cic18", "unsw_pair"]:
        runs, _ = prepare(s)
        if runs:
            ex = eval_ranking(runs, s)
            for i, r in enumerate(runs):
                row = dict(setting=s, cell=r["cell"], model=r["model"],
                           seed=r["seed"], fold=r["fold"])
                for k, v in ex.items():
                    row[f"excess_{k}"] = float(v[i])
                rows.append(row)
    runs, nd = prepare("unsw_nd")
    if runs:
        ex = eval_ranking(runs, "unsw_nd (service-axis fold)")
        fpr = eval_operational(runs, "unsw_nd", nd)
        for i, r in enumerate(runs):
            row = dict(setting="unsw_nd", cell=r["cell"], model=r["model"],
                       seed=r["seed"], fold=r["fold"])
            for k, v in ex.items():
                row[f"excess_{k}"] = float(v[i])
            for k, (f, _t) in fpr.items():
                row[f"fpr_{k}"] = float(f[i])
            rows.append(row)

    if rows:
        cols = list(rows[0].keys())
        for row in rows:
            for c in row:
                if c not in cols:
                    cols.append(c)
        with open("results/rules_per_run.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-run details → results/rules_per_run.csv  ({len(rows)} rows)")
    sys.exit(0)
