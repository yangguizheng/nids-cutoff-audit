#!/usr/bin/env python


from __future__ import annotations

import argparse
import itertools
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20261101
NBOOT = 10000
MODELS = ["lstm", "tcn", "transformer", "rf", "hgb"]
EPS_BY_METRIC = {"excess": [0.0], "aplift": [0.0], "roc_auc": [0.0, 0.001, 0.005]}
SUBSETS = ["all", "tw30s", "host+service"]
TOST_MARGIN = 0.005
TOST_ALPHA = 0.05
SMALL_EPS = 0.001

DATASETS = [

    ("CIC17", "results/det_cic17.db", False),
    ("CIC18", "results/det_cic18.db", False),
    ("UNSW", "results/det_unsw.db", True),
]


def floor_of(p: float) -> float:

    return max((1.0 - p) / (2.0 - p), p / (1.0 + p))


def load_all() -> pd.DataFrame:
    out = []
    for label, path, has_folds in DATASETS:
        con = sqlite3.connect(path)
        df = pd.read_sql_query("SELECT * FROM runs", con)
        con.close()
        if has_folds:
            df["fold"] = df["run_id"].str.extract(r"\|(f\d)\|")[0]
            if df["fold"].isna().any():
                raise SystemExit(f"{path}: some run_id values have no parseable fold")
        else:
            df["fold"] = "-"
        df["dataset"] = label
        df["ds_key"] = np.where(df["fold"] == "-", label, label + ":" + df["fold"])
        out.append(df)
    d = pd.concat(out, ignore_index=True)

    d["excess"] = d["macro_f1"] - d["test_pos_rate"].map(floor_of)
    d["aplift"] = d["pr_auc"] - d["test_pos_rate"]
    d["route_subset"] = np.where(
        d["route"] == "time_window",
        "tw30s",
        np.where(d["route"].isin(["host", "service"]), "host+service", "other"),
    )
    return d


def subset_df(d: pd.DataFrame, ds_key: str, sub: str) -> pd.DataFrame:
    x = d[d["ds_key"] == ds_key]
    if sub == "tw30s":
        x = x[x["route_subset"] == "tw30s"]
    elif sub == "host+service":
        x = x[x["route_subset"] == "host+service"]
    return x


def group_matrix(d: pd.DataFrame, ds_key: str, sub: str, metric: str):

    x = subset_df(d, ds_key, sub)
    piv = x.pivot_table(index=["cell", "seed"], columns="model",
                        values=metric, aggfunc="mean")
    for m in MODELS:
        if m not in piv.columns:
            piv[m] = np.nan
    piv = piv[MODELS]
    piv = piv.sort_index()
    n_bad = int(piv.isna().any(axis=1).sum())
    piv = piv.dropna(how="any")
    return piv.values.astype(float), piv.index, n_bad


def win_shares(M: np.ndarray, eps: float) -> np.ndarray:

    mx = M.max(axis=1, keepdims=True)
    win = (M >= mx - eps - 1e-12).astype(float)
    return win / win.sum(axis=1, keepdims=True)


def boot_idx(n: int, nb: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(nb, n))


def pct_ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def wp1a(d: pd.DataFrame, outdir: str) -> pd.DataFrame:
    rows = []
    for label, _, has_folds in DATASETS:
        dsk = f"{label}:f0" if has_folds else label
        keys = [dsk] if not has_folds else [f"{label}:f0", f"{label}:f1"]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            for sub in SUBSETS:
                for metric, eps_list in EPS_BY_METRIC.items():
                    M, idx, n_bad = group_matrix(d, ds_key, sub, metric)
                    n = len(M)
                    if n == 0:
                        continue
                    bi = boot_idx(n, NBOOT, SEED)
                    for eps in eps_list:
                        S = win_shares(M, eps)
                        wr = S.mean(axis=0)
                        boot = S[bi].mean(axis=1)
                        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
                        for j, m in enumerate(MODELS):
                            rows.append(dict(
                                dataset=label, fold=fold, route_subset=sub,
                                metric=metric, eps=eps, model=m,
                                winrate=round(float(wr[j]), 6),
                                ci_lo=round(float(lo[j]), 6),
                                ci_hi=round(float(hi[j]), 6),
                                n_groups=int(n), n_groups_dropped_na=n_bad,
                            ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "wp1_winrate.csv"), index=False)
    return df


def exact_perm_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:

    tau_obs = float(stats.kendalltau(a, b).statistic)
    cnt = 0
    tot = 0
    for perm in itertools.permutations(range(len(a))):
        tot += 1
        t = float(stats.kendalltau(np.asarray(a)[list(perm)], b).statistic)
        if abs(t) >= abs(tau_obs) - 1e-12:
            cnt += 1
    return tau_obs, cnt / tot


def wp1b(d: pd.DataFrame, outdir: str) -> pd.DataFrame:
    rows = []
    for label, _, has_folds in DATASETS:
        keys = [f"{label}:f0", f"{label}:f1"] if has_folds else [label]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            for sub in SUBSETS:

                agg = {}
                for metric in ["excess", "roc_auc", "aplift"]:
                    M, idx, _ = group_matrix(d, ds_key, sub, metric)
                    if len(M) == 0:
                        continue
                    agg[(metric, "winrate")] = win_shares(M, 0.0).mean(axis=0)
                    agg[(metric, "mean")] = M.mean(axis=0)
                for other in ["roc_auc", "aplift"]:
                    for how in ["winrate", "mean"]:
                        ka, kb = ("excess", how), (other, how)
                        if ka not in agg or kb not in agg:
                            continue
                        a, b = agg[ka], agg[kb]
                        tau, p = exact_perm_p(a, b)


                        M_e, _, _ = group_matrix(d, ds_key, sub, "excess")
                        M_o, _, _ = group_matrix(d, ds_key, sub, other)
                        n = len(M_e)
                        bi = boot_idx(n, NBOOT, SEED)
                        Se = win_shares(M_e, 0.0)
                        So = win_shares(M_o, 0.0)
                        neg = 0
                        taus = []
                        for b_ in bi:
                            if how == "winrate":
                                xa, xb = Se[b_].mean(axis=0), So[b_].mean(axis=0)
                            else:
                                xa, xb = M_e[b_].mean(axis=0), M_o[b_].mean(axis=0)
                            t = float(stats.kendalltau(xa, xb).statistic)
                            taus.append(t)
                            if t < 0:
                                neg += 1
                        rows.append(dict(
                            dataset=label, fold=fold, route_subset=sub,
                            pair=f"excess~{other}", aggregation=how,
                            tau=round(tau, 6),
                            exact_perm_p=round(p, 6),
                            boot_frac_tau_lt0=round(neg / NBOOT, 6),
                            boot_tau_ci_lo=round(float(np.percentile(taus, 2.5)), 6),
                            boot_tau_ci_hi=round(float(np.percentile(taus, 97.5)), 6),
                            n_groups=int(n),
                        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "wp1_tau.csv"), index=False)
    return df


def hodges_lehmann(x: np.ndarray) -> float:
    n = len(x)
    w = (x[:, None] + x[None, :]) / 2.0
    iu = np.triu_indices(n)
    return float(np.median(w[iu]))


def hl_boot(delta: np.ndarray, bi: np.ndarray, chunk: int = 250) -> np.ndarray:

    n = len(delta)
    out = np.empty(len(bi))
    for s in range(0, len(bi), chunk):
        idx = bi[s:s + chunk]
        m = len(idx)
        D = delta[idx]

        W = (D[:, :, None] + D[:, None, :]) / 2.0
        tri = np.triu(np.ones((n, n), dtype=bool))
        out[s:s + m] = np.median(W[:, tri], axis=1)
    return out


def tost(x: np.ndarray, margin: float = TOST_MARGIN) -> tuple[float, float, float]:
    n = len(x)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    if se == 0:
        return m, float("nan"), 0.0
    t1 = (m + margin) / se
    p1 = 1.0 - stats.t.cdf(t1, n - 1)
    t2 = (m - margin) / se
    p2 = stats.t.cdf(t2, n - 1)
    return m, max(p1, p2), se


def wp1c(d: pd.DataFrame, outdir: str) -> pd.DataFrame:
    rows = []
    for label, _, has_folds in DATASETS:
        keys = [f"{label}:f0", f"{label}:f1"] if has_folds else [label]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            for sub in SUBSETS:
                M, idx, _ = group_matrix(d, ds_key, sub, "roc_auc")
                n = len(M)
                if n < 3:
                    continue
                bi = boot_idx(n, NBOOT, SEED)
                for ia, ib in itertools.combinations(range(len(MODELS)), 2):
                    delta = M[:, ia] - M[:, ib]
                    med = float(np.median(delta))
                    hl = hodges_lehmann(delta)
                    bhl = hl_boot(delta, bi)
                    lo, hi = pct_ci(bhl)
                    mean, p_tost, _ = tost(delta)
                    rows.append(dict(
                        dataset=label, fold=fold, route_subset=sub,
                        model_a=MODELS[ia], model_b=MODELS[ib],
                        n_groups=int(n),
                        median_delta=round(med, 6),
                        mean_delta=round(mean, 6),
                        hl=round(hl, 6),
                        hl_ci_lo=round(lo, 6), hl_ci_hi=round(hi, 6),
                        frac_abs_lt_0p001=round(float(np.mean(np.abs(delta) < 1e-3)), 6),
                        frac_abs_lt_0p0001=round(float(np.mean(np.abs(delta) < 1e-4)), 6),
                        tost_p=(None if not np.isfinite(p_tost) else round(p_tost, 6)),
                        tost_margin=TOST_MARGIN,
                        equivalent=bool(np.isfinite(p_tost) and p_tost < TOST_ALPHA),
                        hl_equiv_ci=bool(lo > -TOST_MARGIN and hi < TOST_MARGIN),
                    ))
                    print(f"    {label}{fold} {sub} {MODELS[ia]}-{MODELS[ib]} "
                          f"hl={hl:.4f} [{lo:.4f},{hi:.4f}] ", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "wp1_pairs.csv"), index=False)
    return df


def marginal_swing(A: np.ndarray, routes: np.ndarray, pads: np.ndarray) -> dict:

    cell_mean = A.mean(axis=1)
    out = {}

    lv = sorted(set(routes.tolist()))
    mm = [cell_mean[routes == r].mean() for r in lv]
    out["segment"] = (float(max(mm) - min(mm)) if len(lv) > 1 else float("nan"),
                      {r: float(v) for r, v in zip(lv, mm)})

    lv = sorted(set(pads.tolist()))
    mm = [cell_mean[pads == r].mean() for r in lv]
    out["padding"] = (float(max(mm) - min(mm)) if len(lv) > 1 else float("nan"),
                      {r: float(v) for r, v in zip(lv, mm)})

    mm = cell_mean.mean(axis=0)
    out["model"] = (float(mm.max() - mm.min()),
                    {m: float(v) for m, v in zip(MODELS, mm)})
    return out


def build_cube(d: pd.DataFrame, ds_key: str, sub: str, metric: str):
    x = subset_df(d, ds_key, sub)
    cells = sorted(x["cell"].unique())
    seeds = sorted(x["seed"].unique())
    A = np.full((len(cells), len(seeds), len(MODELS)), np.nan)
    for i, c in enumerate(cells):
        for j, s in enumerate(seeds):
            y = x[(x["cell"] == c) & (x["seed"] == s)]
            for k, m in enumerate(MODELS):
                v = y.loc[y["model"] == m, metric]
                if len(v):
                    A[i, j, k] = float(v.iloc[0])
    ok_rows = ~np.isnan(A).any(axis=(1, 2))
    A = A[ok_rows]
    routes = np.array([c.split("/")[0] for c in cells])[ok_rows]
    pads = np.array([c.split("/")[-2] for c in cells])[ok_rows]
    return A, routes, pads, len(seeds)


def boot_swings(A: np.ndarray, routes: np.ndarray, pads: np.ndarray,
                nb: int, seed: int) -> dict:

    n_cells, n_seeds, n_models = A.shape
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n_seeds, np.ones(n_seeds) / n_seeds, size=(nb, n_cells))

    E = np.einsum("bcs,csm->bcm", W / n_seeds, A)
    res = {f: np.empty(nb) for f in ["segment", "padding", "model"]}
    ur = sorted(set(routes.tolist()))
    up = sorted(set(pads.tolist()))
    for b in range(nb):
        Eb = E[b]
        seg = [Eb[routes == r].mean() for r in ur]
        pad = [Eb[pads == p].mean() for p in up]
        mod = Eb.mean(axis=0)
        res["segment"][b] = (max(seg) - min(seg)) if len(ur) > 1 else np.nan
        res["padding"][b] = (max(pad) - min(pad)) if len(up) > 1 else np.nan
        res["model"][b] = mod.max() - mod.min()
    return res, E


def wp1d(d: pd.DataFrame, outdir: str) -> pd.DataFrame:
    rows = []
    ratios = []
    for label, _, has_folds in DATASETS:
        keys = [f"{label}:f0", f"{label}:f1"] if has_folds else [label]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            for sub in SUBSETS:
                per_metric = {}
                for metric in ["excess", "roc_auc", "aplift"]:
                    A, routes, pads, n_seeds = build_cube(d, ds_key, sub, metric)
                    if A.size == 0:
                        continue
                    pt = marginal_swing(A, routes, pads)
                    bs, _ = boot_swings(A, routes, pads, NBOOT, SEED)
                    per_metric[metric] = (pt, bs)
                    for fac in ["segment", "padding", "model"]:
                        v, lvl = pt[fac]
                        lo, hi = pct_ci(bs[fac])
                        rows.append(dict(
                            dataset=label, fold=fold, route_subset=sub,
                            metric=metric, factor=fac,
                            swing=round(v, 6),
                            ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                            n_cells=int(A.shape[0]), n_seeds=int(n_seeds),
                            level_means=";".join(f"{k}={vv:.6f}" for k, vv in lvl.items()),
                        ))

                for fac in ["segment", "padding", "model"]:
                    if "excess" in per_metric and "roc_auc" in per_metric:
                        ve = per_metric["excess"][0][fac][0]
                        va = per_metric["roc_auc"][0][fac][0]
                        be = per_metric["excess"][1][fac]
                        ba = per_metric["roc_auc"][1][fac]
                        if not (np.isfinite(ve) and np.isfinite(va) and va > 0):
                            continue
                        rb = be / np.maximum(ba, 1e-12)
                        lo, hi = pct_ci(rb)
                        ratios.append(dict(
                            dataset=label, fold=fold, route_subset=sub,
                            factor=fac,
                            swing_excess=round(ve, 6), swing_roc_auc=round(va, 6),
                            ratio=round(ve / va, 4),
                            ratio_ci_lo=round(lo, 4), ratio_ci_hi=round(hi, 4),
                        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "wp1_swings.csv"), index=False)
    dr = pd.DataFrame(ratios)
    dr.to_csv(os.path.join(outdir, "wp1_swing_ratios.csv"), index=False)
    return df, dr


def eta2_anova(d: pd.DataFrame, outdir: str) -> pd.DataFrame:
    rows = []
    for label, _, has_folds in DATASETS:
        keys = [f"{label}:f0", f"{label}:f1"] if has_folds else [label]
        for ds_key in keys:
            fold = ds_key.split(":")[1] if ":" in ds_key else "-"
            x = d[d["ds_key"] == ds_key]
            for metric in ["excess", "roc_auc"]:
                y = x[metric].values.astype(float)
                grand = y.mean()
                sst = float(((y - grand) ** 2).sum())
                facs = {"segment": x["route"].values,
                        "padding": x["padding"].values,
                        "model": x["model"].values}

                ss = {}
                for name, v in facs.items():
                    g = pd.DataFrame({"v": v, "y": y}).groupby("v")["y"]
                    ss[name] = float((g.mean().sub(grand) ** 2 * g.size()).sum())

                for a, b in itertools.combinations(facs, 2):
                    t = pd.DataFrame({"a": facs[a], "b": facs[b], "y": y}).groupby(["a", "b"])["y"]
                    cellm = t.mean()

                    ma = pd.DataFrame({"a": facs[a], "y": y}).groupby("a")["y"].mean()
                    mb = pd.DataFrame({"b": facs[b], "y": y}).groupby("b")["y"].mean()
                    acc = 0.0
                    for (ia, ib), mv in cellm.items():
                        acc += (mv - ma[ia] - mb[ib] + grand) ** 2
                    ncell = len(y) / len(cellm)
                    ss[f"{a}x{b}"] = float(acc * ncell)

                t3 = pd.DataFrame({"a": facs["segment"], "b": facs["padding"],
                                   "c": facs["model"], "y": y}).groupby(["a", "b", "c"])["y"]
                c3 = t3.mean()
                ma = pd.DataFrame({"a": facs["segment"], "y": y}).groupby("a")["y"].mean()
                mb = pd.DataFrame({"b": facs["padding"], "y": y}).groupby("b")["y"].mean()
                mc = pd.DataFrame({"c": facs["model"], "y": y}).groupby("c")["y"].mean()
                mab = pd.DataFrame({"a": facs["segment"], "b": facs["padding"], "y": y}) \
                    .groupby(["a", "b"])["y"].mean()
                mac = pd.DataFrame({"a": facs["segment"], "c": facs["model"], "y": y}) \
                    .groupby(["a", "c"])["y"].mean()
                mbc = pd.DataFrame({"b": facs["padding"], "c": facs["model"], "y": y}) \
                    .groupby(["b", "c"])["y"].mean()
                acc = 0.0
                for (ia, ib, ic), mv in c3.items():
                    acc += (mv - mab[(ia, ib)] - mac[(ia, ic)] - mbc[(ib, ic)]
                            + ma[ia] + mb[ib] + mc[ic] - grand) ** 2
                ncell = len(y) / len(c3)
                ss["segmentxpaddingxmodel"] = float(acc * ncell)
                sse = sst - sum(ss.values())
                dfres = len(y) - len(c3)
                mse = sse / dfres if dfres > 0 else float("nan")
                DFN = {"segment": len(ma) - 1, "padding": len(mb) - 1,
                       "model": len(mc) - 1,
                       "segmentxpadding": (len(ma) - 1) * (len(mb) - 1),
                       "segmentxmodel": (len(ma) - 1) * (len(mc) - 1),
                       "paddingxmodel": (len(mb) - 1) * (len(mc) - 1),
                       "segmentxpaddingxmodel": (len(ma) - 1) * (len(mb) - 1) * (len(mc) - 1)}
                for name, v in ss.items():
                    dfn = DFN[name]
                    F = (v / dfn) / mse if mse > 0 else float("nan")
                    p = float(1 - stats.f.cdf(F, dfn, dfres)) if np.isfinite(F) else float("nan")
                    rows.append(dict(
                        dataset=label, fold=fold, metric=metric, term=name,
                        ss=round(v, 8), df=int(dfn),
                        eta2=round(v / sst, 6) if sst > 0 else float("nan"),
                        partial_eta2=round(v / (v + sse), 6) if (v + sse) > 0 else float("nan"),
                        F=round(float(F), 4) if np.isfinite(F) else None,
                        p=None if not np.isfinite(p) else float(f"{p:.3e}"),
                    ))
                rows.append(dict(dataset=label, fold=fold, metric=metric, term="residual",
                                 ss=round(sse, 8), df=int(dfres),
                                 eta2=round(sse / sst, 6) if sst > 0 else float("nan"),
                                 partial_eta2=None, F=None, p=None))
                rows.append(dict(dataset=label, fold=fold, metric=metric, term="total",
                                 ss=round(sst, 8), df=int(len(y) - 1), eta2=1.0,
                                 partial_eta2=None, F=None, p=None))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "wp1_eta2.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_v2")
    ap.add_argument("--only", default="", help="Subset of 1a,1b,1c,1d,1e")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    want = set(x.strip() for x in args.only.split(",") if x.strip()) or {"1a", "1b", "1c", "1d", "1e"}

    d = load_all()
    print(f"Loaded {len(d)} runs; datasets {sorted(d['ds_key'].unique())}")
    print(f"NaN check: excess {int(d['excess'].isna().sum())}, roc_auc {int(d['roc_auc'].isna().sum())}, "
          f"pr_auc {int(d['pr_auc'].isna().sum())}")

    if "1a" in want:
        t = wp1a(d, args.outdir)
        print(f"[1a] {len(t)} rows -> wp1_winrate.csv")
    if "1b" in want:
        t = wp1b(d, args.outdir)
        print(f"[1b] {len(t)} rows -> wp1_tau.csv")
    if "1c" in want:
        t = wp1c(d, args.outdir)
        print(f"[1c] {len(t)} rows -> wp1_pairs.csv")
    if "1d" in want:
        t, r = wp1d(d, args.outdir)
        print(f"[1d] {len(t)} rows -> wp1_swings.csv; ratios {len(r)} rows -> wp1_swing_ratios.csv")
    if "1e" in want:
        t = eta2_anova(d, args.outdir)
        print(f"[1e] {len(t)} rows -> wp1_eta2.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
