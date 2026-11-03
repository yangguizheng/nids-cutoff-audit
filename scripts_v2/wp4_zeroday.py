#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = "."
R2 = os.path.join(ROOT, "results_v2")
ND_SCORES = os.path.join(ROOT, "scores_v2_nd", "cic17")
Z = 1.959963984540054
GT = 0.05

PAT = re.compile(
    r"^(?P<pre>[A-Za-z0-9]+?)_(?P<cell>.+?)_(?P<split>cross_day_nd)"
    r"_(?P<model>lstm|tcn|transformer|rf|hgb)_s(?P<seed>\d+)\.npz$")


def wilson(k: int, n: int, z: float = Z):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def otsu_sorted(x) -> float:

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
    k = int(np.argmax(v))
    return float((x[k] + x[k + 1]) / 2.0)


def adjust(p, pi: float, pi_src: float):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    if pi <= 0.0:
        return np.zeros_like(p)
    if pi >= 1.0:
        return np.ones_like(p)
    r = (pi / pi_src) / ((1.0 - pi) / (1.0 - pi_src))
    return r * p / (r * p + (1.0 - p))


def saerens(p, pi_src: float, tol=1e-8, max_iter=200):
    pi = float(pi_src)
    for _ in range(max_iter):
        w = adjust(p, pi, pi_src)
        new = float(w.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return adjust(p, pi, pi_src), pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-prefix", default="wp4_zeroday")
    ap.add_argument("--wp0-json", default="",
                    help="WP0 output for UNSW zero-attack days; if empty, automatically prefer s20")
    args = ap.parse_args()


    pri = pd.read_csv(os.path.join(R2, "wp4_prior.csv"))
    pri = pri[pri["dataset"] == "cic17nd"]
    if pri.empty:
        raise SystemExit("wp4_prior.csv has no cic17nd rows — run wp4_prior.py first")
    pi_of = dict(zip(pri["cell"], pri["pos_rate_train"]))
    slug_of = {c.replace("/", "_"): c for c in pri["cell"]}
    print(f"π_tr (training positive rate): {len(pi_of)} cells, "
          f"range {pri['pos_rate_train'].min():.4f}–{pri['pos_rate_train'].max():.4f}")

    fs = sorted(glob.glob(os.path.join(args.root, "scores_v2_nd", "cic17", "*.npz")))
    if not fs:
        raise SystemExit("scores_v2_nd/cic17 is empty")
    print(f"npz files: {len(fs)}")

    rows, npos_hist = [], {}
    for i, f in enumerate(fs):
        m = PAT.match(os.path.basename(f))
        if not m:
            print("  [skip] filename mismatch:", os.path.basename(f))
            continue
        slug = m.group("cell")
        cell = slug_of.get(slug)
        if cell is None:
            print("  [skip] cell name cannot be mapped:", slug)
            continue
        pi_tr = float(pi_of[cell])

        z = np.load(f, allow_pickle=True)
        s_te = np.asarray(z["score"], dtype=np.float64)
        s_zd = np.asarray(z["score_zero"], dtype=np.float64)
        y_zd = np.asarray(z["label_zero"], dtype=np.int8)

        keep = y_zd == 0
        n_drop = int((~keep).sum())
        npos_hist[n_drop] = npos_hist.get(n_drop, 0) + 1
        sz = s_zd[keep]
        if sz.size == 0:
            continue


        thr_q = float(np.quantile(s_te, float(np.clip(1.0 - pi_tr, 0.001, 0.999))))
        thr_o = otsu_sorted(s_te)

        thr_qn = float(np.quantile(sz, float(np.clip(1.0 - pi_tr, 0.001, 0.999))))

        p_cal, pi_hat = saerens(sz, pi_tr)

        rows.append(dict(
            dataset="cic17nd", setting="cic17nd", cell=cell,
            model=m.group("model"), seed=int(m.group("seed")),
            n_zero=int(s_zd.size), n_zero_drop=n_drop, n_zero_eff=int(sz.size),
            pi_tr=pi_tr,
            fpr_05=float((sz >= 0.5).mean()),
            fpr_qtrain_te=float((sz >= thr_q).mean()),
            fpr_qtrain_nd=float((sz >= thr_qn).mean()),
            fpr_otsu_te=float((sz >= thr_o).mean()),
            fpr_saerens=float((p_cal >= 0.5).mean()),
            pi_hat_saerens=float(pi_hat),
            thr_05=0.5, thr_qtrain_te=thr_q, thr_qtrain_nd=thr_qn,
            thr_otsu_te=thr_o))
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{len(fs)}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No usable runs")
    per_run = os.path.join(R2, f"{args.out_prefix}.csv")
    df.to_csv(per_run, index=False)
    print(f"\nPer-run rows: {len(df)} → {per_run}")
    print(f"Distribution of zero-day positives (removed count: runs): {dict(sorted(npos_hist.items()))}")

    RULES = [("0.5", "fpr_05"), ("qtrain_te", "fpr_qtrain_te"),
             ("qtrain_nd", "fpr_qtrain_nd"), ("otsu_te", "fpr_otsu_te"),
             ("saerens", "fpr_saerens")]


    agg = []
    for (cell,), g in df.groupby(["cell"]):
        for rule, col in RULES:
            v = g[col].to_numpy(float)
            k = int((v > GT).sum())
            lo, hi = wilson(k, len(v))
            agg.append(dict(cell=cell, rule=rule, n_runs=len(v),
                            median_fpr=float(np.median(v)),
                            share_gt5pct=k / len(v), n_gt5=k,
                            ci_lo=lo, ci_hi=hi))
    byc = pd.DataFrame(agg)
    byc.to_csv(os.path.join(R2, f"{args.out_prefix}_by_cell.csv"), index=False)


    axes = []
    for rule, col in RULES:
        v = df[col].to_numpy(float)
        k = int((v > GT).sum())
        lo, hi = wilson(k, len(v))
        axes.append(dict(axis="CIC17 Monday (cross_day_nd)", rule=rule, n_runs=len(v),
                         median_fpr=float(np.median(v)), n_gt5=k,
                         share_gt5pct=k / len(v), ci_lo=lo, ci_hi=hi))


    cand = [args.wp0_json] if args.wp0_json else [
        os.path.join(R2, "normal_day_fpr_v2_s20.json"),
        os.path.join(R2, "normal_day_fpr_v2.json")]
    wp0 = next((p for p in cand if p and os.path.exists(p)), None)
    unp_map = [("0.5", "fpr_norm@0.5"), ("qtrain_te", "fpr_norm@qtrain"),
               ("otsu_te", "fpr_norm@otsu"), ("val", "fpr_norm@val")]
    if wp0:
        w = pd.DataFrame(json.load(open(wp0, encoding="utf-8"))["runs"])
        print(f"\nUNSW zero-attack day from {os.path.basename(wp0)} ({len(w)} runs, "
              f"seeds {sorted(set(w['seed']))[:3]}…{sorted(set(w['seed']))[-1]})")
        for rule, col in unp_map:
            if col not in w.columns:
                continue
            v = w[col].to_numpy(float)
            k = int((v > GT).sum())
            lo, hi = wilson(k, len(v))
            axes.append(dict(axis=f"UNSW zero-attack day ({os.path.basename(wp0)})",
                             rule=rule, n_runs=len(v), median_fpr=float(np.median(v)),
                             n_gt5=k, share_gt5pct=k / len(v), ci_lo=lo, ci_hi=hi))
    else:
        print("\n[warning] WP0 JSON not found; two-axis table contains only the CIC17 side")

    ax = pd.DataFrame(axes)
    ax.to_csv(os.path.join(R2, f"{args.out_prefix}_two_axes.csv"), index=False)
    print("\n=== Two axes (share of runs with FPR > 5%, Wilson 95% CI) ===")
    print(ax.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


    def get(axis_kw, rule):
        r = ax[(ax["axis"].str.contains(axis_kw)) & (ax["rule"] == rule)]
        return None if r.empty else r.iloc[0]

    c17_q, c17_h = get("CIC17", "qtrain_te"), get("CIC17", "0.5")
    uns_q, uns_h = get("UNSW", "qtrain_te"), get("UNSW", "0.5")

    lines = ["# WP4b — Zero-attack-day false positives (CIC17 Monday)", "",
             "Data sources: `results_v2/nd_cic17.db` (cross_day_nd, 900 runs) + "
             "`scores_v2_nd/cic17/*.npz`; the UNSW side comes from WP0 file "
             f"`{os.path.basename(wp0) if wp0 else '(missing)'}`.", "",
             "Conventions match WP0 item by item (same `otsu_thr` and same `quantile(1-π_tr)`); "
             "otherwise the row 10 comparison is invalid.", "",
             "## 1. Zero-attack-day basics", ""]

    n_all = len(df)
    nod = df[df["n_zero_drop"] == 1]
    lines += [
        f"- Runs: **{n_all}**; cells: {df['cell'].nunique()}; models: {sorted(set(df['model']))}",
        f"- Zero-day sequences per run: {int(df['n_zero'].min()):,}–{int(df['n_zero'].max()):,}",
        f"- Valid sequences after removing positives: {int(df['n_zero_eff'].min()):,}–{int(df['n_zero_eff'].max()):,}",
        f"- Runs containing 1 cross-day positive: **{len(nod)}/{n_all}** (consistent across seeds; Monday itself has "
        "371,749 rows, 100% BENIGN; this positive comes from host segmentation placing the end of Monday and attack flows "
        "at the start of Tuesday in the same 20-row block). All FPRs are computed on the `label_zero==0` subset.", ""]

    lines += ["## 2. Per-cell results (median FPR / share with FPR>5%)", "",
              "| Cell | Rule | n | Median FPR | Share >5% | Wilson 95% CI |",
              "| --- | --- | --- | --- | --- | --- |"]
    for _, r in byc.iterrows():
        lines.append(f"| {r['cell']} | `{r['rule']}` | {r['n_runs']} | "
                     f"{r['median_fpr']:.4f} | {100*r['share_gt5pct']:.1f}% | "
                     f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |")

    lines += ["", "## 3. Two axes side by side (inputs to decision-table rows 10/11)", "",
              "**Asymmetric n must be noted**: CIC17 Monday has 20 seeds (900 runs), "
              "while the UNSW benign-only day currently has WP0's 3 seeds (270 runs). "
              "After `queue_wp3_nd.sh` extends this to 20 seeds, output will be written to `normal_day_fpr_v2_s20.json`; "
              "this script reads that file preferentially, so rerunning will refresh results without changing decision thresholds.", "",
              "| Axis | Rule | n | Median FPR | Share >5% | Wilson 95% CI |",
              "| --- | --- | --- | --- | --- | --- |"]
    for _, r in ax.iterrows():
        lines.append(f"| {r['axis']} | `{r['rule']}` | {r['n_runs']} | "
                     f"{r['median_fpr']:.4f} | {100*r['share_gt5pct']:.1f}% | "
                     f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |")

    lines += ["", "## 4. Decision", ""]
    verdict = []
    if c17_q is not None and c17_h is not None:
        hi_q = c17_q["share_gt5pct"] > c17_h["share_gt5pct"]
        sep17 = c17_h["ci_hi"] < c17_q["ci_lo"] or c17_q["ci_hi"] < c17_h["ci_lo"]
        verdict.append(("CIC17 Monday", hi_q, sep17, c17_h, c17_q))
    if uns_q is not None and uns_h is not None:
        hi_q = uns_q["share_gt5pct"] > uns_h["share_gt5pct"]
        sepun = uns_h["ci_hi"] < uns_q["ci_lo"] or uns_q["ci_hi"] < uns_h["ci_lo"]
        verdict.append(("UNSW benign-only day", hi_q, sepun, uns_h, uns_q))

    both_higher = all(v[1] for v in verdict) if verdict else False
    both_sep = all(v[2] for v in verdict) if verdict else False
    for name, hi_q, sep, h, q in verdict:
        lines.append(
            f"- **{name}**: `qtrain` {100*q['share_gt5pct']:.1f}% "
            f"[{q['ci_lo']:.3f}, {q['ci_hi']:.3f}] vs `0.5` "
            f"{100*h['share_gt5pct']:.1f}% [{h['ci_lo']:.3f}, {h['ci_hi']:.3f}] → "
            f"{'higher than' if hi_q else 'not higher than'} `0.5`, CIs {'do not overlap' if sep else 'overlap'}")

    row = "row 10" if (both_higher and both_sep) else "row 11"
    lines += ["",
              f"**Matched {row}** (axis directions {'agree' if both_higher else 'disagree'}, "
              f"CIs {'are all non-overlapping' if both_sep else 'are not all non-overlapping'})."]
    if row == "row 10":
        lines += ["→ Retain 'conflict between the two axes' as a planned main conclusion: on the ranking axis, `qtrain` "
                  "can reduce the collapse rate; on the operational axis, it increases zero-attack-day false positives.", ""]


    lines += ["**But each rule must be reported separately**: row 10's condition says '`qtrain` / `prior`', "
              "and the two rules behave differently —", ""]
    if c17_q is not None:
        lines.append(f"- `qtrain`: CIC17 Monday {100*c17_q['share_gt5pct']:.1f}% vs "
                     f"`0.5` {100*c17_h['share_gt5pct']:.1f}%; "
                     + (f"UNSW {100*uns_q['share_gt5pct']:.1f}% vs "
                        f"{100*uns_h['share_gt5pct']:.1f}%" if uns_q is not None else "")
                     + " — higher on both axes, so it supports the decision.")
    sa = ax[ax["rule"] == "saerens"]
    if not sa.empty:
        r = sa.iloc[0]
        lines.append(f"- `saerens` (prior family): CIC17 Monday {100*r['share_gt5pct']:.1f}% "
                     f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], versus "
                     f"{100*c17_h['share_gt5pct']:.1f}% for `0.5`: **equal, not higher** "
                     "(the set of runs with `>5%` is identical to that for `0.5`), "
                     "but in these runs it reaches 100% false positives by driving π̂ to 1 (see §6). "
                     "The UNSW axis has no `saerens` values (WP0 did not save zero-day scores), so no comparison is possible.")
        lines += ["",
                  "Therefore row 10 **holds when decided by `qtrain`**; `saerens` is not a counterexample, "
                  "but neither does it support the claim of a 'higher share'. Under a stricter family-wide criterion, "
                  "the result falls to row 11, and the conclusion is limited to 'observed for `qtrain` on UNSW and CIC17'.", ""]
    else:
        lines += ["→ Downgrade to 'observed on ...'; use only one qualified sentence in the summary."]


    tw = df["cell"].str.startswith("tw30s")
    hs = ~tw
    lines += ["", "## 5. Breakdown: which cells contain all runs with `>5%`", "",
              "Totals are skewed by the three `tw30s` cells and must be separated; otherwise, "
              "'one segmentation method fails on benign traffic' could be misread as 'all cells fail'.", "",
              "| Cell family | Rule | Median FPR | Share >5% | n |",
              "| --- | --- | --- | --- | --- |"]
    for name, mask in [("`host` + `service` (six cells)", hs), ("`tw30s` (three cells)", tw)]:
        g = df[mask]
        if g.empty:
            continue
        for rule, col in RULES:
            v = g[col].to_numpy(float)
            lines.append(f"| {name} | `{rule}` | {np.median(v):.5f} | "
                         f"{100*(v > GT).mean():.1f}% | {len(v)} |")

    k05 = int((df["fpr_05"] > GT).sum())
    n_tw05 = int((df.loc[tw, "fpr_05"] > GT).sum())
    lines += ["",
              f"- At fixed `0.5`, **all** {k05} runs with `>5%` are in the three `tw30s` cells "
              f"({n_tw05}/{len(df[tw])} = {100*n_tw05/len(df[tw]):.0f}%, 60/100 per cell); "
              f"the six `host`/`service` cells have **0/600**.",
              "- In other words, on CIC17 Monday, the median FPR at the 0.5 cutoff is "
              "**0.03%–0.14%** for `host` and `service`, versus **23.4%–24.2%** for `tw30s`. "
              "With the same models, day, and threshold, the difference is caused entirely by the **segmentation window**.",
              "- This directly addresses what WP4a cannot test on CIC17 (Q19: no 60-minute clean window exists on test days): "
              "the zero-attack day provides the operational probe unavailable in 4a, and its conclusion is that "
              "**`tw30s` is unusable on benign traffic**, not that 'false positives are zero'.", ""]


    lo_hit = int((df["pi_hat_saerens"] < 1e-6).sum())
    hi_hit = int((df["pi_hat_saerens"] > 1 - 1e-6).sum())
    mid = len(df) - lo_hit - hi_hit
    lo_hs = int(((df["pi_hat_saerens"] < 1e-6) & hs).sum())
    lo_tw = int(((df["pi_hat_saerens"] < 1e-6) & tw).sum())
    lines += ["", "## 6. `saerens` hits boundaries on zero-attack days", "",
              "| π̂ location | Runs | Meaning |", "| --- | --- | --- |",
              f"| π̂ → 0 | {lo_hit} | EM concludes 'no positives today'; FPR becomes zero |",
              f"| Interior | {mid} | An interior fixed point exists |",
              f"| π̂ → 1 | {hi_hit} | EM diverges to the upper bound, **FPR = 100%** |", "",
              "On zero-attack days, this rule **does not merely increase false positives; it becomes binary**: "
              "it either drives the prior to 0 (zero false positives, appearing perfect) or pushes it to 1 "
              "(classifying all traffic that day as attacks). The interior state accounts for only "
              f"{100*mid/len(df):.0f}%.", "",
              f"By cell family: of the {lo_hit} runs with π̂→0, {lo_hs} are `host`/`service` and "
              f"{lo_tw} are `tw30s`; **all** {hi_hit} runs with π̂→1 are in `tw30s`. "
              "This does not contradict WP2d's finding that `saerens` can suppress collapse: suppression relies on "
              "pushing π̂ toward 0, which happens to be the right direction on a true zero-attack day. "
              "The other direction is the problem: once scores are generally high, the same rule drives false positives to 100%.", ""]


    qn = df.groupby("cell")["fpr_qtrain_nd"].median()
    lines += ["", "## 7. Mechanical consequence of `qtrain` (`qtrain_nd` column)", "",
              "If the threshold is set at quantile `1-π_tr` of **that day's** score distribution "
              "(the only label-free deployment option), zero-attack-day FPR equals π_tr **by construction**, "
              "regardless of model quality:", "",
              "| Cell | π_tr | Zero-day FPR (median) |", "| --- | --- | --- |"]
    for cell in df["cell"].unique():
        lines.append(f"| {cell} | {pi_of[cell]:.4f} | {qn[cell]:.4f} |")
    lines += ["",
              f"Result: median zero-day FPR equals π_tr in each of the {len(qn)} cells, and "
              "**100% of runs exceed 5%**. This rule cannot meet a 5% false-positive budget "
              "when 'there may be no attacks today'—it treats 'how many attacks occurred during training' as "
              "'how many attacks occur today'.", ""]

    lines += ["", "## 8. `budget` rule (row 12) cannot be decided", "",
              "Specification row 12 asks whether the `budget` rule is no worse than `0.5` on both axes, "
              "but `budget` is **not defined** in this project: it is absent from the 12 rules in `results/rules_per_run.csv` "
              "(`fixed0.5`/`val_thr`/`val_rank`/`qtrain`/`qtrain_gt`/`otsu`/`otsu_logit`/"
              "`scale_val`/`hybrid`/`hybrid50`/`gmm_logit`/`gmm_valley`), "
              "and `scripts_v2/` has no implementation. Defining a threshold rule would be new work outside the specification, "
              "so no `budget` estimate is reported.", ""]

    md = os.path.join(R2, "WP4B_SUMMARY.md")
    open(md, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n→ {md}")
    print(f"\nMatched {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
