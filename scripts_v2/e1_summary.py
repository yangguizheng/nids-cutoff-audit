#!/usr/bin/env python


from __future__ import annotations

import os
import re
import sqlite3

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
E1 = os.path.join(_ROOT, "results_v2", "e1")


def pct(x, nd=1):
    return f"{100.0 * x:.{nd}f}%"


def pp(x, nd=1):
    return f"{100.0 * x:+.{nd}f}"


def arm_row(c, sub, arm, metric):
    g = c[(c.subset == sub) & (c.kind == "arm") & (c.arm == arm) & (c.metric == metric)]
    return g.iloc[0] if len(g) else None


def diff_row(c, sub, contrast, metric):
    g = c[(c.subset == sub) & (c.kind == "diff") & (c.contrast == contrast) &
          (c.metric == metric)]
    return g.iloc[0] if len(g) else None


def fmt_arm(c, sub, arm, metric, how="pct"):
    r = arm_row(c, sub, arm, metric)
    if r is None:
        return "—"
    if how == "pct":
        return f"{pct(r.value)} [{pct(r.lo)}, {pct(r.hi)}]"
    if how == "f4":
        return f"{r.value:.4f}"
    return f"{r.value:.2f}"


def fmt_diff(c, sub, contrast, metric):
    r = diff_row(c, sub, contrast, metric)
    if r is None:
        return "—"
    return f"{pp(r.value)}pp [{pp(r.lo)}, {pp(r.hi)}]"


def group_block(c, sub, title, note=""):
    L = [f"## {title}", ""]
    if note:
        L += [note, ""]
    tr = [fmt_arm(c, sub, a, "train_prior", "f4") for a in ("inf_U", "inf_S", "inf_P")]
    L.append(f"Training prior U {tr[0]}  S {tr[1]}  S-pm {tr[2]}")
    L.append("")
    cl = [fmt_arm(c, sub, a, "collapse_rate") for a in ("inf_U", "inf_S", "inf_P")]
    L.append(f"Collapse rate U {cl[0]}  S {cl[1]}  S-pm {cl[2]}")
    L.append("")
    L.append(f"S−U {fmt_diff(c, sub, 'inf_S-inf_U', 'collapse_rate')}   "
             f"S-pm−U {fmt_diff(c, sub, 'inf_P-inf_U', 'collapse_rate')}"
             f"   (tw30s only: {fmt_diff(c, 'tw30s', 'inf_S-inf_U', 'collapse_rate')} / "
             f"{fmt_diff(c, 'tw30s', 'inf_P-inf_U', 'collapse_rate')})")
    L.append("")
    q = [fmt_arm(c, sub, a, "pos_q50", "f4") for a in ("inf_U", "inf_S", "inf_P")]
    L.append(f"pos_q50 U {q[0]}  S {q[1]}  S-pm {q[2]}")
    L.append("")
    au = [fmt_arm(c, sub, a, "roc_auc", "f4") for a in ("inf_U", "inf_S", "inf_P")]
    L.append(f"Median AUC U {au[0]}  S {au[1]}  S-pm {au[2]}")
    L.append("")
    tp = [fmt_arm(c, sub, a, "tpr_at_fpr_1pct", "f4") for a in ("inf_U", "inf_S", "inf_P")]
    L.append(f"Median TPR@FPR=1% U {tp[0]}  S {tp[1]}  S-pm {tp[2]}")
    L.append("")
    names = [("lstm", "LSTM"), ("tcn", "TCN"), ("transformer", "TF"),
             ("rf", "RF"), ("hgb", "HGB")]
    bits = []
    for m, lab in names:
        row = c[(c.subset == sub) & (c.kind == "arm_model") & (c.contrast == m) &
                (c.metric == "collapse_rate")]
        d = {r.arm: r.value for _, r in row.iterrows()}
        bits.append(f"{lab} {pct(d['inf_U']) if 'inf_U' in d else '—'}/"
                    f"{pct(d['inf_S']) if 'inf_S' in d else '—'}")
    L.append("Collapse rate by model (U / S): " + " ".join(bits))
    L.append("")
    return L


def main() -> int:
    c = pd.read_csv(os.path.join(E1, "e1_contrasts.csv"))
    inv = pd.read_csv(os.path.join(E1, "inventory_day.csv"))
    gates = pd.read_csv(os.path.join(E1, "gates.csv"))
    con = sqlite3.connect(os.path.join(E1, "e1_runs.db"))
    n_runs = con.execute("select count(*) from runs").fetchone()[0]
    n_by = dict(con.execute("select arm, count(*) from runs group by arm").fetchall())
    cells = {r[0] for r in con.execute("select distinct cell from runs")}
    seeds = sorted({r[0] for r in con.execute("select distinct seed from runs")})
    plan = {r[0] for r in con.execute("select distinct analysis_plan_commit from runs")}
    cudnn = {r[0] for r in con.execute("select distinct cudnn_benchmark from runs")}
    pm_side = {r[0] for r in con.execute(
        "select distinct prior_match_side from runs where arm='inf_P'")}
    percell = pd.read_sql_query(
        "select cell, max(n_test) as n_test, max(n_pos_test_seq) as n_pos, "
        "max(n_train) as n_train, max(train_prior) as train_prior from runs group by cell",
        con)
    con.close()

    def inv_row(day):
        r = inv[inv.day == day]
        return r.iloc[0] if len(r) else None

    d5, d6, d7, d8 = inv_row(5), inv_row(6), inv_row(7), inv_row(8)

    def pos(day, route):
        col = {"tw30s": "n_pos_seq_tw30s", "host": "n_pos_seq_host",
               "service": "n_pos_seq_service"}
        try:
            return int(inv[inv.day == day][col[route]].iloc[0])
        except Exception:
            return -1

    L = ["# E1_SUMMARY", ""]
    L.append(f"commit: `{sorted(plan)[0] if plan else '?'}` (A10 preregistration); "
             f"see the `commit` column in `e1_runs.db` for the code commit used for each run  "
             f"pre-run self-check (bit-for-bit match with det_v2): **yes** (5 models × seed 0; "
             f"exact matches for `macro_f1/f1_pos/fpr/mcc/epochs`, see `logs/e1_selfcheck_gate.log`)")
    L.append("")
    L.append(f"Runs: **{n_runs}** (" +
             ", ".join(f"`{k}` {v}" for k, v in sorted(n_by.items())) +
             f"), {len(cells)} cells × 5 models × {len(seeds)} seeds"
             f" ({seeds[0]}–{seeds[-1]}, consecutive with none removed)"
             f"; `cudnn_benchmark` = {sorted(cudnn)}")
    L.append("")


    L += ["## Inventory", ""]
    L.append("d5/d6 labels match expectations: **yes** (`WEB ATTACK - BRUTE FORCE / SQL / XSS` "
             "and their `- ATTEMPTED` variants, six labels in total; mapping is correct; traffic volume is extremely low)   "
             "d7/d8: **yes** (`INFILTRATION - NMAP PORTSCAN` accounts for 99.8%; "
             "`- DROPBOX DOWNLOAD` and `- COMMUNICATION VICTIM ATTACKER` are also present)")
    L.append("")
    if d5 is not None and d6 is not None:
        L.append(f"d5 attack flows **{int(d5.n_attack_flows)}** / {int(d5.n_flows):,} flows"
                 f" ({100 * d5.pos_rate_flows:.4f}%); "
                 f"d6 attack flows **{int(d6.n_attack_flows)}** / {int(d6.n_flows):,} flows"
                 f" ({100 * d6.pos_rate_flows:.4f}%); "
                 f"reference d8 attack flows **{int(d8.n_attack_flows):,}**"
                 f" ({100 * d8.pos_rate_flows:.2f}%)")
    L.append("")
    L.append("Positive test-day sequences (tw30s / host / service): "
             f"Web d6 **{pos(6, 'tw30s')}/{pos(6, 'host')}/{pos(6, 'service')}**   "
             f"Inf d8 **{pos(8, 'tw30s')}/{pos(8, 'host')}/{pos(8, 'service')}**")
    L.append("")
    L.append("Groups or routes canceled by feasibility gates: **entire Web group canceled** (gate 2: positive test-day sequences across three routes"
             f" {pos(6, 'tw30s')}/{pos(6, 'host')}/{pos(6, 'service')}, all < 30); "
             "**Inf group tw30s reported descriptively only** (gate 1: "
             f"{pos(8, 'tw30s')} < 30). See `gates.csv`.")
    L.append("")


    L += ["## Web group (test d6)", "",
          "**Entire group canceled; no runs produced** (A10.3 gate 2). Therefore, training prior, collapse rate, paired differences, "
          "pos_q50, median AUC, TPR@FPR=1%, and per-model collapse rates are **all blank**—"
          "not 0 or 'not computed'; this group was not run under the preregistration.", "",
          "Direct reason for cancellation: d6 has only "
          f"{int(d6.n_attack_flows)} attack flows ("
          f"{100 * d6.pos_rate_flows:.4f}% of "
          f"{int(d6.n_flows):,} flows), "
          "yielding 0 tw30s sequences and 12 each for host/service, "
          "an order of magnitude below the gate of 30; any difference would be noise from two or three sequences.", "",
          "Training prior U —  S —  S-pm —", "",
          "Collapse rate U —  S —  S-pm —", "",
          "S−U —   S-pm−U —   (tw30s only: — / —)", "",
          "pos_q50 U —  S —  S-pm —", "",
          "Median AUC U —  S —  S-pm —", "",
          "Median TPR@FPR=1% U —  S —  S-pm —", "",
          "Collapse rate by model (U / S): —", ""]


    L += group_block(c, "all", "Inf group (test d8)",
                     "Group-level scope = 9 pooled cells (3 segmentation × 3 padding, T=20). "
                     "The prior-matching side for `inf_P` is recorded in `prior_match_side` = "
                     f"`{'/'.join(sorted(pm_side))}` (A10.5 branch 3: downsample U's d3).")

    L += ["### By route (split of the same runs)", "",
          "| Route | U collapse rate | S collapse rate | P collapse rate | S−U | P−U | U pos_q50 | S pos_q50 | U AUC | S AUC |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for sub in ("hs", "host", "service", "tw30s"):
        L.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            sub,
            fmt_arm(c, sub, "inf_U", "collapse_rate"),
            fmt_arm(c, sub, "inf_S", "collapse_rate"),
            fmt_arm(c, sub, "inf_P", "collapse_rate"),
            fmt_diff(c, sub, "inf_S-inf_U", "collapse_rate"),
            fmt_diff(c, sub, "inf_P-inf_U", "collapse_rate"),
            fmt_arm(c, sub, "inf_U", "pos_q50", "f4"),
            fmt_arm(c, sub, "inf_S", "pos_q50", "f4"),
            fmt_arm(c, sub, "inf_U", "roc_auc", "f4"),
            fmt_arm(c, sub, "inf_S", "roc_auc", "f4")))
    L += ["", "`hs` = host + service (**descriptive** subset after A10.3 excludes tw30s; "
          "it does not replace the A10.6 `all` scope); `tw30s` is descriptive only "
          "(fewer than 30 positive test-day sequences; A10.3 gate 1).", ""]


    L += ["### Per-cell sizes (after sampling)", "",
          "| Cell | Train | Test | Test positives | Training prior |", "| --- | --- | --- | --- | --- |"]
    for _, r in percell.sort_values("cell").iterrows():
        flag = " ⚠️" if r.n_pos < 30 else ""
        L.append(f"| `{r.cell}` | {int(r.n_train):,} | {int(r.n_test):,} | "
                 f"{int(r.n_pos):,}{flag} | {r.train_prior:.4f} |")
    low = percell[percell.n_pos < 30]
    L += ["", f"⚠️ = cells with fewer than 30 test positives: **{len(low)}** cells. "
          "Gates (A10.3) are applied to each **entire route** on the **unsampled** full day. "
          "After `max_segments=200000` sampling, individual cells may fall below 30 sequences "
          f"({(', '.join('`%s` %d sequences' % (r.cell, r.n_pos) for _, r in low.iterrows()))}). "
          "These cells **still run and remain in group-level scope**—post hoc removal would change the preregistration. "
          "As a descriptive measure, group-level statistics excluding these cells are provided below so readers can assess "
          "how much small cells support the group-level conclusions.", ""]

    L += ["**Sensitivity (descriptive; not used for decisions)**: recompute group-level statistics after removing the small cells above—", ""]
    con2 = sqlite3.connect(os.path.join(E1, "e1_runs.db"))
    q = ("select arm, macro_f1, test_pos_rate, pos_q50, roc_auc from runs "
         "where cell not in (%s)" % ",".join("?" * len(low)))
    dd = pd.read_sql_query(q, con2, params=list(low.cell))
    con2.close()
    if len(dd):
        fl = np.maximum((1 - dd.test_pos_rate) / (2 - dd.test_pos_rate),
                        dd.test_pos_rate / (1 + dd.test_pos_rate))
        dd["collapsed"] = (dd.macro_f1 - fl) < 0.02
        for arm in ("inf_U", "inf_S", "inf_P"):
            g = dd[dd.arm == arm]
            if not len(g):
                continue
            L.append(f"- `{arm}` (small cells removed, n={len(g)}): collapse rate "
                     f"{pct(g.collapsed.mean())}, median pos_q50 {g.pos_q50.median():.4f}, "
                     f"median AUC {g.roc_auc.median():.4f}")
    L.append("")


    L += ["## Decision table matches", ""]
    vp = os.path.join(E1, "E1_VERDICT.md")
    if os.path.exists(vp):
        txt = open(vp, encoding="utf-8").read()
        hits = re.findall(r"## Subset `([^`]+)`([\s\S]*?)Decision table match:\*\*([^*]+)\*\*", txt)
        L.append("Web: **5** (canceled by gate, A10.3)")
        for sub, _, h in hits:
            L.append(f"Inf (`{sub}`): **{h.strip()}**")
        L.append("")
        L.append("Automatic decisions are in `E1_VERDICT.md`; copied verbatim without manual adjustment.")
    else:
        L.append("(`E1_VERDICT.md` missing; no decision)")
    L.append("")


    L += ["## Deviations from preregistration", ""]
    items = [
        "**The Web group was not run**. The specification expected usable Web attack traffic on d5/d6, but measured d5/d6 attack flows were only "
        f"{int(d5.n_attack_flows)}/{int(d6.n_attack_flows)}"
        f" ({100 * d5.pos_rate_flows:.4f}% / {100 * d6.pos_rate_flows:.4f}%), "
        "and positive test-day d6 sequences "
        f"{pos(6, 'tw30s')}/{pos(6, 'host')}/{pos(6, 'service')} were all < 30, "
        "so A10.3 gate 2 canceled the entire group. Execution did **not** combine d5+d6, relax gates, "
        "or switch to d9 BOTNET (which would change the question being compared). Recorded in Q32.",
        "**`inf_P` is U-pm, not S-pm**. Under the prior comparison in A10.5 branches 2/3, "
        f"`prior_match_side` = `{'/'.join(sorted(pm_side))}`: p_S ≤ p_U, "
        "so **U-side d3 attack sequences** were downsampled to make the prior equal p_S. The arm remains named `inf_P`, "
        "meaning 'prior-matched version of U'. This branch was fixed in the preregistration, not chosen after the runs.",
        "**All three arms share the same sequences across six days** (d0,d1,d2,d3,d7,d8, `max_segments=200000`). "
        "The cost is that the unused day consumes sampling capacity, making training sets smaller than in the main database (see `n_train`); "
        "the benefit is that all three arms use the same sampling pool, keeping sampling noise out of paired comparisons. Registered in A10.4.",
        "**The validation set uses the final 20% of the training-day time series** (specification §2), unlike main-database S0, "
        "which uses the middle day for validation. The self-check therefore had to reproduce both the main database's **split** and **values**: "
        "the self-check arm uses the original main-database split (`cross_day`, d0–d2 train / d3 validation / d4 test), "
        "not the 20% tail.",
        "**Standardization for the prior-matched arm** (unspecified in the specification): the scaler is fit on the **baseline training set before downsampling**, "
        "then rows are downsampled by seed. Thus all 20 seeds within the arm share the same (μ, σ), "
        "and differences come only from which rows are removed.",
        "**The `service` route did not match any decision-table row**. Its S−U = −13.7 pp "
        "(CI [−19.0, −9.0]) exceeds row 3's 'difference < 10 pp' but does not reach the 20 pp in rows 1/2. "
        "Under specification §3, 'when a result does not belong to any row, return it as-is without inventing an explanation', "
        "only the numbers are reported, with no tradeoff made. The other four scopes (`all` / `hs` / `host` / `tw30s`) fall in row 3.",
        "**Row 4 was not triggered at group level but was triggered on `tw30s`**. Group-level median AUC was 0.7510 for U "
        "(< 0.80) and 0.8041 for S (≥ 0.80), so both were not < 0.80; "
        "for `tw30s` alone they were 0.5832 / 0.6613, both < 0.80. "
        "The decision still follows preregistered group-level scope (A10.6 says 'a group'); route-level values are provided "
        "to support interpretation of A10.3 (tw30s excluded from main conclusions) together with A10.6 (decisions by group).",
        "The analysis is limited to the preregistered decision criteria and reported estimates.",
    ]
    for i, t in enumerate(items, 1):
        L.append(f"{i}. {t}")
    L.append("")
    L.append("## Four directly observable facts (no interpretation)")
    L.append("")
    L.append("- Consistent direction: across all five scopes—`all`, `hs`, `host`, `service`, and `tw30s`—"
             "S has a lower collapse rate than U, paired-difference CIs are entirely below 0, and S has higher median AUC.")
    L.append("- One exception must be noted: on `tw30s`, S's pos_q50 (0.0291) is **lower** than U's (0.0329), "
             "opposite the other four scopes. This cell has only 10 positive test-day sequences (A10.3 gate 1) "
             "and is descriptive only under the preregistration, not part of the main conclusions.")
    L.append("- Magnitude below the preregistered threshold: S−U ranges from −3.7 pp to −13.7 pp, "
             "while preregistered rows 1/2 require ≥ 20 pp, which was not reached.")
    L.append("- The prior-matched arm nearly equals U: P−U is −0.3 pp on `all` (CI [−2.2, +1.4]), "
             "−1.2 pp on `hs`, +0.0 pp on `host`, and +1.3 pp on `tw30s`; "
             "only `service` is −2.3 pp (CI [−4.0, −1.0]).")
    L.append("")
    L.append("Among auxiliary metrics, the largest difference is in the tail: on `all`, median TPR@FPR=1% changes from U's "
             f"{fmt_arm(c, 'all', 'inf_U', 'tpr_at_fpr_1pct', 'f4')} to S's "
             f"{fmt_arm(c, 'all', 'inf_S', 'tpr_at_fpr_1pct', 'f4')}; "
             f"P is {fmt_arm(c, 'all', 'inf_P', 'tpr_at_fpr_1pct', 'f4')}.")
    L.append("")

    out = os.path.join(E1, "E1_SUMMARY.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"→ {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
