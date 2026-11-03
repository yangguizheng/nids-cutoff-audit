#!/usr/bin/env python


from __future__ import annotations

import os
import sqlite3

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


BATCHES = [

    ("results/main.db", "v1-main-sweep", "(v1, run_pilot.py)", "(not used for conclusions)",
     "v1 three-dataset main sweep; cudnn.benchmark was not explicitly disabled, so reruns are not bitwise reproducible"),
    ("results/cic18.db", "v1-cic18-sweep", "(v1, run_pilot.py)", "(not used for conclusions)",
     "v1 CIC18 parameter sweep (including exploratory combinations of T and cells)"),
    ("results/unsw.db", "v1-unsw", "(v1, run_pilot.py)", "(not used for conclusions)",
     "v1 UNSW parameter sweep"),
    ("results/det_cic17.db", "v1-det", "(v1, run_pilot.py)", "(not used for conclusions)",
     "Legacy **control** database for deterministic v1 reruns: used by WP2e as a gate, not cited directly"),
    ("results/det_cic18.db", "v1-det", "(v1, run_pilot.py)", "(not used for conclusions)",
     "Same as above; WP2e --verify-db target"),
    ("results/det_unsw.db", "v1-det", "(v1, run_pilot.py)", "(not used for conclusions)",
     "Same as above; WP2e --verify-db target"),
    ("results/runs.db", "v1-probe", "(v1, early probe)", "(not used for conclusions)", "Early smoke test"),
    ("results/repro1.db", "v1-probe", "(v1, single reproducibility run)", "(not used for conclusions)",
     "Residual from a single-run reproducibility check, 1 row"),

    ("results_v2/det_v2_cic17.db", "v2-main", "scripts_v2/queue_wp2e.sh", "§5, §6.1, §6.2",
     "Main database: 9 cells × 5 models × 20 seeds"),
    ("results_v2/det_v2_cic18.db", "v2-main", "scripts_v2/queue_wp2e.sh", "§5, §6.1, §6.2",
     "Main database: 9 cells × 5 models × 20 seeds"),
    ("results_v2/det_v2_unsw.db", "v2-main", "scripts_v2/queue_wp2e.sh",
     "§5, §6.1, §6.2", "Main database: 2 folds × 9 cells × 5 models × 5 seeds"),
    ("results_v2/det_nocw_cic17.db", "v2-nocw", "scripts_v2/queue_wp2a.sh", "§6.1 (M1 mechanism)",
     "Arm without class weighting (--class-weight none)"),
    ("results_v2/det_nocw_cic18.db", "v2-nocw", "scripts_v2/queue_wp2a.sh", "§6.1 (M1 mechanism)",
     "Arm without class weighting"),
    ("results_v2/det_nocw_unsw.db", "v2-nocw", "scripts_v2/queue_wp2a.sh", "§6.1 (M1 mechanism)",
     "Arm without class weighting"),
    ("results_v2/splits_cic18.db", "v2-splits", "scripts_v2/queue_wp2c.sh",
     "§6.2 (split sensitivity)", "Two splits, block_strat / cross_day_ext, 10 seeds"),
    ("results_v2/det_cic18_val.db", "v2-val-dump", "scripts_v2/queue_wp2d.sh + queue_wp2d_tail.sh",
     "§6.2.4 (Saerens/BBSE calibration)", "--dump-val: additionally saves validation-fold scores"),
    ("results_v2/det_cic17_val.db", "v2-val-dump", "scripts_v2/queue_wp4c.sh",
     "§6.2.4", "--dump-val"),
    ("results_v2/det_unsw_val.db", "v2-val-dump", "scripts_v2/queue_wp4c.sh", "§6.2.4",
     "--dump-val"),
    ("results_v2/det_unsw_s20.db", "v2-unsw-s20", "scripts_v2/queue_wp3.sh",
     "§6.2 (seed-count sensitivity)", "UNSW extended to 20 seeds"),
    ("results_v2/nd_cic17.db", "v2-nd", "scripts_v2/queue_wp4b.sh",
     "§6.2.7 (zero-attack-day false positives)", "--split cross_day_nd --zero-days 0"),
    ("results_v2/e1/e1_runs.db", "v2-e1", "scripts_v2/queue_e1.sh",
     "§6.2.6 (within-dataset intervention)", "E1: three arms in the Inf group (inf_U / inf_S / inf_P)"),

    ("results_v2/_probe/probe_det_cic17.db", "v2-probe", "(one-time smoke test)",
     "(not used for conclusions)", "Five smoke-test runs to verify pipeline execution"),
    ("results_v2/_probe/probe_det_cic18.db", "v2-probe", "(one-time smoke test)",
     "(not used for conclusions)", "Five smoke-test runs"),
    ("results_v2/_probe/probe_det_unsw.db", "v2-probe", "(one-time smoke test)",
     "(not used for conclusions)", "Five smoke-test runs"),
]


def scan(path: str) -> dict:
    ab = os.path.join(_ROOT, path)
    if not os.path.exists(ab):
        return dict(n_runs=0, seeds="", n_seeds=0, models="", n_cells=0,
                    cudnn="", plan_commit="", datasets="", splits="")
    con = sqlite3.connect(ab)
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(runs)")]
        n = con.execute("select count(*) from runs").fetchone()[0]
        seeds = sorted({r[0] for r in con.execute("select distinct seed from runs")})
        models = sorted({r[0] for r in con.execute("select distinct model from runs")})
        cells = {r[0] for r in con.execute("select distinct cell from runs")}
        ds = sorted({r[0] for r in con.execute("select distinct dataset from runs")})
        sp = sorted({r[0] for r in con.execute("select distinct split from runs")})
        cudnn = ""
        if "cudnn_benchmark" in cols:
            vs = sorted({r[0] for r in con.execute(
                "select distinct cudnn_benchmark from runs")})
            cudnn = ",".join(str(int(v)) if v is not None else "?" for v in vs)
        pc = ""
        if "analysis_plan_commit" in cols:
            vs = sorted({str(r[0]) for r in con.execute(
                "select distinct analysis_plan_commit from runs")})
            pc = ",".join(vs)
        return dict(n_runs=n, seeds=_rng(seeds), n_seeds=len(seeds),
                    models=",".join(models),
                    n_cells=len(cells), cudnn=cudnn, plan_commit=pc,
                    datasets=",".join(map(str, ds)), splits=",".join(map(str, sp)))
    finally:
        con.close()


def _rng(vals) -> str:
    if not vals:
        return ""
    if len(vals) == 1:
        return str(vals[0])
    return f"{vals[0]}-{vals[-1]}" if vals == list(range(vals[0], vals[-1] + 1)) \
        else ",".join(map(str, vals))


def main() -> int:
    rows = []
    for path, batch, script, used, note in BATCHES:
        s = scan(path)
        rows.append(dict(
            batch=batch, db=path, n_runs=s["n_runs"], seeds=s["seeds"],
            n_seeds=s["n_seeds"],
            datasets=s["datasets"], splits=s["splits"], n_cells=s["n_cells"],
            models=s["models"],

            cudnn_benchmark=s["cudnn"],
            analysis_plan_commit=s["plan_commit"],

            retrained=("yes" if batch.startswith("v2") else "no"),
            producing_script=script, used_in_section=used, note=note))
    d = pd.DataFrame(rows)


    def tot(mask):
        g = d[mask]
        return int(g.n_runs.sum())

    ok = d.used_in_section.str.startswith("§")
    d = pd.concat([d, pd.DataFrame([dict(
        batch="TOTAL citable (v2, cudnn.benchmark=0)", db="(total)",
        n_runs=tot(ok), seeds="", n_seeds=0, datasets="", splits="", n_cells=0,
        models="", cudnn_benchmark="0", analysis_plan_commit="",
        retrained="yes", producing_script="", used_in_section="All manuscript numbers",
        note="All manuscript numbers must come from these databases"),
        dict(batch="TOTAL not used for conclusions (v1 + probes)", db="(total)",
             n_runs=tot(~ok), seeds="", n_seeds=0, datasets="", splits="", n_cells=0,
             models="", cudnn_benchmark="", analysis_plan_commit="",
             retrained="no", producing_script="", used_in_section="(not used for conclusions)",
             note="Historical and gate comparisons only; reruns are not bitwise reproducible")])], ignore_index=True)

    out = os.path.join(_ROOT, "results_v2", "RUN_LEDGER.csv")
    d.to_csv(out, index=False, encoding="utf-8")
    print(f"→ {out} ({len(d)} rows)")
    with pd.option_context("display.width", 200, "display.max_colwidth", 28):
        print(d[["batch", "n_runs", "n_seeds", "cudnn_benchmark", "retrained",
                 "used_in_section"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
