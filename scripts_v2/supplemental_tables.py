#!/usr/bin/env python


from __future__ import annotations

import math
import os
import sqlite3
import sys
import unicodedata

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
OUT = os.path.join(R2, "final", "Supplemental_Tables.xlsx")

F_TITLE = Font(bold=True, size=11)
F_SRC = Font(italic=True, size=8, color="444444")
F_HEAD = Font(bold=True, size=9)
F_BODY = Font(size=9)
FILL_HEAD = PatternFill("solid", fgColor="E8E8E8")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(vertical="top", wrap_text=True)
TOP = Alignment(vertical="top")


RULE_DEFS = [
    ("fixed0.5",
     "Fixed cut-off at 0.5 on the predicted attack score.",
     "test-day scores", "no", ""),
    ("val_thr",
     "The cut-off of the previous day's labelled validation run, applied unchanged to the test day.",
     "validation-day scores and labels", "yes", ""),
    ("val_rank",
     "The quantile position F of the validation cut-off inside the validation scores is mapped onto "
     "the same quantile of the test-day score distribution.",
     "validation-day scores and labels", "yes", ""),
    ("qtrain",
     "Cut-off at the (1 - training attack rate) weighted quantile of the test-day score distribution.",
     "training attack rate", "no", ""),
    ("qtrain_gt",
     "As qtrain, but the cut-off is moved to the next representable double so that p >= thr selects "
     "exactly the windows above the quantile value.",
     "training attack rate", "no", "tie-handling variant of qtrain"),
    ("otsu",
     "Otsu's threshold on the linear score scale: the split that maximises between-class weighted "
     "variance of the test-day scores.",
     "test-day scores", "no", ""),
    ("otsu_logit",
     "Otsu's threshold computed on the log-odds scale, mapped back through the sigmoid.",
     "test-day scores", "no", ""),
    ("scale_val",
     "Otsu on the linear scale and Otsu on the log-odds scale are both scored by macro-F1 on the "
     "previous day's labelled validation run; the better scale is applied to the test day.",
     "validation-day scores and labels", "yes", ""),
    ("hybrid",
     "Otsu on the log-odds scale if that cut-off falls below 0.01 (scores compressed near zero), "
     "otherwise linear Otsu.",
     "test-day scores", "no",
     "the 0.01 bound was chosen post hoc on the same data: diagnostic, not a deployable rule"),
    ("hybrid50",
     "As hybrid, with a 0.05 bound.",
     "test-day scores", "no",
     "the 0.05 bound was chosen post hoc on the same data: diagnostic, not a deployable rule"),
    ("gmm_logit",
     "Two-component weighted Gaussian mixture fitted on the log-odds scale; cut-off at the point "
     "where the two weighted component densities are equal.",
     "test-day scores", "no", ""),
    ("gmm_valley",
     "As gmm_logit, but reverts to 0.5 unless both the density dip ratio and the component "
     "separation pass fixed thresholds.",
     "test-day scores", "no", "falls back to 0.5 when the two clusters are not separated"),
    ("saerens",
     "Saerens-Latinne-Decaestecker EM prior-shift correction starting from the training attack rate; "
     "the adjusted posteriors are thresholded at 0.5.",
     "training attack rate and test-day scores", "no", ""),
    ("bbse",
     "Black-box shift estimation: the test prior is estimated from validation-day true- and "
     "false-positive rates and the test-day predicted-positive rate, the posteriors are re-weighted, "
     "and the result is thresholded at 0.5.",
     "validation-day labels and test-day scores", "yes", ""),
    ("saerens_cal",
     "Isotonic calibration fitted on the validation day, then Saerens EM with the validation-day "
     "positive rate as the source prior; thresholded at 0.5.",
     "validation-day scores and labels", "yes", ""),
]
RULE_ORDER = [r[0] for r in RULE_DEFS]
VAL_RULES = {r[0] for r in RULE_DEFS if r[3] == "yes"}

SETTING_MAP = {
    "cic17": ("CIC-IDS2017", "cross_day"),
    "cic18": ("CSE-CIC-IDS2018", "cross_day"),
    "unsw_pair": ("NF-UNSW-NB15", "cross_day_pair"),
    "cic17nd": ("CIC-IDS2017 (zero-attack day)", "cross_day_nd"),
}


def _disp_len(s) -> int:

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(s))


def _num_fmt(cell) -> None:
    v = cell.value
    if v is None:
        return
    if isinstance(v, str):
        cell.font = F_BODY
        cell.alignment = WRAP
        return
    try:
        f = float(v)
    except (TypeError, ValueError):
        return
    cell.font = F_BODY
    cell.alignment = TOP
    if isinstance(v, int):
        cell.number_format = "0"
        return
    if f == 0:
        cell.number_format = "0"
    elif abs(f) < 1e-4:
        cell.number_format = "0.00E+00"
    elif abs(f) <= 1:
        cell.number_format = "0.0000"
    else:
        cell.number_format = "0.000"


def write_sheet(wb, name, caption, source, df, notes=None, col_widths=None):
    ws = wb.create_sheet(title=name)
    ncol = len(df.columns)

    ws.cell(row=1, column=1, value=caption).font = F_TITLE
    ws.cell(row=2, column=1, value="Source: " + source).font = F_SRC
    r = 3
    for n in (notes or []):
        ws.cell(row=r, column=1, value=n).font = F_SRC
        r += 1
    r += 1


    widths = {}
    for j, c in enumerate(df.columns, start=1):
        if col_widths and c in col_widths:
            w = col_widths[c]
        else:
            body = [_disp_len(x) for x in df[c].head(600) if not isinstance(x, float)]
            w = max([_disp_len(c)] + body) + 2 if body else _disp_len(c) + 2
            w = max(9, min(w, 36))
        widths[j] = w

    hr = r
    for j, c in enumerate(df.columns, start=1):
        cell = ws.cell(row=hr, column=j, value=str(c))
        cell.font = F_HEAD
        cell.fill = FILL_HEAD
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = BORDER

    data = df.astype(object).where(pd.notna(df), None)
    for i, row in enumerate(data.itertuples(index=False), start=hr + 1):
        for j, v in enumerate(row, start=1):
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            cell = ws.cell(row=i, column=j, value=v)
            cell.border = BORDER
            _num_fmt(cell)

    last = hr + len(data)
    ws.auto_filter.ref = "A%d:%s%d" % (hr, get_column_letter(ncol), last)
    if len(data):
        ws.freeze_panes = ws.cell(row=hr + 1, column=1)

    for j, c in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(j)].width = widths[j]


    for i, row in enumerate(data.itertuples(index=False), start=hr + 1):
        lines = 1
        for j, v in enumerate(row, start=1):
            if isinstance(v, str):
                lines = max(lines, math.ceil(_disp_len(v) / max(widths[j] - 1, 4)))
        ws.row_dimensions[i].height = min(max(lines, 1), 9) * 11.6 + 2

    ws.row_dimensions[hr].height = 30


    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.page_setup.fitToWidth = 1 if ncol <= 14 else 2
    ws.page_setup.fitToHeight = 0
    ws.print_title_rows = "1:%d" % hr
    return ws, len(data)


def t_s1():
    x = pd.read_csv(os.path.join(ROOT, "lit", "extraction.csv"))
    d = pd.read_csv(os.path.join(ROOT, "lit", "d1_calls.csv"))
    d = d[["key", "d1", "grade", "source", "sentence"]].rename(
        columns={"d1": "d1_sequence_verdict", "grade": "d1_grade",
                 "source": "d1_source", "sentence": "d1_evidence_sentence"})
    m = x.merge(d, on="key", how="left")
    m["d1_sequence_verdict"] = m["d1_sequence_verdict"].fillna("not_assessed")
    order = ["key", "year", "venue", "doi", "dataset", "split_type", "T",
             "padding_reported", "threshold_reported", "metric", "fpr_reported",
             "n_seeds", "auc_reported", "url_checked",
             "d1_sequence_verdict", "d1_grade", "d1_source", "d1_evidence_sentence"]
    return m[order]


def t_s2():
    w = pd.read_csv(os.path.join(R2, "wp1_winrate_s20.csv"))


    is_4b = (
        (w.dataset == "CIC18")
        & (((w.route_subset == "all") & (w.metric == "roc_auc") & (w.eps == 0.0))
           | ((w.route_subset == "tw30s") & (w.metric == "excess") & (w.eps == 0.0))
           | ((w.route_subset == "tw30s") & (w.metric == "roc_auc") & (w.eps == 0.0))
           | ((w.route_subset == "all") & (w.metric == "roc_auc") & (w.eps == 0.001)))
    )
    w["section"] = np.where(is_4b, "two robustness contrasts (former main-text Table 4b)",
                             "full grid")
    order = ["section", "dataset", "fold", "route_subset", "metric", "eps", "model",
             "winrate", "ci_lo", "ci_hi", "wilson_lo", "wilson_hi",
             "n_groups", "n_groups_dropped_na"]
    w = w[order]

    idx = np.where(is_4b, 0, 1)
    return w.iloc[np.argsort(idx, kind="stable")].reset_index(drop=True)


def t_s3():
    t = pd.read_csv(os.path.join(R2, "wp1_tau_s20.csv"))
    order = ["dataset", "fold", "route_subset", "pair", "aggregation", "tau",
             "exact_perm_p", "boot_frac_tau_lt0", "boot_tau_ci_lo", "boot_tau_ci_hi", "n_groups"]
    return t[order]


def t_s4():
    return pd.read_csv(os.path.join(R2, "day_families.csv"))


def t_s5():
    df = pd.DataFrame(RULE_DEFS, columns=["rule", "definition", "inputs_used",
                                          "requires_validation_labels", "note"])
    return df


def t_s6():
    return pd.read_csv(os.path.join(R2, "wp5_j1.csv"))


def t_s7():
    arms = [
        ("CIC17", "det_v2_cic17.db", "det_nocw_cic17.db"),
        ("CIC18", "det_v2_cic18.db", "det_nocw_cic18.db"),
        ("UNSW", "det_unsw_s20.db", "det_nocw_unsw.db"),
    ]
    rows = []
    for dataset, cw_name, nocw_name in arms:
        arms_rows = []
        for db_name in (cw_name, nocw_name):
            con = sqlite3.connect(os.path.join(R2, db_name))
            data = pd.read_sql_query(
                "select cell, model, seed, fold, test_pos_rate, macro_f1, "
                "pos_q50, roc_auc, thr_oracle from runs", con)
            con.close()
            if data.duplicated(["cell", "model", "seed", "fold"]).any():
                raise ValueError(f"Duplicate S7 pairing key in {db_name}")
            arms_rows.append(data)
        cw, nocw = arms_rows
        c = cw.merge(nocw, on=["cell", "model", "seed", "fold"],
                     suffixes=("_cw", "_nocw"), validate="one_to_one")
        if len(c) != len(cw) or len(c) != len(nocw):
            raise ValueError(f"Incomplete S7 pairing for {dataset}")
        p_cw = c["test_pos_rate_cw"].to_numpy(float)
        p_nocw = c["test_pos_rate_nocw"].to_numpy(float)
        floor_cw = np.maximum((1 - p_cw) / (2 - p_cw), p_cw / (1 + p_cw))
        floor_nocw = np.maximum((1 - p_nocw) / (2 - p_nocw), p_nocw / (1 + p_nocw))
        excess_cw = c["macro_f1_cw"].to_numpy(float) - floor_cw
        excess_nocw = c["macro_f1_nocw"].to_numpy(float) - floor_nocw
        c["dataset"] = dataset
        c["segment"] = c["cell"].map(lambda x: "tw30s" if x.startswith("tw") else x.split("/")[0])
        c["d_excess"] = excess_cw - excess_nocw
        c["d_roc_auc"] = c["roc_auc_cw"] - c["roc_auc_nocw"]
        c["d_pos_q50"] = c["pos_q50_cw"] - c["pos_q50_nocw"]
        c["d_thr_oracle"] = c["thr_oracle_cw"] - c["thr_oracle_nocw"]
        c["coll_cw"] = excess_cw < 0.02
        c["coll_nocw"] = excess_nocw < 0.02
        rows.append(c)
    c = pd.concat(rows, ignore_index=True)
    if len(c) != 3600:
        raise ValueError(f"Expected 3600 S7 pairs, got {len(c)}")
    g = c.groupby(["dataset", "model", "segment"], as_index=False).agg(
        n_runs=("cell", "size"),
        d_excess_median=("d_excess", "median"),
        d_excess_mean=("d_excess", "mean"),
        d_roc_auc_median=("d_roc_auc", "median"),
        collapse_rate_cw=("coll_cw", "mean"),
        collapse_rate_nocw=("coll_nocw", "mean"),
        d_pos_q50_median=("d_pos_q50", "median"),
        d_thr_oracle_median=("d_thr_oracle", "median"),
    )
    g = g.sort_values(["dataset", "segment", "model"]).reset_index(drop=True)
    g.insert(3, "route", g["segment"])
    return g


def t_s8():
    s = pd.read_csv(os.path.join(R2, "wp4d_summary.csv"))
    s = s[s.setting != "unsw_pair"].copy()
    s["dataset"] = s["setting"].map(lambda k: SETTING_MAP[k][0])
    s["split"] = s["setting"].map(lambda k: SETTING_MAP[k][1])

    u = pd.read_csv(os.path.join(R2, "wp4d_rules_unsw_s20_val.csv"))
    rows = []
    groups = [
        (0, "unsw_pair_f0", "cross_day_pair fold 0"),
        (1, "unsw_pair_f1", "cross_day_pair fold 1"),
        (None, "unsw_pair", "cross_day_pair both folds"),
    ]
    for fold, setting, split in groups:
        g = u if fold is None else u[u.fold == fold]
        base = g["excess_fixed0.5"].to_numpy(float)
        for rule in RULE_ORDER:
            row = dict(setting=setting, dataset="NF-UNSW-NB15", split=split, rule=rule,
                       n=np.nan, collapse=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                       excess_median=np.nan, rescue=np.nan, harm=np.nan,
                       fpr_median=np.nan, zero_day_fpr_median=np.nan)
            if rule in VAL_RULES and fold != 0:
                rows.append(row)
                continue
            e = g["excess_" + rule].to_numpy(float)
            ok = np.isfinite(e)
            c = e[ok] < 0.02
            bb = base[ok] < 0.02
            k = int(c.sum())
            n = int(ok.sum())
            z = 1.959963984540054
            p = k / n
            den = 1.0 + z * z / n
            centre = (p + z * z / (2 * n)) / den
            half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
            row.update(n=n, collapse=p, ci_lo=centre - half, ci_hi=centre + half,
                       excess_median=float(np.median(e[ok])),
                       rescue=float((bb & ~c).sum() / max(bb.sum(), 1)),
                       harm=float((~bb & c).sum() / max((~bb).sum(), 1)),
                       fpr_median=float(np.nanmedian(g["fpr_" + rule].to_numpy(float))))
            rows.append(row)
    s = pd.concat([s, pd.DataFrame(rows)], ignore_index=True)
    s["rule_order"] = s["rule"].map({r: i for i, r in enumerate(RULE_ORDER)})
    setting_order = {"cic17": 0, "cic17nd": 1, "cic18": 2,
                     "unsw_pair_f0": 3, "unsw_pair_f1": 4, "unsw_pair": 5}
    s["setting_order"] = s["setting"].map(setting_order)
    s = s.sort_values(["setting_order", "rule_order"])
    order = ["setting", "dataset", "split", "rule", "n", "collapse", "ci_lo", "ci_hi",
             "excess_median", "rescue", "harm", "fpr_median", "zero_day_fpr_median"]
    return s[order].reset_index(drop=True)


def t_s9():
    return pd.read_csv(os.path.join(R2, "wp4_clean_run.csv"))


def t_s10():
    e = pd.read_csv(os.path.join(R2, "wp1_eta2.csv"))
    r2 = pd.read_csv(os.path.join(R2, "r2", "R2_eta2.csv")).drop(columns=["source"])
    e = pd.concat([e[e.dataset == "CIC18"], r2], ignore_index=True)
    e = e.sort_values(["dataset", "fold", "metric", "term"]).reset_index(drop=True)
    return e


def t_s11():
    p = pd.read_csv(os.path.join(R2, "wp2c_splits.csv"))
    p["split_order"] = p["split"].map(lambda s: int(str(s)[1:]) if str(s)[0] == "S" else 9)
    p = p.sort_values(["split_order", "model"]).drop(columns=["split_order"])
    return p.reset_index(drop=True)


def t_s12():


    subsets = {
        "all": dict(n=900, arms={
            "U": (75.4, 72.5, 78.1, -0.0000, 0.7510, 0.0065, 0.0701, 0.1000),
            "S": (66.4, 63.3, 69.5,  0.0000, 0.8041, 0.0188, 0.0642, 0.2854),
            "P": (75.1, 72.2, 77.8,  0.0000, 0.7397, 0.0067, 0.0667, 0.1000),
        }, su=(-9.0, -11.3, -6.8), pu=(-0.3, -2.2, 1.4), verdict="3"),
        "hs": dict(n=600, arms={
            "U": (67.8, 64.0, 71.4, 0.0000, 0.8685, 0.0004, 0.1103, None),
            "S": (59.2, 55.2, 63.0, 0.0000, 0.9067, 0.0011, 0.1079, None),
            "P": (66.7, 62.8, 70.3, 0.0000, 0.8666, 0.0003, 0.1103, None),
        }, su=(-8.7, -11.7, -5.8), pu=(-1.2, -2.8, 0.7), verdict="3"),
        "host": dict(n=300, arms={
            "U": (42.3, 36.9, 48.0, 0.1437, 0.7889, 0.0017, 0.1103, None),
            "S": (38.7, 33.3, 44.3, 0.1847, 0.8410, 0.0021, 0.1079, None),
            "P": (42.3, 36.9, 48.0, 0.1490, 0.7967, 0.0008, 0.1103, None),
        }, su=(-3.7, -6.0, -1.7), pu=(0.0, -3.0, 3.0), verdict="3"),
        "service": dict(n=300, arms={
            "U": (93.3, 89.9, 95.6, -0.0000, 0.9163, 0.0002, 0.0667, None),
            "S": (79.7, 74.8, 83.8, 0.0000, 0.9457, 0.0007, 0.0642, None),
            "P": (91.0, 87.2, 93.7, -0.0000, 0.9235, 0.0003, 0.0667, None),
        }, su=(-13.7, -19.0, -9.0), pu=(-2.3, -4.0, -1.0), verdict="none"),
        "tw30s": dict(n=300, arms={
            "U": (90.7, 86.8, 93.5, -0.0004, 0.5832, 0.0329, 0.0701, None),
            "S": (81.0, 76.2, 85.0, -0.0004, 0.6613, 0.0291, 0.0528, None),
            "P": (92.0, 88.4, 94.6, -0.0002, 0.6102, 0.0483, 0.0529, None),
        }, su=(-9.7, -13.3, -5.7), pu=(1.3, -3.0, 5.7), verdict="3,4"),
    }
    rows = []
    for subset, sd in subsets.items():
        for arm, v in sd["arms"].items():
            rows.append(dict(
                group="Inf", subset=subset, arm=arm, n=sd["n"],
                collapse_rate=v[0] / 100.0, wilson_lo=v[1] / 100.0, wilson_hi=v[2] / 100.0,
                s_minus_u_pp=sd["su"][0], s_minus_u_lo=sd["su"][1], s_minus_u_hi=sd["su"][2],
                p_minus_u_pp=sd["pu"][0], p_minus_u_lo=sd["pu"][1], p_minus_u_hi=sd["pu"][2],
                excess_median=v[3], roc_auc_median=v[4], pos_q50_median=v[5],
                tpr_fpr1pct_median=(v[7] if v[7] is not None else np.nan),
                train_prior=v[6], verdict_row=sd["verdict"],
            ))
    return pd.DataFrame(rows)


def t_s13():


    rows = [
        dict(dataset="CIC-IDS2017",         n_seeds=20, k1=0.533, k2=0.607, k3=0.638, k5=0.671, k10=0.745),
        dict(dataset="CSE-CIC-IDS2018",     n_seeds=20, k1=0.588, k2=0.653, k3=0.691, k5=0.741, k10=0.808),
        dict(dataset="NF-UNSW-NB15 fold 0", n_seeds=20, k1=0.744, k2=0.842, k3=0.887, k5=0.935, k10=0.981),
        dict(dataset="NF-UNSW-NB15 fold 1", n_seeds=20, k1=0.888, k2=0.915, k3=0.916, k5=0.901, k10=1.000),
    ]
    return pd.DataFrame(rows)


def t_s14():
    rows = [
        (1, 'S3: align training and test positive rates',
         'Computed offline: attack sequences of the test day were subsampled without replacement (20 repetitions, averaged) to the training prior of each cell; models were not retrained',
         'Retraining was not needed to change the test prior',
         'S3 cannot change the outcome of runs that predict no positive (163 of 506 collapsed CSE-CIC-IDS2018 runs); 419 had f1_pos = 0, which is not equivalent to predicting no positive. M2 therefore has no effective retraining test (row 11)'),
        (2, 'Validation-based cut-off rules (val_thr, val_rank, scale_val, BBSE, calibrated EM) on NF-UNSW-NB15, both folds',
         'Reported for fold 0 only (900 runs)',
         'The validation tail of fold 1 contains no attack sequences',
         'Table 12 footnote; comparisons are made against fixed 0.5 on the same 900 runs'),
        (3, 'E1 Web group (test day 23 February 2018)',
         'Cancelled before any run',
         'Pre-specified feasibility gate: at most 12 positive test sequences per route against a minimum of 30',
         'E1 rests on the Infiltration group only'),
        (4, 'E1 tw30s route',
         'Reported descriptively',
         'Pre-specified gate: 16 positive test sequences',
         'None on the pre-specified decision'),
        (5, 'Run determinism',
         'cuDNN autotuning disabled; deterministic cuDNN algorithms not enforced',
         'Enforcing them slowed TCN training about 30-fold',
         '666 of 675 re-trained runs reproduced exactly; nine TCN runs differed by at most 0.0018 in macro-F1 and 0.0002 in ROC-AUC'),
        (6, 'Early stopping on validation macro-F1',
         'Unchanged, but in NF-UNSW-NB15 fold 1 the validation tail is attack-free, so the criterion rewards epochs without false alarms',
         'Property of the two-fold split',
         'Disclosed in Experimental Setup and Limitation 5'),
        (7, 'M2: S3 collapse at least 20 points below S0, or saerens/BBSE collapse at most 15%',
         'Validation-calibrated EM (saerens_cal), an added cut-off-rule variant, reduced collapse by 20.5 points relative to fixed 0.5; it was not named in the pre-specified M2 rule',
         'Variant added to the cut-off rule set',
         'Reported as a post-hoc variant; M2 is reported as not supported under the pre-specified rule but not excluded'),
        (8, 'Within-family AUC and median scores in M3; cross-dataset comparison of these quantities',
         'The pre-specified M3 quantities were computed; comparisons between datasets were descriptive and were not added as decision rules',
         '—',
         'Cross-dataset comparisons labelled as descriptive in Results'),
        (9, 'NF-UNSW-NB15: 20 seeds per cell (pre-specified before the new runs)',
         'The first class-weight analysis used seeds 0-4; both class-weight arms and the main 20-seed analyses were subsequently extended to seeds 0-19',
         'Pre-specified target of 20 seeds; preserve paired class-weight comparisons and improve precision',
         'Directions of all results unchanged'),
        (10, 'M1: dataset-level CSE-CIC-IDS2018 collapse-rate difference',
         'Applied the decision rule to aggregate CSE-CIC-IDS2018 collapse rates; per-model results were reported separately',
         'The machine-readable rule subtracts aggregate collapse rates and does not define a per-model decision',
         'No per-model M1 decision was substituted for the pre-specified aggregate rule'),
        (11, 'E1 arm P: subsample U attack sequences until its training prior equals that of S',
         'Implemented on tw30s only. On host and service no attack sequences were available for subsampling (prior_match_note = no_extra_attack in all 600 runs), so P repeats U',
         'U and S training priors already differ by less than 0.003 on these routes (0.110 vs 0.108; 0.067 vs 0.064)',
         'P against U is not a prior test; it is no longer used as M2 evidence. On the keyed routes the S against U contrast holds the prior nearly constant'),
        (12, 'Use the published capture days',
         'Each CSE-CIC-IDS2018 and NF-UNSW-NB15 cell keeps at most 200,000 segments (uniform random, before the split); NF-UNSW-NB15 cells are capped at 80,000 sequences (stratified on the label); CIC-IDS2017 is not subsampled',
         'Memory',
         'Disclosed in Datasets and splits'),
        (13, 'Controlled experiments reuse the 20 seeds of the core library',
         'Split controls S1 and S2 use 10 seeds per cell (450 runs each)',
         'Compute budget',
         'Denominators stated in Experimental Setup and Table S11'),
        (14, 'Cross-day split at the day level',
         'On host and service, the last chunk of a key on one day can continue into the next day and is assigned to the earlier day',
         'The grouping key does not contain the day',
         'Disclosed in Definitions and Limitation 10: service route 5.5% to 11.0% of sequences, host route at most 1.1%, tw30s none; test-day attack flows carried into training or validation: 50 (CIC-IDS2017), 11 (CSE-CIC-IDS2018), 0 (E1), up to 726 (NF-UNSW-NB15 fold 1)'),
    ]
    return pd.DataFrame(rows, columns=["#", "Pre-specified element", "What was done", "Reason",
                                       "Effect on conclusions"])


SPECS = [
    ("S1", "Table S1. Coding of the 49 screened flow-sequence NIDS studies and the 10 that meet "
           "the inclusion rule, with reporting items and supporting passages.",
     "lit/extraction.csv (49 rows, 14 reporting columns) joined to lit/d1_calls.csv on key "
     "(D1 sequence verdict, grade, evidence sentence); assembled by scripts_v2/supplemental_tables.py",
     t_s1,
     ["d1_sequence_verdict: yes = the study builds cross-time-step sequences (the 10 in the denominator); "
      "no = does not meet the sequence/dataset inclusion rule; insufficient = full text read but undecidable; "
      "excluded = exclusion rule.",
      "d1_evidence_sentence is an English source quotation with a section/table locator, or NO_FULLTEXT "
      "when the full text is unavailable."],
     {"d1_evidence_sentence": 70, "metric": 26, "doi": 30}),

    ("S2", "Table S2. Paired win rates with cluster-bootstrap 95% intervals for excess, ROC-AUC and "
           "AP-lift, at tie tolerances of 0, 0.001 and 0.005, for all cells and for the tw30s route alone.",
     "results_v2/wp1_winrate_s20.csv (300 rows: 3 datasets x 3 route subsets x 3 metrics x 3 tie "
     "tolerances x 5 models); assembled by scripts_v2/supplemental_tables.py",
     t_s2,
     ["winrate is a fraction in [0,1], not a percentage.",
      "ci_lo/ci_hi are the cluster-bootstrap 95% interval; wilson_lo/wilson_hi are the Wilson 95% "
      "interval on the same numerator and denominator.",
      "route_subset also contains host+service, which the manuscript caption does not list: the table is "
      "a superset of the caption.",
      "n_groups is the number of paired groups the win rate is computed over; n_groups_dropped_na counts "
      "groups lost to missing values."],
     {"route_subset": 16}),

    ("S3", "Table S3. Kendall's tau with exact permutation p-values under win-rate and mean aggregation.",
     "results_v2/wp1_tau_s20.csv (48 rows); assembled by scripts_v2/supplemental_tables.py",
     t_s3,
     ["exact_perm_p is the exact permutation p-value; boot_frac_tau_lt0 and boot_tau_ci_lo/hi come from "
      "the cluster bootstrap.",
      "pair names the two metrics being rank-correlated."],
     {"route_subset": 16}),

    ("S4", "Table S4. Attack families and flow counts on every capture day of the three datasets.",
     "results_v2/day_families.csv (69 rows); assembled by scripts_v2/supplemental_tables.py",
     t_s4,
     ["attack is True when the family label is an attack family; BENIGN rows give the background count."],
     {"family": 34}),

    ("S5", "Table S5. Definitions of the fifteen cut-off rules.",
     "scripts/eval_rules.py (RULES table and the thr_* functions) plus the three prior-correction rules "
     "in scripts_v2/wp4d_rules_v2.py; transcribed by scripts_v2/supplemental_tables.py",
     t_s5,
     ["None of the fifteen uses test-day labels. The oracle cut-off, which does, is reported separately "
      "and is not part of this table.",
      "requires_validation_labels = yes marks the five rules that cannot run without a labelled "
      "validation day; they are the ones left blank where no validation scores exist."],
     {"definition": 78, "inputs_used": 34, "note": 46}),

    ("S6", "Table S6. Zero-feature key baseline (J1) for every cell.",
     "results_v2/wp5_j1.csv (24 rows: 12 cells on the cross-day split and 12 on the paired split); "
     "assembled by scripts_v2/supplemental_tables.py",
     t_s6,
     ["J1 scores each test sequence by the attack rate its grouping key had in the training data "
      "(the global training rate for unseen keys); it reads no flow feature.",
      "auc_gap_to_best_model is best_model_auc minus j1_roc_auc."],
     {"cell": 30}),

    ("S7", "Table S7. Paired comparison with and without class weighting, by dataset, model and route.",
     "Six SQLite run databases (det_v2_cic17, det_v2_cic18, det_unsw_s20 and their det_nocw "
     "counterparts), paired by cell, model, seed and fold (3600 per-run rows) and aggregated to "
     "dataset x model x route (45 rows) by scripts_v2/supplemental_tables.py",
     t_s7,
     ["d_excess, d_roc_auc, d_pos_q50 and d_thr_oracle are within-run differences, "
      "with-class-weighting minus without.",
      "collapse_rate_cw and collapse_rate_nocw are fractions in [0,1]."],
     {"route": 12}),

    ("S8", "Table S8. Collapse rates of all fifteen rules on the three datasets.",
     "results_v2/wp4d_summary.csv and results_v2/wp4d_rules_unsw_s20_val.csv; NF-UNSW-NB15 "
     "aggregated by fold and across both folds by scripts_v2/supplemental_tables.py",
     t_s8,
     ["collapse is a fraction in [0,1]; ci_lo/ci_hi are Wilson 95% intervals.",
      "For NF-UNSW-NB15, fold 0, fold 1 and both folds are reported separately. The five "
      "validation-based rules (val_thr, val_rank, scale_val, bbse, saerens_cal) are available on "
      "fold 0 only (n=900); their fold 1 and combined cells are blank because the fold 1 validation "
      "tail contains no positive sequences. Other rules use n=900 per fold and n=1,800 combined.",
      "rescue and harm count runs that the rule moves across the collapse boundary in each direction.",
      "cic17nd is the zero-attack Monday of CIC-IDS2017, so zero_day_fpr_median is the quantity of "
      "interest there rather than collapse."],
     {"dataset": 30}),

    ("S9", "Table S9. False alarms on clean windows of the test days.",
     "results_v2/wp4_clean_run.csv (375 runs, one row per run); assembled by "
     "scripts_v2/supplemental_tables.py",
     t_s9,
     ["fpr_<rule> is the false-positive rate on clean windows only; fprfull_<rule> is the rate over all "
      "windows of the day, attack windows included.",
      "thr_<rule> is the cut-off that rule chose for that run.",
      "Only the rules present in this table are shown here; the full fifteen-rule comparison is Table S8."],
     {"cell": 34}),

    ("S10", "Table S10. Three-factor variance decomposition (eta squared) for excess and ROC-AUC.",
     "results_v2/r2/R2_eta2.csv for CIC-IDS2017 (det_v2_cic17.db) and NF-UNSW-NB15 "
     "(20 seeds per fold), and results_v2/wp1_eta2.csv for CSE-CIC-IDS2018; assembled by "
     "scripts_v2/supplemental_tables.py",
     t_s10,
     ["ss is the sum of squares, df the degrees of freedom, eta2 and partial_eta2 the two effect sizes, "
      "F and p the F statistic and its p-value.",
      "All rows use 20 seeds; NF-UNSW-NB15 is reported separately by fold.",
      "The residual and total rows are included so the decomposition can be re-summed."],
     {}),

    ("S11", "Table S11. Split controls S0 to S3 on CSE-CIC-IDS2018, by route and model.",
     "results_v2/wp2c_splits.csv (36 rows: 4 splits x 9 model rows); assembled by "
     "scripts_v2/supplemental_tables.py",
     t_s11,
     ["collapse_rate is a fraction in [0,1]; ci_lo/ci_hi are Wilson 95% intervals.",
      "med_* columns are medians over runs within the split and model."],
     {}),

    ("S12", "Table S12. E1 within-dataset intervention: collapse rate, paired S-U and P-U "
            "differences, and auxiliary medians for the three arms U/S/P over all, hs, host, "
            "service and tw30s (Inf group only; Web group cancelled per A10.3).",
     "results_v2/e1/E1_VERDICT.md and results_v2/e1/E1_SUMMARY.md (transcribed verbatim, "
     "not recomputed); assembled by scripts_v2/supplemental_tables.py",
     t_s12,
     ["collapse_rate, wilson_lo/hi, excess_median, roc_auc_median, pos_q50_median, "
      "tpr_fpr1pct_median and train_prior are fractions in [0,1]; tpr_fpr1pct_median is "
      "given only for subset 'all'.",
      "s_minus_u_pp / p_minus_u_pp and their lo/hi are the paired cluster-bootstrap 95% "
      "difference in percentage points (10,000 draws, seed 20261101, group = cell x seed); "
      "the same subset value is repeated on its three arm rows.",
      "Web group was cancelled as a whole (A10.3 gate 2): d5 attack flows 208 (0.0034%), "
      "d6 attack flows 230 (0.0038%); test-day positive sequences tw30s/host/service = "
      "0/12/12, all < 30. No Web runs exist; its decision-table row is 5.",
      "Per-model collapse rate (Inf, all subset) and the pre-specified A10.6 decision rule "
      "with per-subset verdict are appended below.",
      "service hits no decision row: S-U = -13.7 pp exceeds row-3's '<10 pp' and does not "
      "reach row-1/2's 20 pp, so the number is reported as-is per spec section 3."],
     {"subset": 9, "verdict_row": 9}),

    ("S13", "Table S13. Probability that the best model selected on k randomly drawn seeds equals "
            "the best model on all 20 seeds (10,000 draws per cell, draws shared across models, "
            "metric = excess).",
     "results_v2/final/NUMBERS_DIFF.md L430-L433 (20-seed values; 10,000 draws per cell, "
     "metric = excess, draws shared across models within a cell); assembled by "
     "scripts_v2/supplemental_tables.py",
     t_s13,
     ["k1..k10 are probabilities in [0,1] that the best model selected on k randomly drawn seeds "
      "is the same as the best model selected on all 20 seeds.",
      "All four rows use 20 seeds; UNSW is split by fold because the cross-day_pair split has "
      "two test days."],
     {"dataset": 22}),

    ("S14", "Table S14. Deviations from the analysis plan, and clarifications of how "
            "it was implemented.",
     "Analysis plan commit e0dc336 and E1 addendum commit c1ced61 (private repository; public editions and SHA-256 hashes in preregistration/); "
     "assembled by scripts_v2/supplemental_tables.py",
     t_s14,
     ["Plan documents: results_v2/ANALYSIS_PLAN.md at commit e0dc336 and "
      "results_v2/ANALYSIS_PLAN_ADDENDUM.md at commit c1ced61. The plan is internal and was not lodged with a public registry.",
      "Row 9 distinguishes the pre-specified 20-seed target from its later execution; row 10 "
      "clarifies that M1 is an aggregate, not per-model, decision; rows 11-14 were added after an external pre-submission check."],
     {"Pre-specified element": 46, "What was done": 72, "Reason": 44, "Effect on conclusions": 62}),
]


def main() -> int:
    wb = Workbook()
    readme = wb.active
    readme.title = "README"

    rows = []
    for name, cap, src, fn, notes, widths in SPECS:
        df = fn()
        _ws, n = write_sheet(wb, name, cap, src, df, notes, widths)
        rows.append(dict(sheet=name,
                         caption=cap,
                         source=src.split(";")[0],
                         n_rows=n,
                         n_cols=len(df.columns)))
        print("%-4s %5d rows x %2d cols  %s" % (name, n, len(df.columns), cap[:60]))


    readme["A1"] = "Supplemental Tables S1-S14"
    readme["A1"].font = Font(bold=True, size=12)
    readme["A2"] = ("One worksheet per table; the worksheet name is the table number used in the "
                    "manuscript's Supplemental Information list.")
    readme["A2"].font = F_SRC
    readme["A3"] = ("All values are derived from the analysis outputs by scripts_v2/supplemental_tables.py. "
                    "Cells hold the full-precision value and only the display format rounds to four "
                    "decimals, so clicking a cell shows the exact number. Ratios are stored as fractions "
                    "in [0,1], not as percentages.")
    readme["A3"].font = F_SRC

    hdr = [("sheet", "sheet"), ("caption", "caption"), ("source", "source file"),
           ("n_rows", "rows"), ("n_cols", "columns")]
    for j, (_k, label) in enumerate(hdr, start=1):
        c = readme.cell(row=5, column=j, value=label)
        c.font = F_HEAD
        c.fill = FILL_HEAD
        c.border = BORDER
    for i, r in enumerate(rows, start=6):
        for j, (k, label) in enumerate(hdr, start=1):
            c = readme.cell(row=i, column=j, value=r[k])
            c.font = F_BODY
            c.border = BORDER
            c.alignment = WRAP if k in ("caption", "source") else TOP
    for col, w in zip("ABCDE", (8, 62, 52, 7, 8)):
        readme.column_dimensions[col].width = w
    readme.freeze_panes = "A6"
    readme.page_setup.orientation = "landscape"
    readme.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    readme.page_setup.fitToWidth = 1
    readme.page_setup.fitToHeight = 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wb.save(OUT)
    print("\nWrote %s  (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
