from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _c in (os.path.join(_ROOT, "scripts"),
           os.path.join(_ROOT, "src", "data"),
           os.path.join(_ROOT, "src", "models"),
           os.path.join(_ROOT, "scripts_v2")):
    if os.path.isdir(_c):
        sys.path.insert(0, _c)

import seqdiff
import splits_v2 as SP
import wp2e_dump as W
import e1_dump as E1
import normal_day_fpr_v2 as ND

ROUTES = ("host", "service", "time_window")
ROUTE_LABEL = {"host": "host", "service": "service", "time_window": "tw30s"}
SET_NAME = {-1: "unused", 0: "train", 1: "val", 2: "test", 3: "zero"}
TRAIN, VAL, TEST, ZERO = 0, 1, 2, 3

DB_SPECS = {
    "main_cic17": ("det_v2_cic17.db", "cross_day", -1, None),
    "main_cic18": ("det_v2_cic18.db", "cross_day", -1, None),
    "s2_cic18": ("splits_cic18.db", "cross_day_ext", -1, None),
    "e1_U": ("e1/e1_runs.db", "inf_U", -1, 0),
    "e1_S": ("e1/e1_runs.db", "inf_S", -1, 0),
    "e1_P": ("e1/e1_runs.db", "inf_P", -1, 0),
    "unsw_f0": ("det_unsw_s20.db", "cross_day_pair", 0, 0),
    "unsw_f1": ("det_unsw_s20.db", "cross_day_pair", 1, 0),
    "nd_cic17": ("nd_cic17.db", "cross_day_nd", -1, None),
}


def load_data(root: str, dataset: str, days) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    args = SimpleNamespace(
        root=root, dataset=dataset, days=days, cic18_days=days,
        unsw_days=days, rid_prefix="", feature_subset="")
    df, _, _ = W.load_dataset(args)
    code, fam_names = W.label_codes(df, dataset)
    return df, code, fam_names


def route_cfg(route: str, max_segments: int) -> seqdiff.SeqConfig:
    kw = (dict(convention="by_start", truncate="tail", padding="repeat_last")
          if route == "time_window" else dict(padding="repeat_last"))
    return seqdiff.SeqConfig(segment=route, T=20, seed=0,
                             max_segments=max_segments, **kw)


def build_meta(df: pd.DataFrame, cfg: seqdiff.SeqConfig, code: np.ndarray,
               fam_names: list[str], max_seq: int = 0):
    segs = seqdiff.segment_rows(df, cfg)
    rng = np.random.default_rng(cfg.seed)
    if cfg.max_segments and len(segs) > cfg.max_segments:
        keep = np.sort(rng.choice(len(segs), size=cfg.max_segments, replace=False))
        segs = [segs[i] for i in keep]

    tsec = (df["_t_start"] - df["_t_start"].min()).dt.total_seconds().to_numpy()
    day = df["_day"].to_numpy()
    key_col = {"session": "_sid", "service": "_svid", "host": "_hid",
               "time_window": "_sid"}[cfg.segment]
    grp = df[key_col].to_numpy()
    rows_keep, lengths = [], []
    y_bin, groups, wids, days = [], [], [], []
    for rows in segs:
        kept = rows[-cfg.T:] if len(rows) > cfg.T else rows
        rows_keep.append(np.asarray(kept, dtype=np.int64))
        lengths.append(len(rows))
        y_bin.append(int((code[kept] > 0).any()))
        vals, cnts = np.unique(grp[kept], return_counts=True)
        groups.append(int(vals[int(np.argmax(cnts))]))
        wids.append(int(tsec[kept].min()))
        days.append(int(day[kept].min()))

    n = len(rows_keep)
    b = seqdiff.SeqBatch(
        X=np.zeros((n, 0, 0), dtype=np.float32),
        mask=np.zeros((n, 0), dtype=bool),
        length=np.asarray(lengths, dtype=np.int64),
        y_bin=np.asarray(y_bin, dtype=np.int8),
        y_multi=np.zeros(n, dtype=object),
        group=np.asarray(groups, dtype=np.int64),
        wid=np.asarray(wids, dtype=np.int64),
        day=np.asarray(days, dtype=np.int64), meta={})
    si, _ = W.strat_idx_of(b.y_bin, max_seq, seed=0)

    lengths2, _, _, y2 = W.cell_meta(df, cfg, code, fam_names, si)
    if not np.array_equal(lengths2, b.length[si]) or not np.array_equal(y2, b.y_bin[si]):
        raise RuntimeError(f"Metadata alignment failed for {cfg.tag()}")

    if len(si) != n:
        b = seqdiff.SeqBatch(
            X=b.X[si], mask=b.mask[si], length=b.length[si],
            y_bin=b.y_bin[si], y_multi=b.y_multi[si], group=b.group[si],
            wid=b.wid[si], day=b.day[si], meta=b.meta)
        rows_keep = [rows_keep[i] for i in si]
    return b, rows_keep, tsec


def temporal_flow_sets(df: pd.DataFrame, tsec: np.ndarray, train_days,
                       test_days, val_start: int | None = None,
                       zero_days=()) -> np.ndarray:
    day = df["_day"].to_numpy()
    out = np.full(len(df), -1, dtype=np.int8)
    tr = np.isin(day, list(train_days))
    out[tr] = TRAIN
    if val_start is not None:
        out[tr & (tsec >= val_start)] = VAL
    out[np.isin(day, list(test_days))] = TEST
    out[np.isin(day, list(zero_days))] = ZERO
    return out


def fmt_counts(counter: collections.Counter, family: bool = False) -> str:
    if not counter:
        return "none"
    if family:
        return "; ".join(f"{direction}|{fam}={n}"
                         for (direction, fam), n in sorted(counter.items()))
    return "; ".join(f"{direction}: {n}" for direction, n in sorted(counter.items()))


def check_db(results_root: str, setting: str, route: str,
             observed: tuple[int, int, int]) -> str:
    rel, split, fold, seed = DB_SPECS[setting]
    path = os.path.join(results_root, rel)
    con = sqlite3.connect(path)
    sql = ("select distinct n_seq,n_train,n_test from runs "
           "where route=? and padding='repeat_last' and split=? and fold=?")
    params: list = [route, split, fold]
    if seed is not None:
        sql += " and seed=?"
        params.append(seed)
    expected = con.execute(sql, params).fetchall()
    con.close()
    if expected != [observed]:
        raise RuntimeError(
            f"{setting}/{route}: DB dimensions {expected} != audited {observed}")
    return rel


def check_unsw_zero_json(results_root: str, route: str, fold_id: int,
                         cell: str, n_test: int, n_zero: int) -> str:
    rel = "normal_day_fpr_v2_s20.json"
    with open(os.path.join(results_root, rel), encoding="utf-8") as f:
        payload = json.load(f)
    test_day = (0, 2)[fold_id]
    expected = {(int(r["n_test"]), int(r["n_norm"])) for r in payload["runs"]
                if r["cell"] == cell and int(r["te_day"]) == test_day}
    observed = (n_test, n_zero)
    if expected != {observed}:
        raise RuntimeError(
            f"UNSW zero/{route}/f{fold_id}: JSON dimensions {expected} != audited {observed}")
    return rel


def audit(setting: str, dataset: str, route: str, df: pd.DataFrame,
          code: np.ndarray, fam_names: list[str], b, rows_keep,
          fold: dict, flow_set: np.ndarray, test_days,
          results_root: str, val_start: int | None = None) -> dict:
    assigned = np.full(len(b.wid), -1, dtype=np.int8)
    for key, value in (("train", TRAIN), ("val", VAL), ("test", TEST),
                       ("zero", ZERO), ("norm", ZERO)):
        if key in fold:
            idx = np.asarray(fold[key], dtype=np.int64)
            if np.any(assigned[idx] >= 0):
                raise RuntimeError(f"Overlapping production fold indices in {setting}")
            assigned[idx] = value

    seq_dir = collections.Counter()
    flow_dir = collections.Counter()
    attack_dir = collections.Counter()
    family_dir = collections.Counter()
    test_positive_dest = collections.Counter()
    n_cross = n_flows = n_attack = n_label_changed = 0
    for i, rows in enumerate(rows_keep):
        fs = flow_set[rows]
        if len(np.unique(fs)) > 1:
            n_cross += 1
        a = int(assigned[i])
        if a < 0:
            continue
        foreign = fs != a
        if not foreign.any():
            continue
        actuals = np.unique(fs[foreign])
        for actual in actuals:
            seq_dir[f"{SET_NAME[a]}<-{SET_NAME[int(actual)]}"] += 1
        leaked_rows = rows[foreign]
        leaked_sets = fs[foreign]
        n_flows += len(leaked_rows)
        for actual in actuals:
            m = leaked_sets == actual
            direction = f"{SET_NAME[a]}<-{SET_NAME[int(actual)]}"
            flow_dir[direction] += int(m.sum())
            attack_mask = code[leaked_rows[m]] > 0
            n_a = int(attack_mask.sum())
            attack_dir[direction] += n_a
            n_attack += n_a
            if a in (TRAIN, VAL) and int(actual) == TEST:
                test_positive_dest[SET_NAME[a]] += n_a
            if n_a:
                fams = np.asarray(fam_names, dtype=object)[code[leaked_rows[m]][attack_mask]]
                for fam, n in collections.Counter(fams.tolist()).items():
                    family_dir[(direction, str(fam))] += n
        pos = code[rows] > 0
        if pos.any() and np.all(fs[pos] != a):
            n_label_changed += 1

    test_day_attack = int((np.isin(df["_day"].to_numpy(), list(test_days)) & (code > 0)).sum())
    n_test_leaked = sum(test_positive_dest.values())
    observed = (len(b.wid), len(fold["train"]), len(fold["test"]))
    n_zero = len(fold.get("zero", fold.get("norm", [])))
    if setting in DB_SPECS:
        db_source = check_db(results_root, setting, route, observed)
    else:
        fold_id = int(setting.rsplit("f", 1)[1])
        db_source = check_unsw_zero_json(results_root, route, fold_id,
                                         route_cfg(route, 200000).tag(), len(fold["test"]), n_zero)
    row = dict(
        setting=setting, dataset=dataset, route=ROUTE_LABEL[route],
        n_seq_total=len(b.wid), n_train=len(fold["train"]),
        n_val=len(fold["val"]), n_test=len(fold["test"]),
        n_zero=n_zero, n_seq_cross=n_cross,
        cross_by_boundary=fmt_counts(seq_dir),
        n_flows_leaked=n_flows, flows_leaked_by_boundary=fmt_counts(flow_dir),
        n_attack_flows_leaked=n_attack,
        attack_flows_leaked_by_boundary=fmt_counts(attack_dir),
        attack_families_leaked=fmt_counts(family_dir, family=True),
        n_seq_label_changed=n_label_changed,
        share_of_test_positives=(n_test_leaked / test_day_attack if test_day_attack else np.nan),
        share_test_positives_by_destination=(
            "; ".join(f"{k}: {v}" for k, v in sorted(test_positive_dest.items()))
            if test_positive_dest else "none"),
        n_test_day_attack_flows=test_day_attack,
        val_start_wid=("" if val_start is None else int(val_start)),
        db_check="PASS", db_source=db_source)
    print(f"  {setting:<10} {ROUTE_LABEL[route]:<7} n={len(b.wid):,} cross={n_cross:,} "
          f"leaked_flows={n_flows:,} test_pos_leaked={n_test_leaked:,}/{test_day_attack:,} "
          f"DB=PASS", flush=True)
    return row


def s2_test_family_rows(df, code, fam_names, b, rows_keep, fold, flow_set, route):
    day = df["_day"].to_numpy()
    fam_array = np.asarray(fam_names, dtype=object)
    s0_train_families = set(fam_array[code[(day <= 2) & (code > 0)]])
    s2_train_families = set(fam_array[code[(day <= 3) & (code > 0)]])
    test_families = set(fam_array[code[(day == 4) & (code > 0)]])
    newly_seen = sorted((s2_train_families - s0_train_families) & test_families)
    newly_seen_codes = {i for i, fam in enumerate(fam_names) if fam in newly_seen}
    assigned = np.full(len(b.wid), -1, dtype=np.int8)
    assigned[fold["train"]] = TRAIN
    assigned[fold["val"]] = VAL
    assigned[fold["test"]] = TEST
    out = []
    for i, rows in enumerate(rows_keep):
        if assigned[i] < 0:
            continue
        attack_codes = code[rows][code[rows] > 0]
        if not len(attack_codes):
            continue
        vals, cnts = np.unique(attack_codes, return_counts=True)
        sequence_family_code = int(vals[int(np.argmax(cnts))])
        if sequence_family_code not in newly_seen_codes:
            continue
        mask = np.isin(code[rows], list(newly_seen_codes))
        fams = fam_array[code[rows[mask]]]
        out.append(dict(
            route=ROUTE_LABEL[route], seq_index=i, assigned=SET_NAME[int(assigned[i])],
            sequence_day=int(b.day[i]), sequence_wid=int(b.wid[i]),
            sequence_family=fam_names[sequence_family_code],
            n_flows_kept=len(rows), n_newly_seen_family_flows=int(mask.sum()),
            newly_seen_families=";".join(sorted(set(fams.tolist()))),
            flow_days=";".join(str(x) for x in sorted(set(day[rows].tolist()))),
            flow_sets=";".join(SET_NAME[int(x)] for x in sorted(set(flow_set[rows].tolist()))),
            is_cross_day=int(len(np.unique(day[rows])) > 1),
            is_cross_flow_set=int(len(np.unique(flow_set[rows])) > 1),
            sequence_positive=int(b.y_bin[i])))
    return out, newly_seen


def audit_unsw_zero(root: str, results_root: str) -> list[dict]:
    print("\n===== Loading unsw_zero (normal_day_fpr_v2 production path) =====", flush=True)
    df, code, fam_names = load_data(root, "unsw", [0, 1, 2])
    out = []
    for route in ROUTES:
        print(f"\n--- unsw_zero/{route} ---", flush=True)
        cfg = route_cfg(route, 200000)
        b, kept, tsec = build_meta(df, cfg, code, fam_names, max_seq=0)
        for fold_id, test_day in enumerate((0, 2)):
            base = ND.build_fold(b, test_day, 1)
            train = W.RP.cap_train_indices(b.y_bin, base["train"], 80000, seed=0)
            fold = dict(train=train, val=base["val"], test=base["test"], norm=base["norm"])
            val_start = int(np.min(b.wid[base["val"]]))
            train_day = 2 if test_day == 0 else 0
            fs = temporal_flow_sets(df, tsec, [train_day], [test_day],
                                    val_start=val_start, zero_days=[1])
            out.append(audit(f"unsw_zero_f{fold_id}", "NF-UNSW-NB15", route,
                             df, code, fam_names, b, kept, fold, fs, [test_day],
                             results_root, val_start=val_start))
        del b, kept
        gc.collect()
    del df, code
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..",
                    help="Parent containing protocol_robust_nids_20261028 and nids_audit")
    ap.add_argument("--results-root", default=os.path.join(_ROOT, "results_v2"))
    ap.add_argument("--output-dir", default=os.path.join(_ROOT, "results_v2"))
    ap.add_argument("--only-unsw-zero", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rows, s2_details = [], []

    groups = [
        ("cic17", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], 0, 0),
        ("cic18", [0, 1, 2, 3, 4], 200000, 0),
        ("cic18_e1", [0, 1, 2, 3, 7, 8], 200000, 0),
        ("unsw", [0, 2], 200000, 80000),
    ]
    if args.only_unsw_zero:
        groups = []
    for group_name, days, max_segments, max_seq in groups:
        dataset = "cic18" if group_name == "cic18_e1" else group_name
        print(f"\n===== Loading {group_name} =====", flush=True)
        df, code, fam_names = load_data(args.root, dataset, days)
        for route in ROUTES:
            print(f"\n--- {group_name}/{route} ---", flush=True)
            cfg = route_cfg(route, max_segments)
            b, kept, tsec = build_meta(df, cfg, code, fam_names, max_seq=max_seq)

            if group_name == "cic17":
                fold = seqdiff.make_split(b, "cross_day", seed=0)[0]
                fs = temporal_flow_sets(df, tsec, [0, 1, 2], [4])
                fs[df["_day"].to_numpy() == 3] = VAL
                rows.append(audit("main_cic17", "CIC-IDS2017", route, df, code,
                                  fam_names, b, kept, fold, fs, [4], args.results_root))
                fold_nd = SP.make_split_v2(b, "cross_day_nd", seed=0)[0]
                fs_nd = temporal_flow_sets(df, tsec, [1, 2], [4], zero_days=[0])
                fs_nd[df["_day"].to_numpy() == 3] = VAL
                rows.append(audit("nd_cic17", "CIC-IDS2017", route, df, code,
                                  fam_names, b, kept, fold_nd, fs_nd, [4], args.results_root))

            elif group_name == "cic18":
                fold = seqdiff.make_split(b, "cross_day", seed=0)[0]
                fs = temporal_flow_sets(df, tsec, [0, 1, 2], [4])
                fs[df["_day"].to_numpy() == 3] = VAL
                rows.append(audit("main_cic18", "CSE-CIC-IDS2018", route, df, code,
                                  fam_names, b, kept, fold, fs, [4], args.results_root))
                fold_s2 = SP.make_split_v2(b, "cross_day_ext", seed=0)[0]
                val_start = int(np.min(b.wid[fold_s2["val"]]))
                fs_s2 = temporal_flow_sets(df, tsec, [0, 1, 2, 3], [4], val_start=val_start)
                rows.append(audit("s2_cic18", "CSE-CIC-IDS2018", route, df, code,
                                  fam_names, b, kept, fold_s2, fs_s2, [4],
                                  args.results_root, val_start=val_start))
                detail, newly_seen = s2_test_family_rows(
                    df, code, fam_names, b, kept, fold_s2, fs_s2, route)
                s2_details.extend(detail)
                by_set = collections.Counter(x["assigned"] for x in detail)
                print(f"    S2 newly seen test families={newly_seen}; sequences={dict(by_set)}",
                      flush=True)

            elif group_name == "cic18_e1":
                for arm in ("inf_U", "inf_S", "inf_P"):
                    spec = E1.ARMS[arm]
                    base = E1.base_split(b, spec["train"], spec["test"])
                    fold = base
                    flow_train_days = spec["train"]
                    if arm == "inf_P":
                        u = E1.base_split(b, E1.ARMS["inf_U"]["train"], spec["test"])
                        p_u = float(b.y_bin[u["train"]].mean())
                        p_s = float(b.y_bin[base["train"]].mean())
                        if p_s > p_u:
                            side, target = "S", p_u
                        else:
                            side, target = "U", p_s
                            base = u
                        flow_train_days = E1.ARMS["inf_U" if side == "U" else "inf_S"]["train"]
                        tr_s, _, _ = E1.downsample_extra_day(
                            b, base["train"], 3 if side == "U" else 7, target,
                            np.random.default_rng(0))
                        fold = dict(train=tr_s, val=base["val"], test=base["test"])
                        print(f"    E1-P side={side} pU={p_u:.6f} pS={p_s:.6f}", flush=True)
                    val_start = int(np.min(b.wid[base["val"]]))
                    fs = temporal_flow_sets(df, tsec, flow_train_days, [spec["test"]],
                                            val_start=val_start)
                    rows.append(audit("e1_" + arm[-1], "CSE-CIC-IDS2018", route,
                                      df, code, fam_names, b, kept, fold, fs,
                                      [spec["test"]], args.results_root, val_start=val_start))

            else:
                folds = seqdiff.make_split(b, "cross_day_pair", seed=0)
                days_present = sorted(set(b.day.tolist()))
                if len(folds) != 2:
                    raise RuntimeError(f"Expected two UNSW folds, got {len(folds)}")
                for fold_id, fold in enumerate(folds):
                    test_day = days_present[fold_id]
                    train_day = days_present[1 - fold_id]
                    val_start = int(np.min(b.wid[fold["val"]]))
                    fs = temporal_flow_sets(df, tsec, [train_day], [test_day],
                                            val_start=val_start)
                    rows.append(audit(f"unsw_f{fold_id}", "NF-UNSW-NB15", route,
                                      df, code, fam_names, b, kept, fold, fs,
                                      [test_day], args.results_root, val_start=val_start))
            del b, kept
            gc.collect()
        del df, code
        gc.collect()

    if not args.only_unsw_zero:
        if len(rows) != 27:
            raise RuntimeError(f"Expected 27 audit rows, got {len(rows)}")
        out = pd.DataFrame(rows)
        path = os.path.join(args.output_dir, "A1_cross_day_audit.csv")
        out.to_csv(path, index=False)
        detail_path = os.path.join(args.output_dir, "A1_s2_test_family_sequences.csv")
        pd.DataFrame(s2_details).to_csv(detail_path, index=False)
        print(f"\nWrote {path} ({len(out)} rows; all DB checks passed)")
        print(f"Wrote {detail_path} ({len(s2_details)} rows)")

    zero_rows = audit_unsw_zero(args.root, args.results_root)
    if len(zero_rows) != 6:
        raise RuntimeError(f"Expected six UNSW zero-day audit rows, got {len(zero_rows)}")
    zero_path = os.path.join(args.output_dir, "A1_unsw_zero_day_audit.csv")
    pd.DataFrame(zero_rows).to_csv(zero_path, index=False)
    print(f"Wrote {zero_path} ({len(zero_rows)} rows; production JSON checks passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
