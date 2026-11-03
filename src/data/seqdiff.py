


from __future__ import annotations

from dataclasses import dataclass, field, asdict
import os
from typing import Iterable

import numpy as np
import pandas as pd


TS_FORMAT = {
    "engelen": "%d/%m/%Y %I:%M:%S %p",
    "liu": None,
}
TS_GUARD = {
    "engelen": ("2017-07-01", "2017-07-10"),
    "liu": ("2017-07-01", "2017-07-10"),
}

IDENTITY_COLS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port",
                 "Timestamp", "source_file", "row_idx", "id"]
LABEL_COLS = ["Label", "Attempted Category"]
NON_FEATURE_COLS = set(IDENTITY_COLS) | set(LABEL_COLS)

SEGMENTS = ("time_window", "session", "service", "host")
CONVENTIONS = ("by_start", "by_end", "by_overlap")
TRUNCATES = ("head", "tail", "random")
PADDINGS = ("repeat_last", "zero_mask", "non_padded")

SESSION_COLS = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"]
SERVICE_COLS = ["Src IP", "Dst IP", "Dst Port", "Protocol"]


def load_day(path: str, version: str, columns: Iterable[str] | None = None) -> pd.DataFrame:

    if version not in TS_FORMAT:
        raise ValueError(f"Unknown version {version!r}; known versions {list(TS_FORMAT)}")

    df = pd.read_parquet(path, columns=list(columns) if columns else None)

    fmt = TS_FORMAT[version]
    ts = (pd.to_datetime(df["Timestamp"], errors="coerce") if fmt is None
          else pd.to_datetime(df["Timestamp"], format=fmt, errors="coerce"))

    n_bad = int(ts.isna().sum())
    if n_bad:
        raise ValueError(f"{path}: failed to parse timestamps in {n_bad} rows")

    lo, hi = TS_GUARD.get(version, ("1900-01-01", "2100-01-01"))
    t_min, t_max = ts.min(), ts.max()
    if not (pd.Timestamp(lo) <= t_min and t_max <= pd.Timestamp(hi)):
        raise ValueError(
            f"{path}: time range [{t_min}, {t_max}] exceeds expected [{lo}, {hi}];"
            f"day/month and month/day were likely reversed"
        )

    df["_t_start"] = ts
    dur_us = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["_t_end"] = df["_t_start"] + pd.to_timedelta(dur_us, unit="us")
    df["_t_mid"] = df["_t_start"] + pd.to_timedelta(dur_us / 2.0, unit="us")

    df = df.sort_values("_t_start", kind="mergesort").reset_index(drop=True)
    return df


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def load_days(root: str, version: str = "engelen", days=None,
              processed_subdir: str | None = None) -> pd.DataFrame:


    days = days or DAY_ORDER
    sub = processed_subdir or version
    frames = []
    for i, d in enumerate(DAY_ORDER):
        if d not in days:
            continue
        name = f"{d}-WorkingHours" if version == "engelen" else d.lower()
        p = f"{root}/nids_audit/data/processed/{sub}/{name}.parquet"
        if not os.path.exists(p):
            print(f"  [warning] Missing {p}; skipping")
            continue
        one = load_day(p, version)
        one["_day"] = DAY_ORDER.index(d)
        frames.append(one)
    if not frames:
        raise FileNotFoundError("No day files found")
    df = pd.concat(frames, ignore_index=True)

    df = df.sort_values(["_day", "_t_start"], kind="mergesort").reset_index(drop=True)
    return df


def time_span_minutes(df: pd.DataFrame) -> float:
    return float((df["_t_end"].max() - df["_t_start"].min()).total_seconds() / 60.0)


def add_group_ids(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["_sid"] = df.groupby(SESSION_COLS, sort=False).ngroup().astype(np.int64)
    df["_hid"] = df.groupby("Src IP", sort=False).ngroup().astype(np.int64)
    df["_svid"] = df.groupby(SERVICE_COLS, sort=False).ngroup().astype(np.int64)
    return df


def assign_windows(df: pd.DataFrame, convention: str, window_s: float) -> tuple[np.ndarray, dict]:

    if convention not in CONVENTIONS:
        raise ValueError(f"Unknown convention {convention!r}")

    anchor = df["_t_start"].min().floor("s")
    ws = (df["_t_start"] - anchor).dt.total_seconds().to_numpy()
    we = (df["_t_end"] - anchor).dt.total_seconds().to_numpy()

    if convention == "by_start":
        return np.floor(ws / window_s).astype(np.int64), {}
    if convention == "by_end":
        return np.floor(we / window_s).astype(np.int64), {}

    w_lo = np.floor(ws / window_s).astype(np.int64)
    w_hi = np.clip(np.floor(we / window_s).astype(np.int64), w_lo, w_lo + 4095)
    counts = w_hi - w_lo + 1
    return None, {"w_lo": w_lo, "counts": counts}


@dataclass
class SeqConfig:
    segment: str = "time_window"
    window_s: float = 30.0
    T: int = 20
    convention: str = "by_start"
    truncate: str = "tail"
    padding: str = "repeat_last"
    label_mode: str = "any"
    seed: int = 0
    max_segments: int = 0

    def tag(self) -> str:
        if self.segment == "time_window":
            return f"tw{int(self.window_s)}s/{self.convention}/{self.truncate}/{self.padding}/T{self.T}"
        return f"{self.segment}/{self.padding}/T{self.T}"


def _chunk(group_rows: np.ndarray, T: int) -> list[np.ndarray]:

    return [group_rows[i:i + T] for i in range(0, len(group_rows), T)]


def segment_rows(df: pd.DataFrame, cfg: SeqConfig) -> list[np.ndarray]:

    if cfg.segment == "time_window":
        wid, ov = assign_windows(df, cfg.convention, cfg.window_s)
        if cfg.convention == "by_overlap":

            row_idx = np.repeat(np.arange(len(df), dtype=np.int64), ov["counts"])
            starts = np.repeat(ov["w_lo"], ov["counts"])
            offset = np.arange(len(row_idx)) - np.repeat(np.cumsum(ov["counts"]) - ov["counts"], ov["counts"])
            wid = starts + offset
        order = np.argsort(wid, kind="mergesort")
        wid_sorted = wid[order]
        rows = row_idx if cfg.convention == "by_overlap" else np.arange(len(df), dtype=np.int64)
        rows_sorted = rows[order]
        _, starts = np.unique(wid_sorted, return_index=True)
        ends = np.append(starts[1:], len(wid_sorted))
        return [rows_sorted[starts[k]:ends[k]] for k in range(len(starts))]

    key = {"session": "_sid", "service": "_svid", "host": "_hid"}[cfg.segment]
    if key not in df.columns:
        raise ValueError(f"segment={cfg.segment} requires add_group_ids() first")


    order = np.argsort(df[key].to_numpy(), kind="mergesort")
    keys_sorted = df[key].to_numpy()[order]
    _, starts = np.unique(keys_sorted, return_index=True)
    ends = np.append(starts[1:], len(keys_sorted))

    out = []
    for k in range(len(starts)):
        rows = order[starts[k]:ends[k]]
        out.extend(_chunk(rows, cfg.T))
    return out


@dataclass
class SeqBatch:
    X: np.ndarray
    mask: np.ndarray
    length: np.ndarray
    y_bin: np.ndarray
    y_multi: np.ndarray
    group: np.ndarray
    wid: np.ndarray
    day: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and not c.startswith("_")
            and pd.api.types.is_numeric_dtype(df[c])]


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(a, -1e12, 1e12).astype(np.float32)


def _feature_matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:


    n, nf = len(df), len(feats)
    out = np.empty((n, nf), dtype=np.float32)
    for j, c in enumerate(feats):
        v = np.asarray(df[c].to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
        np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(v, -1e12, 1e12, out=v)
        out[:, j] = v
    return out


def build_sequences(df: pd.DataFrame, cfg: SeqConfig) -> SeqBatch:
    if cfg.segment not in SEGMENTS:
        raise ValueError(f"Unknown segmentation {cfg.segment!r}")
    if cfg.padding not in PADDINGS:
        raise ValueError(f"Unknown padding {cfg.padding!r}")
    if cfg.segment == "time_window" and cfg.truncate not in TRUNCATES:
        raise ValueError(f"Unknown overflow policy {cfg.truncate!r}")

    feats = feature_columns(df)
    F_all = _feature_matrix(df, feats)


    _lab = df["Label"]
    if pd.api.types.is_numeric_dtype(_lab):
        atk = pd.to_numeric(_lab, errors="coerce").fillna(0).to_numpy() != 0
        labels = np.where(atk, "ATTACK", "BENIGN")
    else:
        labels = _lab.astype(str).str.upper().to_numpy()
        atk = labels != "BENIGN"


    _key_col = {"session": "_sid", "service": "_svid", "host": "_hid",
                "time_window": "_sid"}[cfg.segment]
    grp_all = df[_key_col].to_numpy() if _key_col in df else None
    tsec = (df["_t_start"] - df["_t_start"].min()).dt.total_seconds().to_numpy()
    day_all = df["_day"].to_numpy() if "_day" in df.columns else None

    segs = segment_rows(df, cfg)
    rng = np.random.default_rng(cfg.seed)


    n_seg_all = len(segs)
    if cfg.max_segments and n_seg_all > cfg.max_segments:
        keep = np.sort(rng.choice(n_seg_all, size=cfg.max_segments, replace=False))
        segs = [segs[i] for i in keep]

    Xs, Ms, Ls, Yb, Ym, Gs, Ws, Ds = [], [], [], [], [], [], [], []
    mism = 0
    seg_lens = np.empty(len(segs), dtype=np.int64)
    pad_slots = 0
    total_slots = 0

    for s, rows in enumerate(segs):
        n_full = len(rows)
        seg_lens[s] = n_full

        if n_full > cfg.T:
            if cfg.truncate == "head":
                keep = rows[:cfg.T]
            elif cfg.truncate == "tail":
                keep = rows[-cfg.T:]
            elif cfg.truncate == "random":
                keep = np.sort(rng.choice(rows, size=cfg.T, replace=False))
            else:
                raise ValueError(cfg.truncate)
        else:
            keep = rows

        n = len(keep)
        x = F_all[keep]

        if n < cfg.T:
            pad_n = cfg.T - n
            if cfg.padding == "repeat_last":
                x = np.vstack([x, np.repeat(x[-1:], pad_n, axis=0)])
            elif cfg.padding == "zero_mask":
                x = np.vstack([x, np.zeros((pad_n, x.shape[1]), dtype=np.float32)])
            else:
                x = np.vstack([x, np.zeros((pad_n, x.shape[1]), dtype=np.float32)])
            pad_slots += pad_n
        total_slots += cfg.T


        if cfg.padding == "repeat_last":
            m = np.ones(cfg.T, dtype=bool)
        else:
            m = np.zeros(cfg.T, dtype=bool)
            m[:n] = True

        a_keep, a_full = atk[keep], atk[rows]
        if cfg.label_mode == "any":
            yb = int(a_keep.any())
            ym = labels[keep][a_keep][0] if a_keep.any() else "BENIGN"
            seg_yb = int(a_full.any())
            seg_ym = labels[rows][a_full][0] if a_full.any() else "BENIGN"
        else:
            yb = int(a_keep.mean() >= 0.5)
            v, c = np.unique(labels[keep], return_counts=True)
            ym = str(v[int(np.argmax(c))])
            seg_yb = int(a_full.mean() >= 0.5)
            v, c = np.unique(labels[rows], return_counts=True)
            seg_ym = str(v[int(np.argmax(c))])

        mism += int((yb, ym) != (seg_yb, seg_ym))

        Xs.append(x)
        Ms.append(m)
        Ls.append(n_full)
        Yb.append(yb)
        Ym.append(ym)
        g = grp_all[keep] if grp_all is not None else np.array([-1])

        v, c = np.unique(g, return_counts=True)
        Gs.append(int(v[int(np.argmax(c))]))
        Ws.append(int(tsec[keep].min()))
        Ds.append(int(day_all[keep].min()) if day_all is not None else -1)

    X = np.stack(Xs) if Xs else np.zeros((0, cfg.T, len(feats)), np.float32)
    Ls_a = np.asarray(Ls, np.int64)
    Yb_a = np.asarray(Yb, np.int8)
    Ym_a = np.asarray(Ym, object)
    Gs_a = np.asarray(Gs, np.int64)
    Ws_a = np.asarray(Ws, np.int64)
    Ds_a = np.asarray(Ds, np.int64)
    Ms_a = np.stack(Ms) if Ms else np.zeros((0, cfg.T), bool)
    seg_lens = np.asarray(seg_lens, np.int64)

    filtered_out = 0
    if cfg.padding == "non_padded":


        keep = Ls_a >= cfg.T
        filtered_out = int((~keep).sum())
        X, Ms_a, Ls_a, Yb_a, Ym_a, Gs_a, Ws_a, Ds_a = (
            X[keep], Ms_a[keep], Ls_a[keep], Yb_a[keep], Ym_a[keep], Gs_a[keep],
            Ws_a[keep], Ds_a[keep])
    n_seq = len(Ls_a)

    meta = dict(asdict(cfg), tag=cfg.tag(), n_features=len(feats), feature_names=feats,
                n_sequences=n_seq, n_features_raw=len(feats))
    meta["pad_fraction"] = float(pad_slots / total_slots) if total_slots else 0.0
    meta["short_segment_rate"] = float((seg_lens < cfg.T).mean()) if n_seq else 0.0
    meta["overflow_rate"] = float((seg_lens > cfg.T).mean()) if n_seq else 0.0
    meta["mean_seg_len"] = float(seg_lens.mean()) if n_seq else 0.0
    meta["median_seg_len"] = float(np.median(seg_lens)) if n_seq else 0.0
    meta["p90_seg_len"] = float(np.quantile(seg_lens, .9)) if n_seq else 0.0
    meta["max_seg_len"] = int(seg_lens.max()) if n_seq else 0
    meta["pos_rate"] = float(np.mean(Yb_a)) if n_seq else 0.0
    meta["label_input_mismatch"] = float(mism / (n_seq + filtered_out)) if (n_seq + filtered_out) else 0.0
    meta["filtered_out"] = filtered_out
    meta["filtered_out_rate"] = float(filtered_out / (n_seq + filtered_out)) if (n_seq + filtered_out) else 0.0


    meta["n_segments_all"] = int(n_seg_all)
    meta["segment_sampling_rate"] = float(len(segs) / n_seg_all) if n_seg_all else 1.0
    if cfg.segment == "time_window" and cfg.convention == "by_overlap" and n_seq:
        meta["dup_rate"] = float((sum(len(r) for r in segs) - len(df)) / len(df))

    return SeqBatch(X=X, mask=Ms_a, length=Ls_a, y_bin=Yb_a, y_multi=Ym_a,
                    group=Gs_a, wid=Ws_a, day=Ds_a, meta=meta)


def make_split(b: SeqBatch, strategy: str, n_folds: int = 8,
               ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
               seed: int = 0) -> list[dict]:
    n = len(b.wid)
    order_t = np.argsort(b.wid, kind="mergesort")

    if strategy == "random":
        p = np.random.default_rng(seed).permutation(n)
        a, c = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        return [dict(train=p[:a], val=p[a:c], test=p[c:])]

    if strategy == "temporal":
        a, c = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        return [dict(train=order_t[:a], val=order_t[a:c], test=order_t[c:])]

    if strategy == "rolling":
        folds = []
        step = 1.0 / (n_folds + 1)
        for k in range(n_folds):
            cut_tr = int(n * (1 - step * (n_folds - k)))
            cut_te = int(n * (1 - step * (n_folds - k - 1)))
            cut_va = int(cut_tr * 0.85)
            te = order_t[cut_tr:cut_te]
            if len(te) == 0:
                continue
            folds.append(dict(train=order_t[:cut_va], val=order_t[cut_va:cut_tr], test=te))
        return folds

    if strategy in ("group_key", "group_session", "group_host"):


        key = np.asarray(b.group)
        uniq = np.unique(key)
        np.random.default_rng(seed).shuffle(uniq)
        a, c = int(len(uniq) * ratios[0]), int(len(uniq) * (ratios[0] + ratios[1]))
        return [{k: np.where(np.isin(key, s))[0]
                 for k, s in (("train", uniq[:a]), ("val", uniq[a:c]), ("test", uniq[c:]))}]

    if strategy == "cross_day":


        if b.day is None or (np.asarray(b.day) < 0).all():
            raise ValueError("cross_day requires df to contain _day (load with load_days)")
        d = np.asarray(b.day)
        days = sorted(set(d.tolist()))
        if len(days) < 3:
            raise ValueError(f"cross_day requires at least 3 days; got {len(days)}")
        n_tr = max(1, len(days) - 2)
        tr_days, va_days, te_days = days[:n_tr], [days[n_tr]], days[n_tr + 1:]
        return [dict(train=np.where(np.isin(d, tr_days))[0],
                     val=np.where(np.isin(d, va_days))[0],
                     test=np.where(np.isin(d, te_days))[0])]

    if strategy == "cross_day_pair":


        if b.day is None or (np.asarray(b.day) < 0).all():
            raise ValueError("cross_day_pair requires df to contain _day")
        d = np.asarray(b.day)
        days = sorted(set(d.tolist()))
        folds = []
        for te_d in days:
            te = np.where(d == te_d)[0]
            if b.y_bin[te].sum() < 5 or int((b.y_bin[te] == 0).sum()) < 5:
                continue
            tr_all = np.where(d != te_d)[0]
            if b.y_bin[tr_all].sum() < 5:
                continue

            order = tr_all[np.argsort(np.asarray(b.wid)[tr_all], kind="mergesort")]
            n_va = max(1, int(len(order) * 0.2))
            folds.append(dict(train=order[:-n_va], val=order[-n_va:], test=te))
        if not folds:
            raise ValueError("cross_day_pair has no valid folds (every day lacks a class)")
        return folds

    if strategy == "strat_group":


        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        parts = list(sgkf.split(np.zeros(n), b.y_bin, groups=np.asarray(b.group)))
        if len(parts) < 5:
            raise ValueError("Insufficient folds for StratifiedGroupKFold")
        train = np.concatenate([parts[i][0] for i in range(3)])
        val = parts[3][1]
        test = parts[4][1]

        excl = np.concatenate([val, test])
        train = np.setdiff1d(train, excl)
        return [dict(train=train, val=val, test=test)]

    raise ValueError(strategy)


def split_leakage(b: SeqBatch, fold: dict) -> dict:
    g = np.asarray(b.group)
    tr = set(g[fold["train"]].tolist())
    te = g[fold["test"]]
    hit = np.array([x in tr for x in te], bool) if len(te) else np.zeros(0, bool)
    pos = b.y_bin[fold["test"]] == 1
    return dict(
        group_overlap_rate=float(hit.mean()) if len(hit) else 0.0,
        group_overlap_rate_attack=float(hit[pos].mean()) if pos.sum() else 0.0,
        n_train=int(len(fold["train"])), n_val=int(len(fold["val"])), n_test=int(len(fold["test"])),
        test_pos_rate=float(pos.mean()) if len(pos) else 0.0,
    )


CIC18_TS_GUARD = ("2018-02-01", "2018-03-15")


def load_parquet_dir(path: str, ts_guard=CIC18_TS_GUARD, days=None,
                     downcast: bool = True) -> pd.DataFrame:


    import glob

    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {path}")

    if days is not None:


        keep = set(int(d) for d in days)
        picked = []
        for p in files:
            d = int(pd.read_parquet(p, columns=["_day"])["_day"].iloc[0])
            if d in keep:
                picked.append((d, p))
        if not picked:
            raise ValueError(f"days={sorted(keep)} matched no parquet files")
        picked.sort()
        files = [p for _, p in picked]

    frames = []
    for p in files:
        df = pd.read_parquet(p)


        df = df.drop(columns=[c for c in ("Flow ID", "source_file") if c in df.columns])
        if downcast:
            for c in df.columns:
                if c == "Timestamp" or c.startswith("_"):
                    continue
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], downcast="float")
                elif pd.api.types.is_integer_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], downcast="integer")
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
        n_bad = int(ts.isna().sum())
        if n_bad:
            raise ValueError(f"{p}: invalid timestamps in {n_bad} rows")
        lo, hi = ts_guard
        if not (pd.Timestamp(lo) <= ts.min() and ts.max() <= pd.Timestamp(hi)):
            raise ValueError(f"{p}: time range [{ts.min()}, {ts.max()}] exceeds expected [{lo}, {hi}]")
        df["_t_start"] = ts
        dur = pd.to_numeric(df["Flow Duration"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df["_t_end"] = df["_t_start"] + pd.to_timedelta(dur, unit="us")
        df["_t_mid"] = df["_t_start"] + pd.to_timedelta(dur / 2.0, unit="us")
        if "_day" not in df.columns:
            df["_day"] = len(frames)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    del frames
    out = out.sort_values(["_day", "_t_start"], kind="mergesort").reset_index(drop=True)
    deep_gb = out.memory_usage(deep=True).sum() / 1e9
    print(f"  [load_parquet_dir] {len(out):,} rows × {len(out.columns)} columns, "
          f"DataFrame resident size {deep_gb:.1f} GB", flush=True)
    return out
