


from __future__ import annotations

import numpy as np


def block_ids(b, block_minutes: int = 10) -> np.ndarray:

    return (np.asarray(b.wid) // int(block_minutes * 60)).astype(np.int64)


def split_block_strat(b, block_minutes: int = 10, ratios=(0.6, 0.2, 0.2),
                      seed: int = 0) -> list[dict]:
    blk = block_ids(b, block_minutes)
    y = np.asarray(b.y_bin).astype(bool)
    uq, inv = np.unique(blk, return_inverse=True)
    blk_pos = np.zeros(len(uq), dtype=bool)
    np.logical_or.at(blk_pos, inv, y)
    blk_cnt = np.bincount(inv, minlength=len(uq))

    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for s in (True, False):
        sel = np.where(blk_pos == s)[0]
        k = len(sel)
        if k == 0:
            continue
        rng.shuffle(sel)
        if k < 3:
            print("    [S1] stratum pos=%s has only %d blocks; assigning entire stratum to training" % (s, k))
            parts["train"].append(sel)
            continue
        cnt = blk_cnt[sel].astype(float)
        cum = np.cumsum(cnt) / cnt.sum()
        i_tr = int(np.searchsorted(cum, ratios[0]))
        i_va = int(np.searchsorted(cum, ratios[0] + ratios[1]))
        i_tr = min(max(i_tr, 0), k - 2)
        i_va = min(max(i_va, i_tr + 1), k - 1)
        parts["train"].append(sel[:i_tr + 1])
        parts["val"].append(sel[i_tr + 1:i_va + 1])
        parts["test"].append(sel[i_va + 1:])
        print("    [S1] stratum pos=%s: blocks %d → train %d / validation %d / test %d"
              % (s, k, len(parts["train"][-1]), len(parts["val"][-1]),
                 len(parts["test"][-1])))

    out = {}
    for k in ("train", "val", "test"):
        blocks = np.concatenate(parts[k]) if parts[k] else np.array([], np.int64)
        out[k] = np.where(np.isin(inv, blocks))[0]
    return [out]


def split_cross_day_ext(b, train_days=(0, 1, 2, 3), test_days=(4,),
                        val_frac: float = 0.2, seed: int = 0) -> list[dict]:
    d = np.asarray(b.day)
    tr_all = np.where(np.isin(d, np.asarray(train_days)))[0]
    order = tr_all[np.argsort(np.asarray(b.wid)[tr_all], kind="mergesort")]
    n_va = max(1, int(len(order) * val_frac))
    return [dict(train=order[:-n_va], val=order[-n_va:],
                 test=np.where(np.isin(d, np.asarray(test_days)))[0])]


def split_cross_day_nd(b, zero_days=None, seed: int = 0) -> list[dict]:


    d = np.asarray(b.day)
    days = sorted(set(d.tolist()))
    if len(days) < 4:
        raise ValueError(f"cross_day_nd requires at least 4 days; got {len(days)}")
    zero = [days[0]] if zero_days is None else list(zero_days)
    tr_days = days[1:len(days) - 2]
    va_days = [days[len(days) - 2]]
    te_days = [days[len(days) - 1]]
    fold = dict(train=np.where(np.isin(d, tr_days))[0],
                val=np.where(np.isin(d, va_days))[0],
                test=np.where(np.isin(d, te_days))[0],
                zero=np.where(np.isin(d, zero))[0])
    return [fold]


def make_split_v2(b, name: str, block_minutes: int = 10, seed: int = 0) -> list[dict]:
    if name == "block_strat":
        return split_block_strat(b, block_minutes=block_minutes, seed=seed)
    if name == "cross_day_ext":
        return split_cross_day_ext(b, train_days=(0, 1, 2, 3), test_days=(4,), seed=seed)
    if name == "cross_day_nd":
        return split_cross_day_nd(b, seed=seed)
    raise ValueError("Unknown v2 split: %s" % name)


def s0_train_pos_rate(b) -> float:

    from seqdiff import make_split
    f = make_split(b, "cross_day", seed=0)[0]
    return float(np.asarray(b.y_bin)[f["train"]].mean())
