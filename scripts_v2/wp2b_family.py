


from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
ENGELEN = "../nids_audit/data/processed/engelen"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
BENIGN = "BENIGN"
BOOT_SEED = 20261101
NB = 10000
BOOT_COLS = ["score_med", "auc", "recall_05"]


DS = {
    "cic17": dict(prefix="c17det", fam_col="Label", days=[0, 1, 2, 3, 4],
                  parquet=[os.path.join(ENGELEN, "%s-WorkingHours.parquet" % n)
                           for n in DAY_NAMES], auto=False),
    "cic18": dict(prefix="c18det", fam_col="Label", days=[0, 1, 2, 3, 4],
                  parquet=None, auto=True),
    "unsw": dict(prefix="unswdet", fam_col="_Attack", days=[0, 2],
                 parquet=None, auto=True),
}


def _norm(s):
    return str(s).strip().upper()


def day_table(ds):

    cfg = DS[ds]
    out = {}
    if not cfg["auto"]:
        for i, p in enumerate(cfg["parquet"]):
            if i in cfg["days"] and os.path.exists(p):
                out[i] = p
        return dict(sorted(out.items()))
    pat = os.path.join(ROOT, "data/processed/%s/*.parquet" % ds)
    paths = sorted(glob.glob(pat))
    if not paths:
        raise SystemExit("No parquet found for %s: %s" % (ds, pat))
    for p in paths:
        d = pd.read_parquet(p, columns=["_day"])
        for v in pd.unique(d["_day"].to_numpy()):
            if int(v) in cfg["days"]:
                out[int(v)] = p
    missing = set(cfg["days"]) - set(out)
    if missing:
        raise SystemExit("%s is missing days %s (not in parquet)" % (ds, sorted(missing)))
    return dict(sorted(out.items()))


def atk_families(path, col):

    need = [col, "Label"] if col == "_Attack" else [col]
    d = pd.read_parquet(path, columns=need)
    v = d[col].map(_norm)
    if col == "_Attack":
        lab = d["Label"]
        if pd.api.types.is_numeric_dtype(lab):
            atk = pd.to_numeric(lab, errors="coerce").fillna(0).to_numpy() != 0
        else:
            atk = np.array([_norm(x) not in ("0", "BENIGN", "FALSE", "NAN")
                            for x in lab.to_numpy()])
        v = v[atk]
    return set(x for x in v.unique() if x != BENIGN)


def train_day_ids(split, all_days, test_day):
    if split == "cross_day":
        return list(all_days[:max(1, len(all_days) - 2)])
    if split == "cross_day_pair":
        return [d for d in all_days if d != test_day]
    raise SystemExit("Unexpected split: %s" % split)


def parse_rid(rid):

    p = rid.split("|")
    return dict(prefix=p[0], cell=p[1], split=p[2],
                fold=(p[3] if len(p) == 6 else ""),
                model=p[-2], seed=int(p[-1].lstrip("s")))


def cluster_boot_ci(vals, gid, stat=np.median, nb=NB, seed=BOOT_SEED):

    vals = np.asarray(vals, dtype=np.float64)
    gid = np.asarray(["_".join(map(str, g)) if isinstance(g, (tuple, list)) else str(g)
                      for g in gid])
    ok = np.isfinite(vals)
    vals, gid = vals[ok], gid[ok]
    if len(vals) < 3:
        return float("nan"), float("nan")
    uq, inv = np.unique(gid, return_inverse=True)
    buckets = [vals[inv == i] for i in range(len(uq))]
    rng = np.random.default_rng(seed)
    out = np.empty(nb)
    k = len(buckets)
    for b in range(nb):
        pick = rng.integers(0, k, k)
        out[b] = stat(np.concatenate([buckets[i] for i in pick]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    rows, pool = [], []
    for ds, cfg in DS.items():
        files = sorted(glob.glob(os.path.join(ROOT, "scores_v2", ds, "*.npz")))
        if not files:
            print("[SKIP] %s has no score files" % ds)
            continue
        dt = day_table(ds)
        all_days = sorted(dt)
        print("%s days %s → %s" % (ds, all_days, {k: os.path.basename(v) for k, v in dt.items()}))
        seen_cache = {}
        n_skip = 0
        for i, fn in enumerate(files):
            d = np.load(fn)
            rid = str(d["run_id"][0])
            m = parse_rid(rid)
            score = np.asarray(d["score"], float)
            label = np.asarray(d["label"]).astype(int)
            fam = np.asarray(d["family"]).astype(str)
            day = np.asarray(d["day"]).astype(int)
            ben = score[label == 0]
            if len(ben) == 0 or int((label == 1).sum()) == 0:
                n_skip += 1
                continue
            test_day = int(np.bincount(day - day.min()).argmax() + day.min())
            tr_days = tuple(train_day_ids(m["split"], all_days, test_day))
            if (m["split"], tr_days) not in seen_cache:
                s = set()
                for dd in tr_days:
                    s |= atk_families(dt[dd], cfg["fam_col"])
                seen_cache[(m["split"], tr_days)] = s
                print("  [Training-day families] Test day %d, training days %s → %d families: %s"
                      % (test_day, list(tr_days), len(s), sorted(s)[:8]))
            seen = seen_cache[(m["split"], tr_days)]

            pos_fams = fam[label == 1]
            for f in sorted(set(pos_fams.tolist())):
                s = score[fam == f]
                y = np.r_[np.zeros(len(ben), int), np.ones(len(s), int)]
                sc = np.r_[ben, s]
                try:
                    auc = float(roc_auc_score(y, sc))
                except ValueError:
                    auc = float("nan")
                rows.append(dict(
                    dataset=ds.upper(), cell=m["cell"], model=m["model"],
                    seed=m["seed"], fold=m["fold"],
                    route=m["cell"].split("/")[0],
                    family=f, seen_in_train=bool(f in seen),
                    n_family=int(len(s)), n_benign=int(len(ben)),
                    share_pos=float(len(s) / max(1, int((label == 1).sum()))),
                    auc=auc,
                    recall_05=float((s >= 0.5).mean()),
                    score_med=float(np.median(s)),
                    frac_gt_05=float((s > 0.5).mean()),
                    run_id=rid))


            unseen = sorted({f for f in pos_fams.tolist() if f not in seen})
            if unseen:
                mask = (label == 1) & np.isin(fam, unseen)
                su = score[mask]
                y = np.r_[np.zeros(len(ben), int), np.ones(len(su), int)]
                try:
                    au = float(roc_auc_score(y, np.r_[ben, su]))
                except ValueError:
                    au = float("nan")
                pool.append(dict(
                    dataset=ds.upper(), cell=m["cell"], model=m["model"],
                    seed=m["seed"], fold=m["fold"], route=m["cell"].split("/")[0],
                    n_unseen_fam=len(unseen), n_unseen_seq=int(mask.sum()),
                    n_benign=int(len(ben)), auc_pooled=au,
                    score_med_pooled=float(np.median(su)),
                    recall_05_pooled=float((su >= 0.5).mean())))
            if (i + 1) % 300 == 0:
                print("  %s %d/%d" % (ds, i + 1, len(files)))
        print("%s: %d runs, skipped %d runs (missing positive or benign class)" % (ds, len(files), n_skip))

    if not rows:
        print("No data")
        return 1
    df = pd.DataFrame(rows)
    p = os.path.join(R2, "wp2b_family.csv")
    df.to_csv(p, index=False)
    dp = pd.DataFrame(pool)
    pp = os.path.join(R2, "wp2b_runpool.csv")
    dp.to_csv(pp, index=False)
    print("\nWrote %s (%d rows) and %s (%d rows)" % (p, len(df), pp, len(dp)))

    L = ["# WP2b attack-family breakdown (M3: “attack family unseen on test day”)", "",
         "Generation command: `env/py/bin/python scripts_v2/wp2b_family.py`;"
         "See `NUMBERS.md` to recompute individual numbers.", "",
         "See `ANALYSIS_PLAN.md` §4 for family-label rules (mode, lexicographically smallest on ties, no normalized merging).",
         "Training days for `seen_in_train` are derived from each dataset’s **actual selected days** and split rule:",
         "CIC17 d0–d4 (`cross_day` → train on d0–d2), CIC18 d0–d4 (`--cic18-days 0 1 2 3 4`,"
         "train on d0–d2), UNSW `--unsw-days 0 2` (`cross_day_pair` → train on the other day).", "",
         "Cluster bootstrap 95%% CI: cluster = `(cell, seed, fold)`, %d replicates, seed %d." % (NB, BOOT_SEED), ""]

    for ds in ["CIC17", "CIC18", "UNSW"]:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        L += ["## %s" % ds, "",
              "| Family | Seen on training days | (run, family) records | Median family size | Median score | Median AUC "
              "| Median `recall@0.5` | Median fraction with score>0.5 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        g = sub.groupby(["family", "seen_in_train"])
        for (fam, sn), s in sorted(g, key=lambda x: (not x[0][1], x[0][0])):
            L.append("| %s | %s | %d | %d | %.4f | %.4f | %.4f | %.4f |" % (
                fam, "yes" if sn else "**no**", len(s), int(np.median(s.n_family)),
                np.median(s.score_med), np.nanmedian(s.auc),
                np.median(s.recall_05), np.median(s.frac_gt_05)))
        L.append("")

        for tag, ss in [("Unseen families", sub[~sub.seen_in_train]), ("Seen families", sub[sub.seen_in_train])]:
            if ss.empty:
                continue
            gid = list(zip(ss.cell, ss.seed, ss.fold))
            parts = []
            for c in BOOT_COLS:
                lo, hi = cluster_boot_ci(ss[c].to_numpy(), gid)
                parts.append("%s **%.4f** [%.4f, %.4f]" % (c, np.nanmedian(ss[c]), lo, hi))
            L.append("- **%s (reference: equal family weighting)**: %d families, %d (run, family) records; %s"
                     % (tag, ss.family.nunique(), len(ss), "; ".join(parts)))
        psub = dp[dp.dataset == ds]
        if len(psub.dropna(subset=["score_med_pooled"])):
            q = psub.dropna(subset=["score_med_pooled"])
            gid = list(zip(q.cell, q.seed, q.fold))
            parts = []
            for c, lab in [("score_med_pooled", "score_med"), ("auc_pooled", "auc")]:
                lo, hi = cluster_boot_ci(q[c].to_numpy(), gid)
                parts.append("%s **%.4f** [%.4f, %.4f]" % (lab, np.nanmedian(q[c]), lo, hi))
            L.append("- **Unseen families (primary: sequence-weighted)**: %d runs contain unseen families, median %d unseen sequences; %s"
                     % (len(q), int(np.median(q.n_unseen_seq)), "; ".join(parts)))
        L.append("")

        un = sub[~sub.seen_in_train]
        if len(un):
            sm, am = np.median(un.score_med), np.nanmedian(un.auc)
            L += ["Row 4 of the decision table requires “median unseen-family score < 0.5 **and** within-family AUC ≥ 0.95”."
                  "The primary estimand is **sequence-weighted** (Q12 adjudication, 2026-11-02):", ""]
            if len(dp[dp.dataset == ds].dropna(subset=["score_med_pooled"])):
                q = dp[dp.dataset == ds].dropna(subset=["score_med_pooled"])
                sm2, am2 = np.nanmedian(q.score_med_pooled), np.nanmedian(q.auc_pooled)
                lo2, hi2 = cluster_boot_ci(q.score_med_pooled.to_numpy(),
                                           list(zip(q.cell, q.seed, q.fold)))
                edge = (lo2 < 0.5 < hi2)
                L.append("- **Primary estimand (sequence-weighted)**: median score **%.4f** (%s 0.5),"
                         "95%% CI [%.4f, %.4f]; median AUC **%.4f** (%s 0.95)"
                         % (sm2, "<" if sm2 < 0.5 else "≥", lo2, hi2, am2,
                            "≥" if am2 >= 0.95 else "<"))
                if edge:
                    L.append("  - This CI **crosses 0.5**; under Q12, describe the manuscript result as"
                             "“borderline against the criterion”, **not “< 0.5”**")
            L.append("- Reference estimand (equal family weighting): median score **%.4f** (%s 0.5),"
                     "median AUC **%.4f** (%s 0.95)"
                     % (sm, "<" if sm < 0.5 else "≥", am, "≥" if am >= 0.95 else "<"))
            L.append("")
        else:
            L += ["All attack families on this dataset’s test days appeared on training days; this row provides no evidence for M3.", ""]

    p3 = os.path.join(R2, "WP2B_SUMMARY.md")
    with open(p3, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote %s" % p3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
