#!/usr/bin/env python

import os
import sqlite3

import numpy as np
import pandas as pd

DB = "results_v2/det_unsw_s20.db"
OUT = "results_v2/r3/R6_early_stopping.csv"
PATIENCE = 5
MAX_EPOCHS = 20
SEQ = ["lstm", "tcn", "transformer"]

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
d = pd.read_sql_query("select fold, model, cell, seed, epochs from runs", con)
con.close()

rows = []
for fold in [0, 1]:
    for model in SEQ:
        g = d[(d.fold == fold) & (d.model == model)]
        epochs = g.epochs.to_numpy(int)
        early_best = sorted(int(e - PATIENCE) for e in epochs if e < MAX_EPOCHS)
        n = len(epochs)
        if n % 2:
            median_idx = [n // 2]
        else:
            median_idx = [n // 2 - 1, n // 2]

        assert max(median_idx) < len(early_best)
        best_median = float(np.mean([early_best[i] for i in median_idx]))
        n_full20 = int((epochs == MAX_EPOCHS).sum())
        rows.append(dict(
            dataset="NF-UNSW-NB15", fold=fold, model=model, n_runs=n,
            n_cells=g.cell.nunique(), n_seeds=g.seed.nunique(),
            best_epoch_median=best_median,
            best_epoch_min=min(early_best),
            best_epoch_max_lo=15 if n_full20 else max(early_best),
            best_epoch_max_hi=20 if n_full20 else max(early_best),
            best_epoch_eq1_n=early_best.count(1),
            best_epoch_eq1_frac=early_best.count(1) / n,
            n_epochs_median=float(np.median(epochs)),
            n_epochs_min=int(epochs.min()), n_epochs_max=int(epochs.max()),
            n_full20=n_full20, full20_frac=n_full20 / n,
            n_early=n - n_full20,
            source_db=DB, source_table="runs",
        ))

out = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
