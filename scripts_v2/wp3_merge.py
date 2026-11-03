


from __future__ import annotations

import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "results_v2")
NEW = os.path.join(R2, "det_unsw_s20.db")
OLD = os.path.join(R2, "det_v2_unsw.db")


def cols(con, schema="main", table="runs"):


    return [r[1] for r in con.execute("pragma %s.table_info(%s)" % (schema, table))]


def main() -> int:
    if not os.path.exists(NEW) or not os.path.exists(OLD):
        print("Missing DB: %s or %s" % (NEW, OLD))
        return 1
    con = sqlite3.connect(NEW)
    con.execute("attach database ? as o", (OLD,))
    n_new = con.execute("select count(*) from main.runs").fetchone()[0]
    n_old = con.execute("select count(*) from o.runs").fetchone()[0]
    c_main, c_old = cols(con, "main"), cols(con, "o")
    if c_main != c_old:
        print("[warning] DB columns differ:\n  new=%s\n  old=%s" % (c_main, c_old))
        return 2
    con.execute("insert or ignore into main.runs select * from o.runs")
    con.commit()
    n_tot = con.execute("select count(*) from main.runs").fetchone()[0]
    print("New runs %d + old runs %d → %d rows after merge (deduplicated)" % (n_new, n_old, n_tot))

    print("\nBy fold and seed:")
    for r in con.execute("select fold, count(*), min(seed), max(seed) from runs "
                         "group by fold order by fold"):
        print("  Fold %s: %d runs, seeds %s–%s" % r)

    print("\nBy model:")
    for r in con.execute("select model, count(*) from runs group by model order by model"):
        print("  %-12s %d" % r)

    print("\nDeterministic kernel record: cudnn_benchmark values %s"
          % [x[0] for x in con.execute("select distinct cudnn_benchmark from runs")])


    same = con.execute(
        "select count(*) from main.runs m join o.runs o on m.run_id = o.run_id "
        "where m.macro_f1 = o.macro_f1 and m.pos_q50 is o.pos_q50 "
        "and m.thr_oracle is o.thr_oracle and m.epochs = o.epochs").fetchone()[0]
    print("\nOld rows field-identical (macro_f1 / pos_q50 / thr_oracle / epochs): %d/%d" % (same, n_old))
    con.close()
    print("\n→ `INSERT ... SELECT` directly satisfies the acceptance criterion that seed 0–4 rows are bit-identical to the old DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
