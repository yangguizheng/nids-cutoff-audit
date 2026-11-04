#!/bin/bash
# WP3: extend UNSW to 20 seeds (protocol §WP3), starting automatically after WP2d finishes.
# 1) Seeds 5–19, two folds, 9 cells x 5 models = 1,350 runs -> det_unsw_s20.db (with WP2e fields)
# 2) Merge the original 450 runs unchanged, preserving seeds 0–4 bit for bit
# 3) Extend the zero-attack-day (unsw_nd) evaluation to 20 seeds
# 4) Recompute the UNSW Table 4 win-rate rows, collapse rates, and ptr -> wp3_unsw_compare.csv
set -u
cd "$(dirname "$0")/.." || exit 1
PY=./env/py/bin/python
LOG=logs/queue_wp3.log
say() { echo "[3 $(date +%H:%M)] $*" | tee -a "$LOG"; }

while ! grep -q $'WP2d \u5168\u90e8\u7ed3\u675f' logs/queue_wp2d.log 2>/dev/null; do sleep 60; done
say "WP2d completion detected; starting WP3"

for f in 0 1; do
  $PY -u scripts_v2/wp2e_dump.py --dataset unsw --split cross_day_pair --fold $f \
      --rid-prefix unswdet --seeds 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
      --models lstm,tcn,transformer,rf,hgb --t-values 20 \
      --max-seq 80000 --max-segments 200000 \
      --feature-subset data/features_unsw_noTTL.json \
      --db results_v2/det_unsw_s20.db --scores-dir scores_v2_unsw_s20 --resume \
      > "logs/wp3_unsw_f${f}.log" 2>&1
  say "UNSW f$f seeds 5-19 rc=$?"
done

$PY -u scripts_v2/wp3_merge.py > logs/wp3_merge.log 2>&1
say "Merged the original 450 runs rc=$?"

$PY -u scripts_v2/normal_day_fpr_v2.py --seeds "5 6 7 8 9 10 11 12 13 14 15 16 17 18 19" \
    --out results_v2/normal_day_fpr_v2_s20_part.json > logs/wp3_nd.log 2>&1
say "Zero-attack-day seeds 5-19 rc=$?"

$PY -u scripts_v2/wp3_unsw_compare.py > logs/wp3_compare.log 2>&1
say "UNSW comparison rc=$?"
say "WP3 complete"
