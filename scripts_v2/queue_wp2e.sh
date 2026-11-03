#!/bin/bash
set -u
cd "$(dirname "$0")/.." || exit 1
PY=./env/py/bin/python
LOG=logs/queue_wp2e.log
say() { echo "[2e $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "WP2e started"
SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

say "1/3 CIC17 (900)"
$PY scripts_v2/wp2e_dump.py --dataset cic17 --rid-prefix c17det \
    --seeds $SEEDS --models lstm,tcn,transformer,rf,hgb --t-values 20 \
    --db results_v2/det_v2_cic17.db --verify-db results/det_cic17.db --resume \
    > logs/wp2e_cic17.log 2>&1
rc=$?; say "1 finished rc=$rc"; [ $rc -ne 0 ] && exit $rc

say "2/3 CIC18 (900)"
$PY scripts_v2/wp2e_dump.py --dataset cic18 --cic18-days 0 1 2 3 4 \
    --max-segments 200000 --max-seq 0 --rid-prefix c18det \
    --seeds $SEEDS --models lstm,tcn,transformer,rf,hgb --t-values 20 \
    --db results_v2/det_v2_cic18.db --verify-db results/det_cic18.db --resume \
    > logs/wp2e_cic18.log 2>&1
rc=$?; say "2 finished rc=$rc"; [ $rc -ne 0 ] && exit $rc

for f in 0 1; do
    say "3/3 UNSW fold $f (225)"
    $PY scripts_v2/wp2e_dump.py --dataset unsw --split cross_day_pair --fold "$f" \
        --rid-prefix unswdet --seeds 0 1 2 3 4 \
        --models lstm,tcn,transformer,rf,hgb --t-values 20 \
        --max-seq 80000 --max-segments 200000 \
        --feature-subset data/features_unsw_noTTL.json \
        --db results_v2/det_v2_unsw.db --verify-db results/det_unsw.db --resume \
        > "logs/wp2e_unsw_f${f}.log" 2>&1
    rc=$?; say "3 fold $f finished rc=$rc"; [ $rc -ne 0 ] && exit $rc
done

say "WP2e completed"
