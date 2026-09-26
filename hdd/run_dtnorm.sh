#!/usr/bin/env bash
# The exploratory retraining of docs/explore_2026-09-26_dt_units.md: LSTM+dt, CfC and LTC (24
# sub-steps) with dt divided by the train-split median step (0.099875 s), everything else as in
# D4, into d4/dinov2_dtnorm/. LSTM and CfC run in one branch, LTC in another. Every epoch is
# checkpointed, so rerunning this after a shutdown resumes where it stopped. Validation only.
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
export D4_OUT=d4/dinov2_dtnorm UNIT=0.099875
mkdir -p "$D4_OUT/logs"
grid_one() {
  local extra=""; [ "$1" = ltc ] && extra="--ode-unfolds 24 --compile-ltc"
  nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore d4_run.py --cache cache/dinov2_10hz \
      --arm "$1" --stage lr --lrs "$2" --tag "_$2" --out "$D4_OUT" --dt-unit "$UNIT" $extra \
      >> "$D4_OUT/logs/grid_$1_$2.log" 2>&1
}
export -f grid_one
(
  for arm in lstm cfc; do for lr in 1e-3 3e-4 1e-4 3e-5; do echo "$arm $lr"; done; done \
      | xargs -P 4 -L1 bash -c 'grid_one "$0" "$1"'
  echo "$(date '+%F %T') LSTM/CfC grid done"
  D4_EXTRA_ARGS="--dt-unit $UNIT" bash run_d4_final.sh lstm "0 1 2 3" "4 5 6 7"
  D4_EXTRA_ARGS="--dt-unit $UNIT" bash run_d4_final.sh cfc "0 1 2 3" "4 5 6 7"
  echo "$(date '+%F %T') LSTM/CfC FINAL DONE"
) &
(
  printf '%s\n' 1e-3 3e-4 1e-4 3e-5 | xargs -P 4 -I{} bash -c 'grid_one ltc {}'
  echo "$(date '+%F %T') LTC grid done"
  D4_EXTRA_ARGS="--dt-unit $UNIT --ode-unfolds 24 --compile-ltc" bash run_d4_final.sh ltc 0 1 2 3 4 5 6 7
  echo "$(date '+%F %T') LTC FINAL DONE"
) &
wait
echo "$(date '+%F %T') DTNORM ALL DONE"
