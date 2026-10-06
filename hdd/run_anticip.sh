#!/usr/bin/env bash
# Steering anticipation (docs/plan_2026-10-05_steering_anticipation.md): for each of the five arms,
# the four-point learning-rate grid at seed 0 and then seeds 0-7 at the chosen rate, into
# d4/anticip/. The four fast arms share one branch; LTC (24 sub-steps, compiled cell) has its own.
# Validation only. Every epoch is checkpointed, so rerunning this resumes where it stopped.
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
export D4_OUT=d4/anticip D4_RUNNER=anticip_run.py
mkdir -p "$D4_OUT/logs"
grid_one() {
  local extra=""; [ "$1" = ltc ] && extra="--ode-unfolds 24 --compile-ltc"
  nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore anticip_run.py --cache cache/dinov2_10hz \
      --arm "$1" --stage lr --lrs "$2" --tag "_$2" --out "$D4_OUT" $extra \
      >> "$D4_OUT/logs/grid_$1_$2.log" 2>&1
}
export -f grid_one
(
  for arm in frame lstm cfc transformer; do for lr in 1e-3 3e-4 1e-4 3e-5; do echo "$arm $lr"; done; done \
      | xargs -P 4 -L1 bash -c 'grid_one "$0" "$1"'
  echo "$(date '+%F %T') fast arms grid done"
  for arm in frame lstm cfc transformer; do
    bash run_d4_final.sh $arm "0 1 2 3" "4 5 6 7"
  done
  echo "$(date '+%F %T') fast arms FINAL DONE"
) &
(
  printf '%s\n' 1e-3 3e-4 1e-4 3e-5 | xargs -P 4 -I{} bash -c 'grid_one ltc {}'
  echo "$(date '+%F %T') LTC grid done"
  D4_EXTRA_ARGS="--ode-unfolds 24 --compile-ltc" bash run_d4_final.sh ltc 0 1 2 3 4 5 6 7
  echo "$(date '+%F %T') LTC FINAL DONE"
) &
wait
echo "$(date '+%F %T') ANTICIP ALL DONE"
