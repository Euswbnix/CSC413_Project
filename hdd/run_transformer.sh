#!/usr/bin/env bash
# The supplementary Transformer (docs/preregistration_2026-09-22.md 9.1): the four-point
# learning-rate grid at seed 0, then seeds 0-7 at the chosen rate, same budget as the other arms.
# Every epoch is checkpointed, so rerunning this after a shutdown resumes where it stopped.
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
grid_one() {
  nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore d4_run.py --cache cache/dinov2_10hz \
      --arm transformer --stage lr --lrs "$1" --tag "_$1" --out d4/dinov2 \
      >> "d4_transformer_lr_$1.log" 2>&1
}
export -f grid_one
printf '%s\n' 1e-3 3e-4 1e-4 3e-5 | xargs -P 4 -I{} bash -c 'grid_one {}'
echo "$(date '+%F %T') Transformer grid done"
bash run_d4_final.sh transformer 0 1 2 3 4 5 6 7
echo "$(date '+%F %T') Transformer FINAL DONE"
