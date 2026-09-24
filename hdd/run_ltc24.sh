#!/usr/bin/env bash
# LTC at 24 ODE sub-steps (the pre-registered integration check failed at 6): the learning-rate
# grid, then the eight final seeds, never more than two processes at once because each needs
# about 11.4 GB of GPU memory. Every epoch is checkpointed, so rerunning this after a shutdown
# resumes where it stopped; finished runs return immediately from their checkpoints.
umask 077
cd "$HOME/data/hdd"
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/data/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
grid_one() {
  nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore d4_run.py --cache cache/dinov2_10hz \
      --arm ltc --stage lr --lrs "$1" --tag "_$1" --out d4/dinov2 --ode-unfolds 24 >> "d4_ltc_lr_$1.log" 2>&1
}
export -f grid_one
printf '%s\n' 1e-3 3e-4 1e-4 3e-5 | xargs -P 2 -I{} bash -c 'grid_one {}'
echo "$(date '+%F %T') LTC grid done"
D4_EXTRA_ARGS="--ode-unfolds 24" bash run_d4_final.sh ltc "0 1 2 3" "4 5 6 7"
echo "$(date '+%F %T') LTC FINAL DONE"
