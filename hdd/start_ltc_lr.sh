#!/usr/bin/env bash
# Start (or, after a shutdown, resume) the four LTC grid points at 24 ODE sub-steps (the
# pre-registered integration check failed at 6); each epoch is checkpointed.
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
for lr in 1e-3 3e-4 1e-4 3e-5; do
  setsid nohup nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore d4_run.py \
      --cache cache/dinov2_10hz --arm ltc --stage lr --lrs "$lr" --tag "_$lr" --out d4/dinov2 --ode-unfolds 24 \
      >> "d4_ltc_lr_$lr.log" 2>&1 < /dev/null &
done
