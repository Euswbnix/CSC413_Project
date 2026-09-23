#!/usr/bin/env bash
# Restart the three unfinished LTC grid points with checkpointing; rerunning this resumes them.
umask 077
cd "$HOME/data/hdd"
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/data/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
for lr in 3e-4 1e-4 3e-5; do
  setsid nohup nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore d4_run.py \
      --cache cache/dinov2_10hz --arm ltc --stage lr --lrs "$lr" --tag "_$lr" --out d4/dinov2 \
      >> "d4_ltc_lr_$lr.log" 2>&1 < /dev/null &
done
