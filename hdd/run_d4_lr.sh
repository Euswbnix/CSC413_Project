#!/usr/bin/env bash
# Stage 1: the pre-registered learning-rate grid, one process per (arm, rate), six at a time.
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
PY="$HOME/miniconda/envs/DL/bin/python"
run_one() {
  nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore "$HOME/workspace/hdd/d4_run.py" \
      --cache cache/dinov2_10hz --arm "$1" --stage lr --lrs "$2" --tag "_$2" --out d4/dinov2 \
      2>&1 | grep -v Warning
}
export -f run_one
for arm in lstm cfc ltc; do
  for lr in 1e-3 3e-4 1e-4 3e-5; do
    printf '%s %s\n' "$arm" "$lr"
  done
done | xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"'
echo "LR STAGE DONE"
