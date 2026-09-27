#!/usr/bin/env bash
# The exploratory Transformer time-code ablation (docs/explore_2026-09-27_transformer_time_code.md):
# the minimal Transformer with compacted indices ("index") and with no code ("none"), each with
# the registered budget, into d4/dinov2_tcode_<mode>/. The "real" variant is the existing 9.1 run.
# Validation only. Every epoch is checkpointed, so rerunning this resumes where it stopped.
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
grid_one() {
  nice -n 10 "$HOME/miniconda/envs/DL/bin/python" -W ignore d4_run.py --cache cache/dinov2_10hz \
      --arm transformer --stage lr --lrs "$2" --tag "_$2" --out "d4/dinov2_tcode_$1" --time-code "$1" \
      >> "d4/dinov2_tcode_$1/logs/grid_$2.log" 2>&1
}
export -f grid_one
for mode in index none; do
  (
    mkdir -p "d4/dinov2_tcode_$mode/logs"
    printf '%s\n' 1e-3 3e-4 1e-4 3e-5 | xargs -P 4 -I{} bash -c "grid_one $mode {}"
    echo "$(date '+%F %T') $mode grid done"
    D4_OUT="d4/dinov2_tcode_$mode" D4_EXTRA_ARGS="--time-code $mode" \
        bash run_d4_final.sh transformer "0 1 2 3" "4 5 6 7"
    echo "$(date '+%F %T') $mode FINAL DONE"
  ) &
done
wait
echo "$(date '+%F %T') TCODE ALL DONE"
