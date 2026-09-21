#!/usr/bin/env bash
# Waits for the field-of-view training to finish, then evaluates every run under the
# rollout protocol on both splits. Each condition must be evaluated against ITS OWN
# processed directory -- the pixels differ, so pointing all of them at data/processed
# would silently score the fov200/fov256 weights on the 150-row images they never saw.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

while pgrep -f "fov_experiment.sh" > /dev/null; do sleep 60; done

for R in 200 256; do
  for D in runs_fov$R/*/; do
    [ -f "$D/checkpoints/best.pt" ] || continue
    for S in val test; do
      nice -n 10 $PY evaluate.py "$D" --split "$S" --save-predictions \
        --processed "data/processed_fov$R" >/dev/null 2>&1
    done
  done
done
echo "fov evaluation done"
