#!/usr/bin/env bash
# Re-evaluate every collected run on the VAL split, recording BOTH protocols.
# Purpose: separate the protocol effect (windowed vs rollout) from the split effect
# (val vs test). train.py selects on windowed-val; evaluate.py reports rollout-test.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
N=6
ok=0; fail=0
for R in lab/runs_fleet/*/ lab/runs_lr/*/ lab/runs_aug/*/ lab/runs/*/; do
  [ -f "$R/checkpoints/best.pt" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do wait -n; done
  ( $PY evaluate.py "$R" --split val --save-predictions --processed data/processed \
      >"$R/val_eval.log" 2>&1 || echo "FAIL $R" >> /tmp/val_sweep_fail.txt ) &
done
wait
echo "done"
