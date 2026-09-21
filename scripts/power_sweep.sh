#!/usr/bin/env bash
# Bring the project's central comparison to adequate power.
#
# CfC against the parameter-matched LSTM, at the pre-registered configuration, on test:
#   Pearson r   +0.1254 (n=9)  vs +0.0569 (n=11)   d=+0.78  p=0.075
#   macro MAE   14.37    (n=10) vs 13.43   (n=11)   d=+0.74  p=0.099
# Both trend the same way and neither is significant, because seed-to-seed sigma is 1.60 deg
# on macro MAE and n is about 10. For d=0.78 at 80% power the requirement is ~26 seeds per
# arm. Each run is roughly three minutes, so the fix is cheap and it is the experiment that
# actually answers the research question.
#
# Seeds 10-34, added to the existing 0-9. Nothing else changes: this is the pre-registered
# configuration, not a new condition.
#
# Concurrency 2, OMP capped, niced: this host also serves JupyterHub, an LLM proxy and a RAG
# app, and each worker pins ~3 GB of host RAM that cannot page out.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for S in $(seq 10 34); do
  for ARM in cfc lstm; do
    D="runs_power/${ARM}_s${S}_T16_lr0.001_k0.1_aug-full"
    [ -f "$D/final_metrics.json" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
    ( nice -n 10 $PY train.py --arm "$ARM" --seed "$S" --T 16 --lr 1e-3 --k 0.1 --aug full \
        --epochs 30 --processed data/processed --runs runs_power --device cuda \
        --min-free-gb 3.0 > "logs_power_${ARM}_s${S}.log" 2>&1 \
        || echo "FAIL $ARM $S" >> /tmp/power_fail.txt ) &
  done
done
wait

for D in runs_power/*/; do
  [ -f "$D/checkpoints/best.pt" ] || continue
  for SP in val test; do
    [ -f "$D/final_metrics_$SP.json" ] && continue
    nice -n 10 $PY evaluate.py "$D" --split "$SP" --save-predictions \
      --processed data/processed >/dev/null 2>&1
  done
done
echo "power sweep done"
