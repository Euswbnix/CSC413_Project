#!/usr/bin/env bash
# Mechanism screen proposed by the independent review (docs/diagnosis_2026-09-20.md section 18;
# results in section 17.2):
#   CfC, seeds 0-2, {feature LayerNorm off/on} x {encoder lr 1e-3 / 1e-4}; recurrent block and
#   readout stay at 1e-3; everything else is the pre-registered configuration.
# All twelve runs train in ONE environment (this host, torch 2.8) -- including the untreated
# cell, which is the paired same-seed control the earlier interventions lacked.
# Decisions use VALIDATION only; test is not evaluated here.
#
# First job: a regression check. Retrain runs_power's cfc seed 10 with the current code at the
# default configuration; its weights must equal the original's exactly, which shows that the
# new health logging and optimiser code leave default training untouched.
#
# Concurrency 2, niced, OMP capped: this host also serves JupyterHub, an LLM proxy and a RAG app.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
COMMON="--arm cfc --T 16 --lr 1e-3 --k 0.1 --aug full --epochs 30 --processed data/processed --device cuda --min-free-gb 3.0"

launch() {  # log-name, extra args...
  local LOG=$1; shift
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
  ( nice -n 10 $PY train.py $COMMON "$@" > "logs_2x2/$LOG.log" 2>&1 \
      || echo "FAIL $LOG" >> /tmp/2x2_fail.txt ) &
}
mkdir -p logs_2x2
launch regress_s10 --seed 10 --runs runs_regress
for S in 0 1 2; do
  launch base_s$S    --seed $S --runs runs_2x2
  launch fn_s$S      --seed $S --runs runs_2x2 --feature-norm
  launch elr_s$S     --seed $S --runs runs_2x2 --encoder-lr 1e-4
  launch fn_elr_s$S  --seed $S --runs runs_2x2 --feature-norm --encoder-lr 1e-4
done
wait
echo "training done"

for D in runs_2x2/*/; do
  nice -n 10 $PY evaluate.py "$D" --split val --save-predictions --processed data/processed \
    > "$D/eval_val.log" 2>&1 || echo "EVAL FAIL $D" >> /tmp/2x2_fail.txt
done
echo "2x2 done"
