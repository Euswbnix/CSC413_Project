#!/usr/bin/env bash
# Re-evaluate the 115 lab-fleet runs on TEST with --save-predictions, so every run carries the
# v2 metrics and a predictions file. Inference only. The headline runs (cfc/lstm at the
# pre-registered configuration) go first so the main table is available early.
#
# The first test evaluation of these runs happened on the lab 4080s under torch 2.13; this
# one is on the 5090 under torch 2.8. The old files are copied aside first so the two
# platforms can be compared -- the reported numbers should not depend on the GPU.
#
# Concurrency 2, niced, OMP capped: this host also serves JupyterHub, an LLM proxy and a RAG app.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

ALL=$(ls -d lab/*/*/ | sort)
HEAD=$(echo "$ALL" | grep -E '/(cfc|lstm)_s[0-9]+_T16_lr0\.001_k0\.1_aug-full/$')
REST=$(echo "$ALL" | grep -vE '/(cfc|lstm)_s[0-9]+_T16_lr0\.001_k0\.1_aug-full/$')

for D in $HEAD $REST; do
  [ -f "$D/checkpoints/best.pt" ] || continue
  [ -f "$D/predictions_test.npz" ] && continue
  [ -f "$D/final_metrics_test.json" ] && [ ! -f "$D/final_metrics_test.lab4080.json" ] && \
    cp "$D/final_metrics_test.json" "$D/final_metrics_test.lab4080.json"
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
  ( nice -n 10 $PY evaluate.py "$D" --split test --save-predictions --processed data/processed \
      > "$D/reeval_test.log" 2>&1 || echo "FAIL $D" >> /tmp/reeval_fail.txt ) &
done
wait
echo "lab test re-eval done"
$PY scripts/recompute_metrics.py lab runs_power runs_reg_do30_wd1e3 runs_reg_do50_wd1e2 \
    runs_fov200 runs_fov256 runs_pilot_balanced --processed data/processed
echo "recompute done"
