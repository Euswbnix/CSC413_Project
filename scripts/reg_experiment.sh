#!/usr/bin/env bash
# Regularisation sweep. The encoder is 168,244 of 193,141 parameters (87.1%) and carries no
# dropout and no normalisation; its training data is one continuous drive whose adjacent
# frames are near-duplicates. Measured over 30 epochs: train loss -58%, validation MSE
# +69% (0.285 -> 0.482), validation global MAE 9.96 -> 13.41 deg. Early stopping then fires
# at a median epoch of 4 of 30, and 15% of the 115 collected runs never beat epoch 0 -- which
# is why they read as near-constant predictors.
#
# Defaults (dropout 0.0, wd 1e-4) reproduce every collected run, so those 115 are the control.
# Concurrency 2, OMP capped: this host also serves JupyterHub, an LLM proxy and a RAG app.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

run() {  # tag dropout wd
  local TAG=$1 DO=$2 WD=$3
  for S in 0 1 2; do
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
    ( nice -n 10 $PY train.py --arm cfc --seed "$S" --T 16 --lr 1e-3 --k 0.1 --aug full \
        --epochs 30 --dropout "$DO" --weight-decay "$WD" \
        --processed data/processed --runs "runs_reg_$TAG" --device cuda --min-free-gb 3.0 \
        > "logs_reg_${TAG}_s${S}.log" 2>&1 || echo "FAIL $TAG s$S" >> /tmp/reg_fail.txt ) &
  done
  wait
}

run do30_wd1e3 0.3 1e-3
run do50_wd1e2 0.5 1e-2

for T in do30_wd1e3 do50_wd1e2; do
  for D in runs_reg_$T/*/; do
    [ -f "$D/checkpoints/best.pt" ] || continue
    for SP in val test; do
      nice -n 10 $PY evaluate.py "$D" --split "$SP" --save-predictions \
        --processed data/processed >/dev/null 2>&1
    done
  done
done
echo "reg experiment done"
