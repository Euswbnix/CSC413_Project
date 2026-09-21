#!/usr/bin/env bash
# Field-of-view experiment. Only --src-crop-rows changes: the stored tensor stays 66x240 and
# the model input stays 66x200, so the encoder, its parameter count and the LSTM parameter
# match are byte-identical across conditions. What varies is how far up the source frame the
# crop reaches -- and therefore whether the horizon and the distant road are in the input at
# all. 150 is the reference loader's [-150:] and reproduces every run collected so far.
#
# The tradeoff is real and is what the experiment measures: 150 source rows squeezed into 66
# is a 2.27x vertical squash, 200 is 3.0x, 256 is 3.9x. More field of view costs resolution.
#
# Concurrency 2 and OMP capped: this host also serves JupyterHub, an LLM proxy and a RAG app.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for R in 200 256; do
  OUT=data/processed_fov$R
  if [ ! -f "$OUT/frames.npy" ]; then
    echo "=== preprocess src_crop_rows=$R -> $OUT"
    nice -n 10 $PY data/preprocess.py --data-root data/raw --out "$OUT" --src-crop-rows "$R" \
      > "logs_prep_fov$R.log" 2>&1 || { echo "PREP FAIL $R"; continue; }
  fi
  echo "=== train fov$R"
  for S in 0 1 2; do
    while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
    ( nice -n 10 $PY train.py --arm cfc --seed "$S" --T 16 --lr 1e-3 --k 0.1 --aug full \
        --epochs 30 --processed "$OUT" --runs "runs_fov$R" --device cuda --min-free-gb 3.0 \
        > "logs_fov${R}_s${S}.log" 2>&1 || echo "FAIL fov$R s$S" >> /tmp/fov_fail.txt ) &
  done
  wait
done
echo done
