#!/usr/bin/env bash
# Pilot: does aligning the objective with the reported metric change the outcome?
# Everything except --loss is copied from the collected runs.
#
# Concurrency 2, niced, and OMP capped. This host also serves JupyterHub, an LLM proxy and
# a RAG app; an earlier attempt at 6-way with default OMP threading took the load average
# to 287 and made sshd unreachable for four minutes. Each worker also pins ~3 GB of host
# RAM (data/dataset.py copies the memmap and calls pin_memory), which does not page out --
# so concurrency here is bounded by host RAM, not VRAM.
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
N=2
for ARM in cfc lstm; do
 for S in 0 1 2; do
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do wait -n; done
  ( nice -n 10 $PY train.py --arm $ARM --seed $S --T 16 --lr 1e-3 --k 0.1 --aug full \
      --epochs 30 --loss bin-balanced --processed data/processed \
      --runs runs_pilot_balanced --device cuda --min-free-gb 3.0 \
      > logs_pilot_${ARM}_s${S}.log 2>&1 || echo "FAIL $ARM $S" >> /tmp/pilot_fail.txt ) &
 done
done
wait
echo done
