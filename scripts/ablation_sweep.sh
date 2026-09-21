#!/usr/bin/env bash
set -u
cd ~/workspace/csc413
PY=~/.conda/envs/llm-ui/bin/python
N=6
for R in lab/runs_fleet/*/ lab/runs_aug/*/ lab/runs_lr/*/ lab/runs/*/; do
  [ -f "$R/checkpoints/best.pt" ] || continue
  case "$(basename "$R")" in cfc_*|lstm_*|gru_*|cnn_avg_*) ;; *) continue ;; esac
  [ -f "$R/state_ablation_val.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do wait -n; done
  ( $PY scripts/state_ablation.py "$R" --split val --processed data/processed \
      >"$R/ablation.log" 2>&1 || echo "FAIL $R" >> /tmp/abl_fail.txt ) &
done
wait
echo done
