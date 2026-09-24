#!/usr/bin/env bash
# Wait for the LTC learning-rate grid to finish, then start LTC's eight seeds.
# Seed 0 at the chosen rate is the same run as that grid point, so it resumes from the grid's
# checkpoint instead of training again.
cd "$HOME/data/hdd"
until [ "$(ls d4/dinov2/ltc_lr_[0-9]*.json 2>/dev/null | wc -l)" -ge 4 ]; do sleep 300; done
echo "$(date '+%F %T') LTC grid complete, starting finals"
# one process per seed: LTC is bound by its per-step Python loop, so eight processes use the GPU
# far better than four processes running two seeds each
bash run_d4_final.sh ltc 0 1 2 3 4 5 6 7
echo "$(date '+%F %T') LTC FINAL DONE"
