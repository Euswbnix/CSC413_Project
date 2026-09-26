#!/usr/bin/env bash
# The single test evaluation (docs/preregistration_2026-09-22.md sections 7 and 9.4), or a full
# rehearsal of it on the validation split.
#   bash run_test_eval.sh rehearse          # val, into d4_rehearsal/, untracked; must reproduce d4/dinov2
#   bash run_test_eval.sh test              # the one real run; refuses if it was ever started before
#   bash run_test_eval.sh test --resume     # only to finish a real run that was interrupted
# Steps for test: extract the test sessions' features with the train/val settings, append them to
# the cache, score every finished checkpoint of the four arms at the frozen learning rates (d4_run
# refuses to train on the test split), then the registered analysis and the descriptive
# Transformer comparison. Nothing here chooses anything.
set -o pipefail
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
PY="$HOME/miniconda/envs/DL/bin/python"
MODE="$1"
MARK=d4/TEST_EVALUATION_STARTED
case "$MODE" in
  rehearse) SPLIT=val; OUT=d4_rehearsal; export CSC413_SWANLAB=off ;;
  test)     SPLIT=test; OUT=d4/dinov2 ;;
  *) echo "usage: $0 rehearse | test [--resume]"; exit 2 ;;
esac
say() { echo "$(date '+%F %T') $*"; }
lr_of() { "$PY" -c "import json; print(json.load(open('d4/dinov2/$1_lr_chosen.json'))['chosen_lr'])"; }
extra_of() { [ "$1" = ltc ] && echo "--ode-unfolds 24 --compile-ltc"; }

if [ "$MODE" = test ]; then
  if [ -e "$MARK" ] && [ "$2" != "--resume" ]; then
    echo "the test evaluation was already started:"; cat "$MARK"; echo "only '--resume' may finish it"; exit 1
  fi
  if [ ! -e "$MARK" ]; then
    { echo "started $(date '+%F %T')"; sha256sum d4_run.py d4_analyze.py make_cache.py extract_features.py \
        tracking.py metrics.py run_test_eval.sh d4/dinov2/*_lr_chosen.json; } > "$MARK"
  fi
  sessions=$("$PY" -c "import json; s = json.load(open('split_a/split.json'))['sessions']; \
print(' '.join(sorted(k for k, m in s.items() if m['split'] == 'test')))")
  say "extracting features for $(echo $sessions | wc -w) test sessions (settings of features/dinov2_s1)"
  "$PY" -W ignore extract_features.py --videos video --raw raw --out features/dinov2_s1 --encoder dinov2_vitb14 \
      --stride 1 --width 392 --height 224 --batch 64 --workers 10 --cv-threads 2 --speed-ratio 3.6 \
      --sessions $sessions || exit 1
  "$PY" make_cache.py --features features/dinov2_s1 --split split_a --out cache/dinov2_10hz \
      --base-hz 10 --splits test --add || exit 1
else
  # the rehearsal scores copies of the same finished checkpoints, so the real ones are never written
  mkdir -p "$OUT/ckpt"
  for arm in lstm cfc ltc transformer; do
    lr=$(lr_of $arm); u=""; [ $arm = ltc ] && u="_u24"
    for s in 0 1 2 3 4 5 6 7; do
      f="d4/dinov2/ckpt/${arm}_lr$("$PY" -c "print(f'{$lr:g}')")_s${s}${u}.pt"
      [ -f "$f" ] || { echo "missing $f"; exit 1; }
      [ -f "$OUT/ckpt/$(basename "$f")" ] || cp "$f" "$OUT/ckpt/"
    done
  done
fi

say "scoring the finished checkpoints on $SPLIT"
pids=()
for arm in lstm cfc ltc transformer; do
  "$PY" -W ignore d4_run.py --cache cache/dinov2_10hz --arm $arm --stage final --lr "$(lr_of $arm)" \
      --seeds 0 1 2 3 4 5 6 7 --out "$OUT" --split-name $SPLIT $(extra_of $arm) \
      2>&1 | grep --line-buffered -v Warning > "$OUT/score_${arm}_${SPLIT}.log" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { say "scoring failed; see $OUT/score_*_${SPLIT}.log"; exit 1; }

mkdir -p "$OUT/analysis"
say "registered analysis (primary + Holm-corrected secondary family)"
"$PY" -W ignore d4_analyze.py --pred "$OUT" --family --split $SPLIT --out "$OUT/analysis/family_${SPLIT}.json" || exit 1
for b in lstm cfc; do
  say "descriptive only (pre-registration 9.1): transformer vs $b"
  "$PY" -W ignore d4_analyze.py --pred "$OUT" --a transformer --b $b --split $SPLIT \
      --out "$OUT/analysis/${SPLIT}_transformer_vs_${b}_descriptive.json" || exit 1
done
say "done ($MODE)"
