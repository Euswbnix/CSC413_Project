#!/usr/bin/env bash
# The single test evaluation of the steering-anticipation plan (section 4), or a rehearsal on val.
#   bash run_anticip_test.sh rehearse        # val, into d4/anticip_rehearsal/, untracked; must reproduce d4/anticip
#   bash run_anticip_test.sh test            # the one real run; refuses if it was ever started before
#   bash run_anticip_test.sh test --resume   # only to finish a real run that was interrupted
# The test sessions are already in the cache (appended for the D4 test evaluation, 2026-09-25).
# This scores the 40 finished checkpoints at the frozen learning rates and runs the fixed analysis;
# anticip_run.py refuses to train with the test split as validation. Nothing here chooses anything.
set -o pipefail
umask 077
cd "$HOME/workspace/hdd"
. ./env.sh
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/workspace/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
PY="$HOME/miniconda/envs/DL/bin/python"
MODE="$1"
SRC=d4/anticip
MARK=$SRC/TEST_EVALUATION_STARTED
ARMS="frame lstm cfc ltc transformer"
case "$MODE" in
  rehearse) SPLIT=val; OUT=d4/anticip_rehearsal; export CSC413_SWANLAB=off ;;
  test)     SPLIT=test; OUT=$SRC ;;
  *) echo "usage: $0 rehearse | test [--resume]"; exit 2 ;;
esac
say() { echo "$(date '+%F %T') $*"; }
lr_of() { "$PY" -c "import json; print(json.load(open('$SRC/$1_lr_chosen.json'))['chosen_lr'])"; }
extra_of() { [ "$1" = ltc ] && echo "--ode-unfolds 24 --compile-ltc"; }

if [ "$MODE" = test ]; then
  "$PY" -c "import json, sys; s = json.load(open('cache/dinov2_10hz/index.json'))['sessions']; \
n = sum(m['split'] == 'test' for m in s.values()); print(n, 'test sessions in the cache'); sys.exit(n == 0)" || exit 1
  if [ -e "$MARK" ] && [ "$2" != "--resume" ]; then
    echo "the anticipation test evaluation was already started:"; cat "$MARK"; echo "only '--resume' may finish it"; exit 1
  fi
  if [ ! -e "$MARK" ]; then
    { echo "started $(date '+%F %T')"; sha256sum anticip_run.py anticip_analyze.py d4_run.py d4_analyze.py \
        tracking.py metrics.py run_anticip_test.sh $SRC/*_lr_chosen.json; } > "$MARK"
  fi
else
  mkdir -p "$OUT/ckpt"
  for arm in $ARMS; do
    lr=$(lr_of $arm); u=""; [ $arm = ltc ] && u="_u24"
    for s in 0 1 2 3 4 5 6 7; do
      f="$SRC/ckpt/${arm}_lr$("$PY" -c "print(f'{$lr:g}')")_s${s}${u}.pt"
      [ -f "$f" ] || { echo "missing $f"; exit 1; }
      [ -f "$OUT/ckpt/$(basename "$f")" ] || cp "$f" "$OUT/ckpt/"
    done
  done
fi

say "scoring the finished checkpoints on $SPLIT"
pids=()
for arm in $ARMS; do
  "$PY" -W ignore anticip_run.py --cache cache/dinov2_10hz --arm $arm --stage final --lr "$(lr_of $arm)" \
      --seeds 0 1 2 3 4 5 6 7 --out "$OUT" --split-name $SPLIT $(extra_of $arm) \
      2>&1 | grep --line-buffered -v Warning > "$OUT/score_${arm}_${SPLIT}.log" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { say "scoring failed; see $OUT/score_*_${SPLIT}.log"; exit 1; }

say "the fixed analysis (Q1 primary; Q2 and Q3 Holm-corrected; Q4 descriptive)"
"$PY" -W ignore anticip_analyze.py --pred "$OUT" --split $SPLIT --require-all \
    --out "$OUT/analysis/${SPLIT}.json" || exit 1
say "done ($MODE)"
