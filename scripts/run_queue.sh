#!/usr/bin/env bash
# Run a list of training jobs N at a time. The protocol calls for a few hundred runs; firing
# them all at once and hoping is how you get a card full of half-dead processes.
#
#   printf '%s\n' "--arm cfc --seed 0" "--arm lstm --seed 0" | bash scripts/run_queue.sh
#   bash scripts/run_queue.sh --jobs 4 < jobs.txt
#
# Reads one job per line from stdin: the arguments to pass to train.py. Shared flags come
# from CSC413_TRAIN_ARGS.
#
# The default job count is derived from FREE VRAM rather than guessed. Measured footprint is
# ~2.45 GB per concurrent run (6 runs held 14.7 GB of a 16 GB card), so the divisor is 3 GB
# with a safety margin. It is also capped by CPU cores: this workload is kernel-launch-bound,
# so each job saturates one dispatch thread and more jobs than cores buys nothing.
set -uo pipefail

JOBS=""
[ "${1:-}" = "--jobs" ] && { JOBS="$2"; shift 2; }
PER_RUN_GB="${CSC413_GB_PER_RUN:-3}"
EXTRA="${CSC413_TRAIN_ARGS:-}"
RUNS_DIR="${CSC413_RUNS:-runs}"
LOG_DIR="${CSC413_LOGS:-logs}"
mkdir -p "$LOG_DIR"

if [ -z "$JOBS" ]; then
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  CORES=$(nproc)
  if [ -n "$FREE_MB" ]; then
    BY_VRAM=$(( FREE_MB / (PER_RUN_GB * 1024) ))
    JOBS=$(( BY_VRAM < CORES - 2 ? BY_VRAM : CORES - 2 ))
    [ "$JOBS" -lt 1 ] && JOBS=1
    echo "==> ${FREE_MB} MiB free, ${PER_RUN_GB} GiB/run, ${CORES} cores -> ${JOBS} concurrent"
  else
    JOBS=1; echo "==> no GPU visible -> 1 job"
  fi
fi

mapfile -t QUEUE
TOTAL=${#QUEUE[@]}
echo "==> ${TOTAL} jobs, ${JOBS} at a time, runs -> ${RUNS_DIR}/"

i=0; failed=0
for args in "${QUEUE[@]}"; do
  [ -z "${args// }" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 5; done
  i=$((i + 1))
  name=$(echo "$args" | tr -cd 'A-Za-z0-9.-' | cut -c1-60)
  echo "  [$i/$TOTAL] $args"
  (
    # Retry a VRAM refusal rather than dropping the job. Refusal is a TRANSIENT condition --
    # the preflight is comparing against whatever the other concurrent runs happen to hold at
    # that instant, and a slot freeing in the job table does not mean its memory has been
    # released yet. Dropping the job instead cost a real run: the sixth job of a six-job
    # batch was refused while five others held 2.5 GiB each, and the batch finished 5/6.
    for attempt in 1 2 3 4 5 6; do
      python3 -u train.py $args $EXTRA --runs "$RUNS_DIR" > "$LOG_DIR/${name}.log" 2>&1 && exit 0
      grep -q "REFUSING TO START" "$LOG_DIR/${name}.log" || {
        echo "  FAILED: $args (see $LOG_DIR/${name}.log)" >&2; exit 1; }
      echo "  waiting for VRAM, retry $attempt: $args"
      sleep 60
    done
    echo "  GAVE UP after 6 VRAM retries: $args" >&2; exit 1
  ) &
  sleep 3            # stagger, so N processes do not all allocate in the same instant
done
wait

for f in "$LOG_DIR"/*.log; do
  grep -q "REFUSING TO START\|OutOfMemory\|Traceback" "$f" 2>/dev/null && {
    echo "  incomplete: $(basename "$f")"; failed=$((failed + 1)); }
done
echo "==> done. ${failed} job(s) did not finish cleanly."
[ "$failed" -gt 0 ] && exit 1 || exit 0
