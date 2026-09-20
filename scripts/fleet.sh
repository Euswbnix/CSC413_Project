#!/usr/bin/env bash
# Run the protocol across several identical lab machines.
#
#   export CSC413_FLEET="huangjiz@dh2026pc02.utm.utoronto.ca huangjiz@dh2026pc04..."
#   bash scripts/fleet.sh survey                 # who is on each box, is its GPU free
#   bash scripts/fleet.sh stage                  # processed data -> shared home, once
#   bash scripts/fleet.sh provision              # shared home -> each box's local disk
#   bash scripts/fleet.sh run jobs.txt           # dispatch, round-robin
#   bash scripts/fleet.sh collect                # gather runs/ back
#
# Why it is shaped this way
# -------------------------
# * The fleet shares ONE home over NFS, so ~/csc413 is already identical everywhere and the
#   repo never needs copying. Only the 3 GB memmap does, and NFS is the staging medium --
#   which also means no host-to-host SSH keys are needed.
# * The memmap is COPIED rather than regenerated per host. Re-running preprocess on each box
#   would work, but copying guarantees byte-identical inputs, and a comparison across
#   machines is only meaningful if the data is the same bytes.
# * Runs are written to LOCAL disk, not the shared home. Home is NFS over a TLS tunnel with
#   rsize=wsize=8192; a checkpoint written every epoch by every concurrent job on every host
#   would pour that through an 8 KB pipe. `collect` rsyncs them back at the end.
# * Only machines with NO other logged-in user are used. These are teaching lab machines
#   that people walk up to.
set -uo pipefail

FLEET="${CSC413_FLEET:-}"
DATA_LOCAL="${CSC413_DATA_LOCAL:-/var/tmp/csc413_data/processed}"
# RELATIVE on purpose: it is resolved by the REMOTE shell, in the remote home. Writing
# $HOME here expands on the machine running this script -- which is the laptop, not the
# fleet -- and the stage step then tries to mkdir /Users/... on a Linux box.
DATA_STAGE="${CSC413_DATA_STAGE:-csc413_stage}"
RUNS_LOCAL="${CSC413_RUNS_LOCAL:-/var/tmp/csc413_runs}"
RUNS_HOME="${CSC413_RUNS_HOME:-runs_fleet}"   # local to wherever this script runs
PER_HOST="${CSC413_PER_HOST:-5}"
CMD="${1:-help}"; shift || true

hosts() { [ -n "$FLEET" ] || { echo "set CSC413_FLEET first" >&2; exit 2; }; echo $FLEET; }

case "$CMD" in

survey)
  for h in $(hosts); do
    ( r=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        'u=$(who | awk "{print \$1}" | sort -u | grep -v "^$(whoami)$" | tr "\n" ",");
         g=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null|tr -d " ");
         d=$([ -f '"$DATA_LOCAL"'/frames.npy ] && echo data-ready || echo no-data);
         echo "${u:-free}|${g:-nogpu}|$d"' 2>/dev/null)
      printf "  %-46s %s\n" "$h" "${r:-<unreachable>}" ) &
  done; wait ;;

stage)
  # One 3 GB write to the shared home. Every host reads it from there, so no host-to-host auth.
  echo "==> staging $DATA_LOCAL -> $DATA_STAGE (shared home)"
  src=$(echo $FLEET | awk '{print $1}')
  ssh -o BatchMode=yes "$src" "mkdir -p \$HOME/$DATA_STAGE && cp -u $DATA_LOCAL/* \$HOME/$DATA_STAGE/ && du -sh \$HOME/$DATA_STAGE && ls \$HOME/$DATA_STAGE"
  ;;

provision)
  for h in $(hosts); do
    ( echo "  provisioning $h"
      ssh -o BatchMode=yes "$h" "
        mkdir -p $DATA_LOCAL $RUNS_LOCAL
        for f in \$HOME/$DATA_STAGE/*; do
          b=\$(basename \$f)
          [ -f $DATA_LOCAL/\$b ] && [ \$(stat -c%s \$f) -eq \$(stat -c%s $DATA_LOCAL/\$b) ] && continue
          cp \$f $DATA_LOCAL/
        done
        ls $DATA_LOCAL | tr '\n' ' '" ) &
  done; wait; echo; echo "==> provisioned" ;;

run)
  JOBS="${1:?usage: fleet.sh run jobs.txt}"
  mapfile -t Q < "$JOBS"
  HS=($(hosts)); n=${#HS[@]}; i=0
  for h in "${HS[@]}"; do
    # Round-robin: every nth line to this host. Keeps each host's queue independent, so one
    # slow or dead box cannot stall the others.
    awk -v k="$i" -v n="$n" 'NR % n == k' "$JOBS" > "/tmp/fleet_$i.txt"
    c=$(grep -c . "/tmp/fleet_$i.txt")
    echo "  $h <- $c jobs"
    scp -q -o BatchMode=yes "/tmp/fleet_$i.txt" "$h:/tmp/fleet_jobs.txt"
    ssh -o BatchMode=yes "$h" "cd ~/csc413 && \
      CSC413_RUNS=$RUNS_LOCAL CSC413_LOGS=/var/tmp/csc413_logs \
      CSC413_TRAIN_ARGS='--processed $DATA_LOCAL' \
      nohup bash scripts/run_queue.sh --jobs $PER_HOST < /tmp/fleet_jobs.txt \
      > /var/tmp/fleet_queue.log 2>&1 &" &
    i=$((i+1))
  done; wait
  echo "==> dispatched ${#Q[@]} jobs over $n hosts. 'fleet.sh status' to watch." ;;

status)
  for h in $(hosts); do
    ( r=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
        "echo \"\$(pgrep -cf '[t]rain.py --arm') running, \$(ls $RUNS_LOCAL 2>/dev/null | wc -l) runs, \$(tail -1 /var/tmp/fleet_queue.log 2>/dev/null | cut -c1-40)\"" 2>/dev/null)
      printf "  %-46s %s\n" "$h" "${r:-<unreachable>}" ) &
  done; wait ;;

collect)
  mkdir -p "$RUNS_HOME"
  for h in $(hosts); do
    echo "  collecting $h"
    rsync -a --ignore-existing -e "ssh -o BatchMode=yes" \
      --exclude 'checkpoints/last.pt' "$h:$RUNS_LOCAL/" "$RUNS_HOME/" 2>/dev/null || true
  done
  echo "==> $(ls "$RUNS_HOME" | wc -l) runs in $RUNS_HOME"
  echo "    (last.pt excluded -- best.pt is what evaluate.py loads, and last.pt is 800 KB"
  echo "     per run of resume state that nothing downstream reads)" ;;

*) sed -n '2,30p' "$0" ;;
esac
