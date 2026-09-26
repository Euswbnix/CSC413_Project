#!/usr/bin/env bash
# Start or stop a D4 pipeline (run_<name>.sh) as one process group, so stopping it also stops the
# xargs scheduler and every python worker it spawned (killing only the parent orphans them).
#   bash d4_ctl.sh start [name [args]]   # start, or resume from checkpoints after a shutdown
#   bash d4_ctl.sh stop [name]      # stop everything; the last finished epoch of each run is kept
#   bash d4_ctl.sh status [name]
# name: ltc24 (default), transformer, or test_eval (args: rehearse | test [--resume]).
# Log: d4_<name>.log; any further arguments go to run_<name>.sh.
cd "$HOME/workspace/hdd"
. ./env.sh
P="${2:-ltc24}"
[ -f "run_$P.sh" ] || { echo "no pipeline run_$P.sh"; exit 2; }
PIDFILE="d4_$P.pgid"
case "$1" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 -- "-$(cat $PIDFILE)" 2>/dev/null; then
      echo "already running (process group $(cat $PIDFILE))"; exit 1
    fi
    setsid bash "run_$P.sh" "${@:3}" >> "d4_$P.log" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "started process group $(cat $PIDFILE)" ;;
  stop)
    [ -f "$PIDFILE" ] || { echo "not running"; exit 0; }
    kill -- "-$(cat $PIDFILE)" 2>/dev/null
    for i in 1 2 3 4 5 6; do kill -0 -- "-$(cat $PIDFILE)" 2>/dev/null || break; sleep 5; done
    kill -9 -- "-$(cat $PIDFILE)" 2>/dev/null
    rm -f "$PIDFILE"
    echo "stopped; half-written checkpoints: $(ls d4/dinov2/ckpt/*.tmp 2>/dev/null | wc -l)" ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 -- "-$(cat $PIDFILE)" 2>/dev/null; then
      echo "running: $(pgrep -g "$(cat $PIDFILE)" -f 'python' | wc -l) python workers"
    else
      echo "not running"
    fi ;;
  *) echo "usage: $0 start|stop|status [ltc24|transformer|test_eval [args]]"; exit 2 ;;
esac
