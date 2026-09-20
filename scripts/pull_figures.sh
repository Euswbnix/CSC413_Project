#!/usr/bin/env bash
# Local dev convenience: copy generated figures and stats down from the training box.
# The training machine is headless, so nothing rendered there can be looked at in place.
#
#   export CSC413_REMOTE=user@host          # once, in your shell profile
#   bash scripts/pull_figures.sh            # -> figures/, stats.txt
#   bash scripts/pull_figures.sh --with-gif # also the frame/label alignment GIF (~5 MB)
#
# Deliberately not part of the training or evaluation pipeline, and the host is read from
# the environment so no personal address is committed. Everything it fetches is ignored by
# .gitignore: no dataset frames enter the repository, including the ones inside figures/.
set -euo pipefail

REMOTE="${CSC413_REMOTE:-}"
REMOTE_DIR="${CSC413_REMOTE_DIR:-workspace/csc413}"
[ -n "$REMOTE" ] || { echo "set CSC413_REMOTE=user@host first" >&2; exit 2; }

mkdir -p figures artifacts_2017
echo "==> $REMOTE:$REMOTE_DIR"

scp -q "$REMOTE:$REMOTE_DIR/figures/*.png" figures/ 2>/dev/null || echo "  (no figures yet)"
scp -q "$REMOTE:$REMOTE_DIR/stats.txt" ./stats.txt 2>/dev/null || echo "  (no stats.txt yet)"
scp -q "$REMOTE:$REMOTE_DIR/artifacts_2017/stats_2017.txt" artifacts_2017/ 2>/dev/null || true

if [ "${1:-}" = "--with-gif" ]; then
  scp -q "$REMOTE:$REMOTE_DIR/data/processed/verify_*.gif" figures/ 2>/dev/null \
    || echo "  (no verify GIF yet -- run preprocess.py --verify-gif N, pointing N at a curvy stretch)"
fi

echo "==> local:"
ls -1 figures/*.png figures/*.gif 2>/dev/null | sed 's/^/  /' || true
[ -f stats.txt ] && echo "  stats.txt ($(wc -l < stats.txt) lines)"
