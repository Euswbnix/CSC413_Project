#!/usr/bin/env bash
# Stage 2: eight seeds per arm at the rate the grid chose. Usage: run_d4_final.sh ARM "SEEDS" ["SEEDS" ...]
# Each quoted group of seeds is one process; the groups run in parallel.
umask 077
cd "$HOME/data/hdd"
export OMP_NUM_THREADS=4 PYTHONPATH="$HOME/data/hdd" XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1
PY="$HOME/miniconda/envs/DL/bin/python"
arm="$1"; shift
lr=$("$PY" - "$arm" <<'PYEOF'
import glob, json, sys
arm = sys.argv[1]
# only the grid points ({arm}_lr_<rate>.json), not the summary this script writes
rows = [r for f in glob.glob(f"d4/dinov2/{arm}_lr_[0-9]*.json") for r in json.load(open(f))["rows"]]
if len(rows) != 4:
    sys.exit(f"{arm}: expected 4 grid points, found {len(rows)}")
# a grid point that diverged is not eligible, whatever its score before diverging
ok = [r for r in rows if not r.get("diverged")]
if not ok:
    sys.exit(f"{arm}: every grid point diverged; no valid comparison (pre-registration section 6)")
best = min(ok, key=lambda r: r["mean_macro_mae"])
json.dump(dict(arm=arm, rows=sorted(rows, key=lambda r: -r["lr"]), chosen_lr=best["lr"]),
          open(f"d4/dinov2/{arm}_lr_chosen.json", "w"), indent=1)
print(best["lr"])
PYEOF
) || exit 1
echo "$arm: chosen lr $lr"
for group in "$@"; do
  nice -n 10 "$PY" -W ignore d4_run.py --cache cache/dinov2_10hz --arm "$arm" --stage final \
      --lr "$lr" --seeds $group --out d4/dinov2 2>&1 | grep --line-buffered -v Warning &
done
wait
echo "$arm FINAL DONE"
