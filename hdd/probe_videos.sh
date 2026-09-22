#!/usr/bin/env bash
# Read every HDD video's container metadata (frame rate, frame count, size) without keeping any
# video: each mp4 is streamed out of hdd.tar.gz into a temp file, opened with OpenCV, and deleted.
# No frame is decoded or shown. Then compares each frame count with its png_timestamp.csv rows.
# Output: $HDD/checks/mp4_meta.csv (server-only). Usage, on the server: bash hdd/probe_videos.sh
set -o pipefail
umask 077
HDD=${HDD:-$HOME/data/hdd}
PY=${PY:-$HOME/.conda/envs/llm-ui/bin/python}
export PYTHONPATH=${PYTHONPATH:-$HOME/.cache/csc413_testdeps}

if [ "$1" = "--member" ]; then
    # called by tar once per archive member, with the member's bytes on stdin
    case "$TAR_FILENAME" in *.mp4) ;; *) cat > /dev/null; exit 0 ;; esac
    tmp=$(mktemp -p "$HDD/tmp" XXXXXX.mp4)
    cat > "$tmp"
    "$PY" - "$tmp" "$TAR_FILENAME" "$TAR_SIZE" >> "$HDD/checks/mp4_meta.csv" <<'PY'
import sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
g = cap.get
parts = sys.argv[2].split("/")
print(",".join(map(str, [parts[-4], parts[-1], sys.argv[3], int(cap.isOpened()), g(cv2.CAP_PROP_FPS),
      int(g(cv2.CAP_PROP_FRAME_COUNT)), int(g(cv2.CAP_PROP_FRAME_WIDTH)), int(g(cv2.CAP_PROP_FRAME_HEIGHT))])))
PY
    rm -f "$tmp"
    exit 0
fi

mkdir -p "$HDD/tmp" "$HDD/checks"
echo "session,file,bytes,opened,fps,frame_count,width,height" > "$HDD/checks/mp4_meta.csv"
N="nice -n 19 ionice -c3"
$N pigz -dc "$HDD/hdd.tar.gz" | $N tar -xOf - ./hdd_data/release_2019_07_08.tar.gz | $N pigz -dc \
    | $N tar -xf - --wildcards '*.mp4' --to-command="bash $(realpath "$0") --member"
rc=$?
rmdir "$HDD/tmp" 2>/dev/null
echo "probe rc=$rc"
"$PY" - "$HDD" <<'PY'
import glob, os, sys
import pandas as pd
hdd = sys.argv[1]
m = pd.read_csv(os.path.join(hdd, "checks", "mp4_meta.csv"), dtype={"session": str})
rows = {p.split(os.sep)[-4]: sum(1 for _ in open(p)) - 1
        for p in glob.glob(os.path.join(hdd, "raw", "release_2019_07_25", "*", "*", "camera", "center", "png_timestamp.csv"))}
same = (m.frame_count == m.session.map(rows)).sum()
print(f"videos {len(m)}, opened {int(m.opened.sum())}, fps {sorted(m.fps.unique())}, "
      f"sizes {sorted((m.width.astype(str) + 'x' + m.height.astype(str)).unique())}, "
      f"frame count equal to timestamp rows: {same}/{len(m)}")
PY
