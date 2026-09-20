#!/usr/bin/env bash
# Fetch the SullyChen 2018 driving dataset and verify it. Records provenance for the
# README's Data Source section. Never commits anything (see .gitignore).
#
# Usage:  bash data/download.sh [dest_dir]
set -euo pipefail

DEST="${1:-data/raw}"
# Google Drive file id for the 2018 release, from https://github.com/SullyChen/driving-datasets
FILE_ID="${SULLYCHEN_2018_ID:-1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7}"
ARCHIVE="$DEST/driving_dataset.zip"
EXPECTED_SHA256="${SULLYCHEN_2018_SHA256:-}"

mkdir -p "$DEST"

# gdown, NOT wget/curl. A plain wget on a large Google Drive file silently saves the
# virus-scan interstitial HTML page instead of the zip -- you get a 3 KB "download" that
# unzips to nothing, or worse, a file named .zip that is HTML.
if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install it:  pip install gdown" >&2
  echo "Do NOT substitute wget or curl -- see the comment above." >&2
  exit 1
fi

if [ ! -f "$ARCHIVE" ]; then
  echo "==> downloading (3.1 GB)"
  gdown "$FILE_ID" -O "$ARCHIVE"
fi

# Refuse to continue on a hash mismatch rather than warning. A corrupted or
# silently-substituted archive would produce a plausible-looking project built on the
# wrong bytes.
ACTUAL_SHA256=$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)
echo "sha256: $ACTUAL_SHA256"
if [ -n "$EXPECTED_SHA256" ]; then
  if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "SHA-256 MISMATCH -- refusing to continue." >&2
    echo "  expected $EXPECTED_SHA256" >&2
    echo "  actual   $ACTUAL_SHA256" >&2
    exit 1
  fi
  echo "sha256 OK"
else
  echo "NOTE: SULLYCHEN_2018_SHA256 is unset, so the archive was NOT verified."
  echo "      Record the hash above in README.md (Data Source) and export it so every"
  echo "      later run is checked:  export SULLYCHEN_2018_SHA256=$ACTUAL_SHA256"
fi

echo "==> extracting"
unzip -q -o "$ARCHIVE" -d "$DEST"
# the archive may nest everything one level down
if [ ! -f "$DEST/data.txt" ] && [ -f "$DEST/driving_dataset/data.txt" ]; then
  mv "$DEST"/driving_dataset/* "$DEST"/ && rmdir "$DEST/driving_dataset"
fi

# The file/row audit. A discrepancy of roughly 161 is EXPECTED and is not an error: those
# images have no data.txt line (repo issue #2) and the author has confirmed the remaining
# angles are unaffected. What it means is that the image/label join must be by FILENAME --
# data.txt line i IS memmap row i -- and never by position in the image directory.
LINES=$(grep -c . "$DEST/data.txt")
JPGS=$(find "$DEST" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')
echo "==> audit"
echo "  data.txt lines : $LINES"
echo "  .jpg files     : $JPGS"
echo "  difference     : $((JPGS - LINES))   (expect roughly +161; see repo issue #2)"
if [ "$JPGS" -lt "$LINES" ]; then
  echo "  FAIL: fewer images than label lines. data.txt references files that do not exist." >&2
  exit 1
fi

cat > "$DEST/PROVENANCE.txt" <<PROV
source        : https://github.com/SullyChen/driving-datasets (2018 release)
google_drive  : $FILE_ID
archive       : $(basename "$ARCHIVE")
sha256        : $ACTUAL_SHA256
accessed      : $(date -u +%Y-%m-%dT%H:%M:%SZ)
data.txt_lines: $LINES
jpg_files     : $JPGS
licence       : MIT, Copyright (c) 2018 Sully Chen (LICENSE file in the repository)
restriction   : the repository README asks, under a heading titled "IMPORTANT", that these
                datasets never be used to pilot a car. That is a request by the author, NOT
                a licence term. This project honours it: offline prediction only.
PROV
echo "==> wrote $DEST/PROVENANCE.txt -- paste these lines into README.md (Data Source)"
