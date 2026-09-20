#!/usr/bin/env bash
# Fetch a SullyChen driving dataset, verify it IS the release it claims to be, and record
# provenance for the README's Data Source section. Commits nothing (see .gitignore).
#
#   bash data/download.sh              # 2018 release -> data/raw        (the project dataset)
#   bash data/download.sh 2017         # 2017 release -> data/raw_2017   (debug fixture ONLY)
#
# The 2017 set is a SMOKE-TEST FIXTURE. Do not merge it into training: if the two releases
# cover the same roads, merging reintroduces exactly the near-duplicate leakage the
# chronological split exists to prevent, across a boundary the split logic cannot see.
set -euo pipefail

VERSION="${1:-2018}"
case "$VERSION" in
  2018) FILE_ID="1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B"; DEST="data/raw"
        EXPECT_TIMESTAMPS=1; EXPECT_FRAMES=63000
        KNOWN_SHA="${SULLYCHEN_2018_SHA256:-}" ;;
  2017) FILE_ID="1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7"; DEST="data/raw_2017"
        EXPECT_TIMESTAMPS=0; EXPECT_FRAMES=45500
        KNOWN_SHA="${SULLYCHEN_2017_SHA256:-d25082b8890afea797ca39ebefac38904973d32c6b121906ddab073a2a8fbe5f}" ;;
  *) echo "usage: $0 [2018|2017]" >&2; exit 2 ;;
esac
ARCHIVE="$DEST/driving_dataset.zip"
mkdir -p "$DEST"

# gdown, NOT wget/curl. A plain wget on a large Google Drive file silently saves the
# virus-scan interstitial HTML instead of the zip.
command -v gdown >/dev/null 2>&1 || { echo "gdown not found:  pip install gdown" >&2; exit 1; }

[ -f "$ARCHIVE" ] || { echo "==> downloading $VERSION release"; gdown "$FILE_ID" -O "$ARCHIVE"; }

ACTUAL_SHA=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
echo "sha256: $ACTUAL_SHA"
if [ -n "$KNOWN_SHA" ]; then
  [ "$ACTUAL_SHA" = "$KNOWN_SHA" ] || {
    echo "SHA-256 MISMATCH -- refusing to continue." >&2
    echo "  expected $KNOWN_SHA" >&2; echo "  actual   $ACTUAL_SHA" >&2; exit 1; }
  echo "sha256 OK"
else
  echo "NOTE: no known hash for the $VERSION release yet. Record the one above in README.md"
  echo "      (Data Source) and export it so later runs are verified:"
  echo "      export SULLYCHEN_${VERSION}_SHA256=$ACTUAL_SHA"
fi

echo "==> extracting"
unzip -q -o "$ARCHIVE" -d "$DEST"

# Flatten if the archive nests one level down. `mv dir/*` is wrong twice over: the glob skips
# dotfiles (the archive ships a 2.3 MB .DS_Store, which then leaves the directory non-empty),
# and with 63,000 entries it can blow past ARG_MAX. find -exec ... + handles both.
if [ ! -f "$DEST/data.txt" ] && [ -f "$DEST/driving_dataset/data.txt" ]; then
  find "$DEST/driving_dataset" -mindepth 1 -maxdepth 1 -exec mv -t "$DEST" {} +
  rmdir "$DEST/driving_dataset" || echo "  note: $DEST/driving_dataset not empty, left in place"
fi
find "$DEST" -maxdepth 1 -name '.DS_Store' -delete    # macOS junk shipped inside the archive

echo "==> audit"
[ -f "$DEST/data.txt" ] || { echo "  FAIL: no data.txt under $DEST" >&2; exit 1; }
LINES=$(grep -c . "$DEST/data.txt")
JPGS=$(find "$DEST" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')
FIRST=$(head -1 "$DEST/data.txt")
echo "  data.txt lines : $LINES"
echo "  .jpg files     : $JPGS"
echo "  difference     : $((JPGS - LINES))   (expect roughly +161; repo issue #2, labels unaffected)"
echo "  first line     : $FIRST"

# THE CHECK THIS SCRIPT ORIGINALLY LACKED. Verifying a hash only proves the bytes arrived
# intact; it does not prove they are the dataset you asked for. A wrong-release download is
# silent and expensive: the 2017 labels carry no timestamp, so measured fps, the
# decorrelation-derived split buffer and gap-based segmentation all quietly become
# unavailable, and the project would be built on assumed values instead.
if [ "$EXPECT_TIMESTAMPS" = 1 ]; then
  echo "$FIRST" | grep -qE '^[^ ]+\.jpg -?[0-9.]+,[0-9]{4}-[0-9]{2}-[0-9]{2} ' || {
    echo "  FAIL: expected the $VERSION label format 'file.jpg angle,YYYY-MM-DD HH:MM:SS:mmm'," >&2
    echo "        but data.txt has no timestamp field. This archive is the 2017 release." >&2
    exit 1; }
  echo "  format         : timestamps present (2018 layout) OK"
else
  echo "$FIRST" | grep -qE ',' && {
    echo "  FAIL: the 2017 release should have no timestamp field, but this one does." >&2; exit 1; }
  echo "  format         : no timestamps (2017 layout, as expected for the fixture) OK"
fi
[ "$JPGS" -ge "$LINES" ] || { echo "  FAIL: fewer images than label lines." >&2; exit 1; }
PCT=$(( LINES * 100 / EXPECT_FRAMES ))
[ "$PCT" -ge 80 ] && [ "$PCT" -le 125 ] || \
  echo "  WARNING: $LINES lines is $PCT% of the ~$EXPECT_FRAMES expected for $VERSION."

cat > "$DEST/PROVENANCE.txt" <<PROV
release       : SullyChen driving dataset, $VERSION
source        : https://github.com/SullyChen/driving-datasets
google_drive  : $FILE_ID
sha256        : $ACTUAL_SHA
accessed      : $(date -u +%Y-%m-%dT%H:%M:%SZ)
data.txt_lines: $LINES
jpg_files     : $JPGS
label_format  : $FIRST
licence       : MIT, Copyright (c) 2018 Sully Chen (LICENSE file in the repository)
restriction   : the repository README asks, under a heading titled "IMPORTANT", that these
                datasets never be used to pilot a car. That is a request by the author, NOT
                a licence term. This project honours it: offline prediction only.
PROV
echo "==> wrote $DEST/PROVENANCE.txt -- paste these lines into README.md (Data Source)"
