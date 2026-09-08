#!/bin/bash
# Download the released EmoSphere++ checkpoint (README "TTS module trained on 11M",
# https://works.do/xO6ZtDB -> Naver Works Drive public link). The share page is a
# Next.js app; the underlying REST API resolved 2026-09-07:
#   GET https://api.drive.worksmobile.com/v1/shared-links/<linkKey>              -> rootFileId
#   GET https://api.drive.worksmobile.com/v1/shared-links/<linkKey>/files/<id>   -> downloadUrl
# (model_ckpt_steps_1100000.ckpt, 338,738,261 bytes). Usage: download_ckpt.sh <dest_path>
set -euo pipefail
DEST="$1"
LINK_KEY="3hFrTbhK9IME_XpE5IygXw.MZZC_xtN6LuYqLdbRZ8K5XW5gBOM0rZNfHEo20ww7jrDe5YLq1xCfYROrwpBwg237Ck2-JeN48CIeI6SyMCVNg"
API="https://api.drive.worksmobile.com/v1/shared-links/$LINK_KEY"
EXPECTED_BYTES=338738261
if [ -s "$DEST" ] && [ "$(stat -c %s "$DEST")" -eq "$EXPECTED_BYTES" ]; then
  echo "checkpoint already present: $DEST"; exit 0
fi
mkdir -p "$(dirname "$DEST")"
FID=$(curl -sf --max-time 60 "$API" -H "Accept: application/json" | python3 -c "import sys,json; print(json.load(sys.stdin)['rootFileId'])")
DL=$(curl -sf --max-time 60 "$API/files/$FID" -H "Accept: application/json" | python3 -c "import sys,json; print(json.load(sys.stdin)['downloadUrl'])")
echo "downloading $DL"
curl -fL --retry 5 --retry-delay 10 -C - -o "$DEST" "$DL"
SIZE=$(stat -c %s "$DEST")
[ "$SIZE" -eq "$EXPECTED_BYTES" ] || { echo "size mismatch: $SIZE != $EXPECTED_BYTES"; exit 1; }
echo "OK: $DEST ($SIZE bytes)"
