#!/usr/bin/env bash
# Measures the six reference configurations (`yue2 references`) on one request, CLAUDE.md
# protocol: 120 s cool-down before each point, outputs under .local-runs/bench.noindex/
# (Spotlight-free), competing-process check before each run. Usage:
#   Scripts/bench-references.sh <request.json> [semantic tokens, default 1750 = 70 s] [passes, default 1]
set -uo pipefail
cd "$(dirname "$0")/.."
REQ="${1:?request json}"; TOKENS="${2:-1750}"; PASSES="${3:-1}"
B=.xcodebuild/Build/Products/Release/yue2
STAMP=$(date +%Y%m%d-%H%M)
for pass in $(seq 1 "$PASSES"); do
  for ref in 4bit-fast 4bit-lean 8bit-fast 8bit-lean 16bit-fast 16bit-lean; do
    OUT=".local-runs/bench.noindex/ref-$STAMP-$ref-$pass"; rm -rf "$OUT"
    sleep 120
    busy=$(ps -Ao %cpu,comm | sort -nr | head -1)
    echo "=== $ref pass $pass $(date +%H:%M) top: $busy"
    "$B" generate --request "$REQ" --semantic-min-tokens "$TOKENS" --semantic-max-tokens "$TOKENS" \
      --reference "$ref" --profile --out "$OUT" 2>&1 | grep -E "generated|reference|rror"
  done
done
echo "REFERENCES DONE"
