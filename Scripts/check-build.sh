#!/usr/bin/env bash
# Build Release du produit `yue2` avec xcodebuild ; verdict en dernière ligne.
# Journal complet : .local-runs/build.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local-runs
SCHEME="${1:-yue2}"
xcodebuild -scheme "$SCHEME" -configuration Release -derivedDataPath .xcodebuild \
  -destination 'platform=macOS' -skipMacroValidation build > .local-runs/build.log 2>&1
status=$?
errors=$(grep -c "error:" .local-runs/build.log || true)
if [ "$status" -eq 0 ] && grep -q "BUILD SUCCEEDED" .local-runs/build.log; then
  echo "BUILD OK"
  exit 0
fi
grep -n "error:" .local-runs/build.log | head -20
echo "BUILD FAILED ($errors erreurs) — journal : .local-runs/build.log"
exit 1
