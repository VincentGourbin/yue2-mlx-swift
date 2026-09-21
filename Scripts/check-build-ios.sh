#!/usr/bin/env bash
# Build du produit YuE2Core pour iOS (generic/platform=iOS) avec xcodebuild ; verdict en
# dernière ligne. Journal complet : .local-runs/build-ios.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local-runs
SCHEME="${1:-YuE2Core}"
xcodebuild -scheme "$SCHEME" -derivedDataPath .xcodebuild \
  -destination 'generic/platform=iOS' build > .local-runs/build-ios.log 2>&1
status=$?
errors=$(grep -c "error:" .local-runs/build-ios.log || true)
if [ "$status" -eq 0 ] && grep -q "BUILD SUCCEEDED" .local-runs/build-ios.log; then
  echo "BUILD IOS OK"
  exit 0
fi
grep -n "error:" .local-runs/build-ios.log | head -20
echo "BUILD IOS FAILED ($errors erreurs) — journal : .local-runs/build-ios.log"
exit 1
