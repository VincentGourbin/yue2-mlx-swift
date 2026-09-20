#!/usr/bin/env bash
# Lance la suite (ou une suite : Scripts/check-tests.sh ProtocolTests) via
# run-tests.sh ; verdict en dernière ligne. Journal : .local-runs/tests.log
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p .local-runs
if [ $# -ge 1 ]; then
  Scripts/run-tests.sh "-only-testing:YuE2Tests/$1" > .local-runs/tests.log 2>&1
else
  # No filter: call with zero extra args. `"${args[@]}"` on an empty array trips macOS's
  # bash 3.2 `set -u` ("unbound variable") even after `args=()` — the fix isn't a guard
  # expression, it's not referencing an empty array under nounset at all.
  Scripts/run-tests.sh > .local-runs/tests.log 2>&1
fi
status=$?
if [ "$status" -eq 0 ] && grep -q "TEST SUCCEEDED\|✔ Test run" .local-runs/tests.log; then
  n=$(grep -o "Test run with [0-9]* test" .local-runs/tests.log | tail -1 | grep -o "[0-9]*")
  # Ne compter que les tests sautés par swift-testing (↷) ou XCTest, pas les lignes
  # de log d'xcodebuild du type « Metadata extraction skipped ».
  skipped=$(grep -c "↷ Test \|Test Case .* skipped" .local-runs/tests.log || true)
  echo "TESTS OK (${n:-?} tests, ${skipped} skippés)"
  exit 0
fi
# Le marqueur de succès explicite (ci-dessus) est prioritaire : le log contient aussi du
# bruit système macOS (CoreData/AddressBook, …) qui inclut la sous-chaîne « error: » sans
# rapport avec les tests, ce qui déclenchait un faux négatif même quand tout passait.
if grep -q "TEST FAILED\|✘ Test run\|error:" .local-runs/tests.log; then
  grep -n "✘\|error:\|failed" .local-runs/tests.log | head -20
  echo "TESTS FAILED — journal : .local-runs/tests.log"
  exit 1
fi
tail -20 .local-runs/tests.log
echo "TESTS FAILED (statut $status) — journal : .local-runs/tests.log"
exit 1
