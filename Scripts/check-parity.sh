#!/usr/bin/env bash
# Parité tier 2 : Scripts/check-parity.sh {vae|lm|nar}. Exige YUE2_MODELS_DIR et
# le binaire Release (Scripts/check-build.sh d'abord). Journal : .local-runs/parity-<c>.log
set -uo pipefail
cd "$(dirname "$0")/.."
comp="${1:?composant requis : vae|lm|nar}"
: "${YUE2_MODELS_DIR:?YUE2_MODELS_DIR non défini}"
mkdir -p .local-runs
bin=.xcodebuild/Build/Products/Release/yue2
[ -x "$bin" ] || { echo "PARITY FAILED : binaire absent, lancer Scripts/check-build.sh"; exit 1; }
"$bin" parity "$comp" --models-dir "$YUE2_MODELS_DIR" > ".local-runs/parity-$comp.log" 2>&1
status=$?
grep "^PARITY" ".local-runs/parity-$comp.log" | head -40
last=$(grep "^PARITY OK\|^PARITY FAILED" ".local-runs/parity-$comp.log" | tail -1)
[ "$status" -eq 0 ] && [[ "$last" == PARITY\ OK* ]] && { echo "$last"; exit 0; }
echo "${last:-PARITY FAILED (statut $status)} — journal : .local-runs/parity-$comp.log"
exit 1
