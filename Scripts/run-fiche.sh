#!/usr/bin/env bash
# Exécute UNE fiche avec pi en mode non interactif, journal dans .local-runs/fiches/.
#   Scripts/run-fiche.sh T-1.0            # nouvelle session
#   Scripts/run-fiche.sh T-1.0 --continue # reprend la session précédente (fiche bloquée puis débloquée)
set -euo pipefail
cd "$(dirname "$0")/.."
fiche="${1:?fiche requise, ex. T-1.0}"; shift || true
[ -f "tasks/$fiche.md" ] || { echo "tasks/$fiche.md introuvable"; exit 1; }
: "${PI_MODEL:?PI_MODEL non défini : passer par Scripts/launch-with-local.sh <fiche>}"
mkdir -p .local-runs/fiches
log=".local-runs/fiches/$fiche-$(date +%Y%m%d-%H%M%S).log"
start=$(date +%s)
pi --approve -p --thinking low --model "$PI_MODEL" "$@" \
  "Exécute la fiche tasks/$fiche.md en suivant AGENTS.md. Commence par lire tasks/STATE.md puis la fiche. Termine par la ligne FICHE $fiche VALIDÉE ou FICHE $fiche BLOQUÉE." \
  2>&1 | tee "$log"
echo "--- durée : $(( $(date +%s) - start )) s · journal : $log"
grep -E "FICHE $fiche (VALIDÉE|BLOQUÉE)" "$log" | tail -1 || echo "FICHE $fiche : pas de ligne de fin (budget épuisé ?)"
