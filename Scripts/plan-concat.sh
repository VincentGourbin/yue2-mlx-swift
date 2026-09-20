#!/usr/bin/env bash
# Assemble le plan complet sur la sortie standard (lecture humaine). Ne pas
# rediriger vers un fichier du dépôt : le plan doit rester découpé pour l'agent.
cd "$(dirname "$0")/.."
sed -n '1,/^---$/p' PLAN.md
for f in plan/*.md; do echo; echo "---"; echo; cat "$f"; done
