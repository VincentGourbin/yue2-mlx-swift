#!/usr/bin/env bash
# Environnement de référence SheetSage2 (plan/14-sheetsage2.md) : Python 3.11, versions épinglées
# par requirements.txt amont, code amont dans reference/sheetsage2 à une révision fixe.
# Accès Hugging Face requis (dépôts gated m-a-p/SheetSage2 et m-a-p/MERT-v2-FullSong).
set -euo pipefail
cd "$(dirname "$0")/.."
REVISION=398b22834dac7dd05e09b9c4e40a39fc479ec502
command -v uv >/dev/null || { echo "uv requis (https://docs.astral.sh/uv/)"; exit 1; }
[ -d .venv-sheetsage2 ] || uv venv .venv-sheetsage2 --python 3.11
uv pip install --python .venv-sheetsage2/bin/python "huggingface-hub==0.36.0"
.venv-sheetsage2/bin/huggingface-cli download m-a-p/SheetSage2 --revision "$REVISION" --local-dir reference/sheetsage2 >/dev/null
uv pip install --python .venv-sheetsage2/bin/python -r reference/sheetsage2/requirements.txt soundfile
# transformers 4.45 ne copie qu'une partie des modules relatifs d'un dépôt local : on les copie tous.
M="$HOME/.cache/huggingface/modules/transformers_modules/sheetsage2"
mkdir -p "$M" && cp reference/sheetsage2/*.py "$M"/
echo "SHEETSAGE2 ENV OK — revision ${REVISION:0:12}"
