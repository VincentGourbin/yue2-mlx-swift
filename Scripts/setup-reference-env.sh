#!/usr/bin/env bash
# Environnement Python de référence (versions épinglées par upstream pyproject.toml)
# + installation du paquet yue2 depuis reference/yue + pytest upstream.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d reference/yue ] || Scripts/link-references.sh
command -v uv >/dev/null || { echo "uv requis (https://docs.astral.sh/uv/)"; exit 1; }
[ -d .venv-ref ] || uv venv .venv-ref --python 3.12
uv pip install --python .venv-ref/bin/python \
  "torch==2.10.0" "transformers==4.57.6" "huggingface-hub==0.36.2" "safetensors==0.7.0" \
  "tiktoken==0.12.0" "numpy==2.2.6" "soundfile==0.13.1" "accelerate==1.13.0" "pytest==9.0.3" "tokenizers"
uv pip install --python .venv-ref/bin/python -e reference/yue --no-deps
(cd reference/yue && ../../.venv-ref/bin/python -m pytest tests -q -x)
echo "REFERENCE ENV OK — upstream $(git -C reference/yue rev-parse --short HEAD)"
