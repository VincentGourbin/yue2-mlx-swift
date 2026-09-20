#!/usr/bin/env bash
# Vérifie les fixtures tiny committées et l'environnement de référence Python.
set -uo pipefail
cd "$(dirname "$0")/.."
missing=0
for f in parity/tiny_vae.safetensors parity/tiny_lm.safetensors parity/sampling.safetensors parity/protocol.json parity/tokenizer_corpus.json; do
  [ -f "$f" ] || { echo "FIXTURES MISSING $f"; missing=1; }
done
[ -x .venv-ref/bin/python ] || { echo "FIXTURES MISSING .venv-ref (Scripts/setup-reference-env.sh)"; missing=1; }
[ "$missing" -eq 0 ] || exit 1
.venv-ref/bin/python - <<'PY' || exit 1
from safetensors import safe_open
for name in ("tiny_vae", "tiny_lm", "sampling"):
    with safe_open(f"parity/{name}.safetensors", framework="pt") as f:
        meta = f.metadata() or {}
        assert meta.get("kind") == name, f"{name}: metadata.kind manquant"
        assert "seed" in meta, f"{name}: metadata.seed manquant"
        print(f"{name}: {len(list(f.keys()))} tenseurs, seed {meta['seed']}, upstream {meta.get('upstream_sha','?')}")
PY
echo "FIXTURES OK"
