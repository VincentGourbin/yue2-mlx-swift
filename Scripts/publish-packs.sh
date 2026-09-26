#!/usr/bin/env bash
# Prepares the reference prequantized packs for a Hugging Face upload (docs/Weights.md) and
# prints the `hf upload` commands. Never uploads by itself: publishing CC-BY-NC-4.0 derivatives
# is the author's decision and account.
# Usage: Scripts/publish-packs.sh [staging dir, default .local-runs/hf-packs] [repo id, default VincentGourbin/yue2-mlx-packs]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${YUE2_MODELS_DIR:?export YUE2_MODELS_DIR first}"
STAGE="${1:-.local-runs/hf-packs}"; REPO="${2:-VincentGourbin/yue2-mlx-packs}"
SRC="$YUE2_MODELS_DIR/YuE2-3B/mlx-prequantized"
mkdir -p "$STAGE"
for pack in int4-mixed-head qint8-all-head int4-head; do
  f="$SRC/$pack/model.safetensors"
  [ -f "$f" ] || { echo "missing $f (run a generate with that preset once)"; continue; }
  mkdir -p "$STAGE/$pack"; ln -sf "$f" "$STAGE/$pack/model.safetensors"
  shasum -a 256 "$f" | awk '{print $1}' > "$STAGE/$pack/model.safetensors.sha256"
  echo "$pack: $(du -h "$f" | cut -f1) sha256 $(cat "$STAGE/$pack/model.safetensors.sha256")"
done
cat > "$STAGE/README.md" <<'CARD'
---
license: cc-by-nc-4.0
base_model: m-a-p/YuE2-3B
tags: [mlx, swift, music-generation, quantized]
---
# YuE2 MLX packs — poids pré-quantifiés pour yue2-mlx-swift

Dérivés de m-a-p/YuE2-3B (CC-BY-NC-4.0) pour https://github.com/VincentGourbin/yue2-mlx-swift (MIT).
Usage non commercial, attribution m-a-p. Format safetensors MLX (poids packés uint32 + scales + biases,
groupe 64, affine), métadonnées `format=yue2-prequantized-v1`. Voir docs/Weights.md du dépôt.
CARD
echo
echo "To publish (needs \`hf auth login\`):"
echo "  hf repo create $REPO --type model"
echo "  hf upload $REPO $STAGE . --include '*.safetensors' --include '*.sha256' --include 'README.md'"
