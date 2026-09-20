#!/usr/bin/env bash
# Crée reference/ : liens symboliques vers les dépôts frères (sources à copier)
# et clone superficiel de la référence Python YuE. Tout est gitignoré.
set -euo pipefail
cd "$(dirname "$0")/.."
DEV="${YUE2_SIBLINGS_DIR:-$(cd .. && pwd)}"
mkdir -p reference
link() { [ -e "reference/$1" ] || ln -s "$DEV/$2" "reference/$1"; }
link flux-2   flux-2-swift-mlx
link gemma-4  gemma-4-swift-mlx
link h3       h3-swift-mlx
link ltx      ltx-video-swift-mlx
link voxtral  convertvoxtral
link qwen38   qwen38-mlx-swift
link void     netflix-void-swift-mlx
link profiler swift-mlx-profiler
if [ ! -d reference/yue ]; then
  git clone --depth 1 https://github.com/multimodal-art-projection/YuE.git reference/yue
fi
echo "upstream YuE commit: $(git -C reference/yue rev-parse --short HEAD)"
ls -l reference
