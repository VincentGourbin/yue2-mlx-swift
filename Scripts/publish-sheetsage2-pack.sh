#!/usr/bin/env bash
# Prepares the merged fp16 SheetSage2 pack (plan/14-sheetsage2.md) for a Hugging Face upload and
# prints the `hf upload` commands. Never uploads by itself: publishing a CC-BY-NC-4.0 derivative is
# the author's decision and account.
#
# The pack is the upstream merge (`sheetsage2_fixtures.py convert` → $YUE2_MODELS_DIR/SheetSage2-merged)
# cast to float16, except what every profile keeps in float32 (mel front-end buffers, ConvNeXt
# front, layer-mix logits): loading it gives exactly the model `SheetSage2Model.load(profile:)`
# builds from the upstream release.
# Usage: Scripts/publish-sheetsage2-pack.sh [staging dir, default .local-runs/hf-sheetsage2] [repo id]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${YUE2_MODELS_DIR:?export YUE2_MODELS_DIR first}"
STAGE="${1:-.local-runs/hf-sheetsage2}"; REPO="${2:-VincentGOURBIN/sheetsage2-mlx-fp16}"
SRC="$YUE2_MODELS_DIR/SheetSage2-merged"
[ -f "$SRC/model.safetensors" ] || { echo "missing $SRC (run sheetsage2_fixtures.py convert)"; exit 1; }
mkdir -p "$STAGE"
.venv-sheetsage2/bin/python - "$SRC" "$STAGE" <<'PY'
import json, struct, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

src, stage = Path(sys.argv[1]), Path(sys.argv[2])
weights = load_file(src / "model.safetensors")
keep32 = ("encoder.feature_extractor.", "encoder.subsampling_module.")
out = {}
for key, value in sorted(weights.items()):
    float32 = key.startswith(keep32) or key == "layer_weight" or not value.is_floating_point()
    out[key] = value.contiguous() if float32 else value.to(torch.float16).contiguous()
meta = {"format": "sheetsage2-merged-fp16-v1", "upstream": "m-a-p/SheetSage2@398b22834dac7dd05e09b9c4e40a39fc479ec502",
        "parent": "m-a-p/MERT-v2-FullSong@d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"}
path = stage / "model.safetensors"
save_file(out, path, metadata=meta)
# Canonical header (same reason as Scripts/reference/_common.py): byte-reproducible file.
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); data = f.read()
canonical = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(canonical))); f.write(canonical); f.write(data)
config = json.loads((src / "config.json").read_text())
config["torch_dtype"] = "float16"
(stage / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
print(f"{len(out)} tensors, {path.stat().st_size / 1e9:.2f} GB")
PY
cp reference/sheetsage2/LICENSE "$STAGE/LICENSE"
shasum -a 256 "$STAGE/model.safetensors" | awk '{print $1}' > "$STAGE/model.safetensors.sha256"
cat > "$STAGE/README.md" <<'CARD'
---
license: cc-by-nc-4.0
base_model:
- m-a-p/SheetSage2
- m-a-p/MERT-v2-FullSong
tags: [mlx, swift, music-transcription, abc-notation, audio]
---
# SheetSage2 for MLX Swift — merged fp16

Music recording → editable ABC score, on Apple silicon (Mac, iPhone), with
[yue2-mlx-swift](https://github.com/VincentGourbin/yue2-mlx-swift) (`SheetSage2Core`, `yue2 transcribe`).
The score feeds a YuE2 cover with `cot: melody`.

## Source and changes
A derivative of **SheetSage2** ([m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2), revision
`398b22834dac`) and of its encoder parent **MERT2** ([m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong),
revision `d8ba1c745e73`, the one SheetSage2 pins), both by m-a-p, CC BY-NC 4.0. Changes made here:
- the LoRA adapters merged into MERT2's attention projections (upstream `merge_lora`, float32);
- weights cast to float16, except the mel front-end buffers, the ConvNeXt front and the layer-mix
  logits, kept in float32 as the Swift port computes them;
- packaged as one safetensors file (PyTorch key layout, `weights_format: merged`), SHA-256 in
  `model.safetensors.sha256`.
No retraining, no other modification. The weights are not endorsed by m-a-p.

## Use
```bash
yue2 download --model sheetsage2-fp16             # 1.4 GB, SHA-256 verified
yue2 transcribe --audio song.m4a --profile 16bit-lean --out run/score
```
Fidelity: with the Swift port, this pack gives the same tokens and ABC as the upstream release merged
at load (`16bit-*` profiles); in float32 the port is byte-identical to the upstream PyTorch code.
Measurements and profiles: the repository's `docs/References.md`.

## License
CC BY-NC 4.0 (see `LICENSE`, copied from the upstream release): non-commercial use only, with
attribution to MERT2 and SheetSage2 (m-a-p) and their repositories above.
CARD
du -h "$STAGE/model.safetensors" | cut -f1; cat "$STAGE/model.safetensors.sha256"
echo
echo "To publish (needs \`hf auth login\`):"
echo "  hf repo create $REPO --type model"
echo "  hf upload $REPO $STAGE ."
