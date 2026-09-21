"""Exports the YuE2 VAE decoder (and encoder) to Core AI (.aimodel), fp16, enumerated shapes.

Usage: .venv-ref/bin/python Scripts/coreai/export_vae.py --models-dir $YUE2_MODELS_DIR

Weight-norm is fused (torch.nn.utils.remove_weight_norm) BEFORE export (pitfall #9) — the
released checkpoint's WNConv1d/WNConvTranspose1d only carry weight_g/weight_v, and Core AI's
convolution op expects a single plain weight tensor like any other backend.

Uses the fiche's documented fallback (enumerated shapes) rather than dynamic shapes
(torch.export.Dim(min=..., max=...)): one fixed-shape program per enumerated size, each its
own entrypoint on the same asset (`TorchConverter.add_exported_program` called once per size,
`entrypoint_name=f"len{length}"`).

DECODER_FRAMES are *latent* frames (`z: [1, 64, T]`, T ∈ {288, 544, 1056}, per plan/13-ios-backend.md
§13.3). ENCODER_SAMPLES are raw *audio* samples (`audio: [1, 2, S]`) — S must be a multiple of
the VAE's downsampling ratio (1920). An earlier version of this script reused DECODER_FRAMES's
small numbers for the encoder too, starving its last downsampling stage (stride 6, kernel 12) of
enough input to even run — that threw "Kernel size can't be greater than actual input size", a
units bug in this script's example inputs, not a Core AI or torch.export limitation.
"""

import argparse
import sys
from pathlib import Path

import coreai_torch
import torch
from coreai_torch import TorchConverter

DOWNSAMPLING_RATIO = 1920
DECODER_FRAMES = (288, 544, 1056)
ENCODER_SAMPLES = tuple(frames * DOWNSAMPLING_RATIO for frames in DECODER_FRAMES)


def _fuse_weight_norm(module: torch.nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "weight_g"):
            torch.nn.utils.remove_weight_norm(child)


def _export_module(
    name: str, module: torch.nn.Module, example_channels: int, lengths: tuple[int, ...], out_path: Path
) -> None:
    module = module.eval().half()
    _fuse_weight_norm(module)

    converter = TorchConverter()
    for length in lengths:
        example = torch.randn(1, example_channels, length, dtype=torch.float16)
        ep = torch.export.export(module, (example,))
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        converter = converter.add_exported_program(
            ep, input_names=["z"], output_names=["out"], entrypoint_name=f"len{length}"
        )

    program = converter.to_coreai()
    program.optimize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(out_path)
    print(f"exported {name} -> {out_path} ({', '.join(f'len{n}' for n in lengths)})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True, type=Path)
    args = parser.parse_args()

    from yue2.modeling_vae import YuE2VAE

    model = YuE2VAE.from_pretrained(args.models_dir / "YuE2-Vae", decoder_only=False, device="cpu")

    coreai_dir = args.models_dir / "YuE2-Vae" / "coreai"
    _export_module(
        "decoder", model.decoder, example_channels=64, lengths=DECODER_FRAMES,
        out_path=coreai_dir / "YuE2VaeDecoder.aimodel",
    )
    _export_module(
        "encoder", model.encoder, example_channels=2, lengths=ENCODER_SAMPLES,
        out_path=coreai_dir / "YuE2VaeEncoder.aimodel",
    )

    print("COREAI VAE EXPORT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
