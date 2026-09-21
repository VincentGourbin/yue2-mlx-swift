"""Parity check: Core AI asset output vs PyTorch fp32, on freshly-seeded random latents at each
enumerated shape (the tiny parity fixtures' T=40/1100 aren't in the enumerated set — 288/544/1056
per plan/13-ios-backend.md §13.3 — so this checks the shapes Core AI actually specializes for,
not the existing fixtures).

Usage: .venv-ref/bin/python Scripts/coreai/check_asset.py --models-dir $YUE2_MODELS_DIR
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from coreai.runtime import AIModel
from coreai.runtime._ndarray import NDArray


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    noise = reference.astype(np.float64) - test.astype(np.float64)
    signal_power = np.mean(reference.astype(np.float64) ** 2)
    noise_power = np.mean(noise**2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def cosine(reference: np.ndarray, test: np.ndarray) -> float:
    a, b = reference.astype(np.float64).ravel(), test.astype(np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def _check_one(model, entrypoint: str, ref_module: torch.nn.Module, example: torch.Tensor) -> tuple[float, float]:
    fn = model.load_function(entrypoint)
    result = await fn(inputs={"z": NDArray(example.numpy())})
    coreai_out = result["out"].numpy()
    with torch.no_grad():
        torch_out = ref_module(example.float()).numpy()
    return snr_db(torch_out, coreai_out), cosine(torch_out, coreai_out)


async def main_async(models_dir: Path) -> int:
    from yue2.modeling_vae import YuE2VAE

    model = YuE2VAE.from_pretrained(models_dir / "YuE2-Vae", decoder_only=False, device="cpu")
    decoder_fp32 = model.decoder.eval()
    encoder_fp32 = model.encoder.eval()

    decoder_asset = await AIModel.load(models_dir / "YuE2-Vae" / "coreai" / "YuE2VaeDecoder.aimodel")
    encoder_asset = await AIModel.load(models_dir / "YuE2-Vae" / "coreai" / "YuE2VaeEncoder.aimodel")

    worst_snr = float("inf")
    worst_cos = 1.0
    torch.manual_seed(11)
    for frames in (288, 544, 1056):
        z = torch.randn(1, 64, frames, dtype=torch.float16)
        snr, cos = await _check_one(decoder_asset, f"len{frames}", decoder_fp32, z)
        print(f"COREAI VAE decoder len{frames}: snr={snr} cos={cos}")
        worst_snr, worst_cos = min(worst_snr, snr), min(worst_cos, cos)

    for frames in (288, 544, 1056):
        samples = frames * 1920
        audio = torch.randn(1, 2, samples, dtype=torch.float16)
        snr, cos = await _check_one(encoder_asset, f"len{samples}", encoder_fp32, audio)
        print(f"COREAI VAE encoder len{samples}: snr={snr} cos={cos}")
        worst_snr, worst_cos = min(worst_snr, snr), min(worst_cos, cos)

    print(f"COREAI VAE OK snr={worst_snr} cos={worst_cos}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True, type=Path)
    args = parser.parse_args()
    return asyncio.run(main_async(args.models_dir))


if __name__ == "__main__":
    sys.exit(main())
