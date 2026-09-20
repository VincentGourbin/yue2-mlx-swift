"""Tiny YuE2VAE fixture (seed 231): decoder weights + merged weight-norm expectations
+ full/tiled decode + per-decoder-layer taps. Uses the upstream YuE2VAE unmodified."""
import torch
from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig

from _common import META, save_file_deterministic


def tiny_config():
    common = dict(channels=1, c_mults=[1] * 6, strides=[2, 2, 4, 4, 5, 6], use_snake=True)
    return YuE2VAEConfig(
        encoder_config=dict(common, in_channels=2, latent_dim=4),
        decoder_config=dict(common, out_channels=2, latent_dim=2, snake_type="vanilla", final_tanh=False),
        latent_dim=2,
    )


def build():
    torch.manual_seed(231)
    model = YuE2VAE(tiny_config()).eval()
    out = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("decoder.")}
    out.update({k: v.clone() for k, v in model.state_dict().items() if k.startswith("encoder.")})
    for name, module in model.decoder.named_modules():
        if hasattr(module, "weight_g"):
            out[f"expected_merged.decoder.{name}.weight"] = module.weight.detach().clone()
    for name, module in model.encoder.named_modules():
        if hasattr(module, "weight_g"):
            out[f"expected_merged.encoder.{name}.weight"] = module.weight.detach().clone()
    with torch.inference_mode():
        for length in (1, 15, 17, 33):
            z = torch.randn(1, 2, length, generator=torch.Generator().manual_seed(1000 + length))
            out[f"input_T{length}"] = z
            out[f"expected_full_T{length}"] = model.decode(z)
        x = out["input_T33"]
        for i, layer in enumerate(model.decoder.layers):
            x = layer(x)
            out[f"tap_T33.after_layer_{i}"] = x.clone()
        for length, core in ((17, 8), (33, 16)):
            out[f"expected_tiled_T{length}_core{core}"] = model.decode_tiled(out[f"input_T{length}"], core_frames=core)

        # Encoder: T-5.1. downsampling_ratio for the tiny config is the same 1920 as real
        # (product of the shared strides list), so audio_len_1920*T mirrors the decoder's T naming.
        # Only two lengths (unlike the decoder's four): raw audio is 1920x heavier per frame than
        # a latent, so a matching fixture would blow the <5 MB parity/ budget (T-1.3 note).
        ratio = model.config.downsampling_ratio
        for length in (1, 9):
            audio = torch.randn(1, 2, ratio * length, generator=torch.Generator().manual_seed(2000 + length))
            out[f"audio_T{length}"] = audio
            out[f"expected_encoded_T{length}"] = model.encode(audio)
        # Layers 0/1 (the initial plain conv and the first EncoderBlock) run at full input
        # resolution and dominate the byte budget; they also reuse already-validated ResidualUnit/
        # SnakeBeta/weight-norm-merge code, so bisection starts at layer 2 to stay under 5 MB.
        ax = out["audio_T9"]
        for i, layer in enumerate(model.encoder.layers):
            ax = layer(ax)
            if i >= 2:
                out[f"tap_encode_T9.after_layer_{i}"] = ax.clone()
    return out


def save(path="parity/tiny_vae.safetensors"):
    save_file_deterministic(build(), path, metadata={**META, "kind": "tiny_vae", "seed": "231"})


if __name__ == "__main__":
    save()
