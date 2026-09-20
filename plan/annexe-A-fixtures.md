## Annexe A — Squelette de `Scripts/reference/tiny_fixtures.py` (à compléter en 1.3)

```python
"""Tiny seeded fixtures for YuE2-mlx-swift tier-1 parity tests. Uses upstream yue2 classes only."""
import json, datetime, subprocess, torch
from safetensors.torch import save_file
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
from yue2 import nar
from yue2.sampling import distribution
from yue2.protocol import Sampling

torch.set_num_threads(1)
META = {"upstream_sha": "<sha du clone>", "generated": datetime.date.today().isoformat()}

def tiny_vae():
    common = dict(channels=1, c_mults=[1]*6, strides=[2,2,4,4,5,6], use_snake=True)
    cfg = YuE2VAEConfig(encoder_config=dict(common, in_channels=2, latent_dim=4),
                        decoder_config=dict(common, out_channels=2, latent_dim=2, snake_type="vanilla", final_tanh=False),
                        latent_dim=2)
    torch.manual_seed(231); model = YuE2VAE(cfg).eval()
    out = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("decoder.")}
    # merged weight-norm expectations (torch layout), computed by torch itself:
    for name, module in model.decoder.named_modules():
        if hasattr(module, "weight_g"):
            out[f"expected_merged.decoder.{name}.weight"] = module.weight.detach().clone()
    with torch.inference_mode():
        for T in (1, 15, 17, 33):
            z = torch.randn(1, 2, T, generator=torch.Generator().manual_seed(1000 + T))
            out[f"input_T{T}"] = z; out[f"expected_full_T{T}"] = model.decode(z)
        x = out["input_T33"]
        for i, layer in enumerate(model.decoder.layers):
            x = layer(x); out[f"tap_T33.after_layer_{i}"] = x.clone()
        for T, core in ((17, 8), (33, 16)):
            out[f"expected_tiled_T{T}_core{core}"] = model.decode_tiled(out[f"input_T{T}"], core_frames=core)
    save_file(out, "parity/tiny_vae.safetensors", metadata={**META, "kind": "tiny_vae", "seed": "231"})

def tiny_lm():
    cfg = YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=4, vocab_size=32, max_position_embeddings=128,
                     latent_dim=64, max_latent_frames=128)
    with torch.random.fork_rng():
        torch.manual_seed(173); model = YuE2ForCausalLM(cfg).eval()
    out = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        ids = torch.tensor([[3, 5, 8, 7, 4, 9]]); out["ids_full"] = ids
        out["logits_full"] = model(ids, use_cache=False).logits
        # per-layer taps: register forward hooks on model.model.layers[i]
        # chunks (0,2),(2,5),(5,6) with StaticKVCache -> logits_chunk_{i}
        # explicit positions [1,4,9,10] -> logits_explicit
        # rope: model.model.rotary_emb(torch.tensor([[0,1,4,9,10]])) -> rope_cos, rope_sin
        chunk = nar.Chunk([2, 3, 4, 5], torch.randn((5, 64), generator=torch.Generator().manual_seed(42)))
        engine = nar.CachedNAR(model, chunk)
        for raw in (20.0, 0.0, -2.3):
            out[f"velocity_raw{raw}"] = engine.velocity(chunk.noise, raw)
            out[f"time_emb_raw{raw}"] = model.time_embedder(model._shift_t_value(raw, "cpu", torch.float32).expand(1))
        out["nar_ar_tokens"] = torch.tensor(chunk.ar_tokens); out["nar_noise"] = chunk.noise
        chunk4 = nar.Chunk([2, 3, 4, 5], torch.randn((3, 64), generator=torch.Generator().manual_seed(381)))
        out["solve4_noise"] = chunk4.noise; out["solve4_expected"] = nar.CachedNAR(model, chunk4).solve(steps=4)
        # greedy: monkeypatch sampling constants (EOD 5, ABC_END 6, MUSIC_END 7, CODEC_OFFSET 8, CODEC_SIZE 20)
        # and call yue2.sampling.generate_tokens(model, prefix=[2,3,4], Sampling(temperature=0, min_tokens=0, max_tokens=8, ...), seed=0, phase="semantic", use_cuda_graph=False)
    save_file(out, "parity/tiny_lm.safetensors", metadata={**META, "kind": "tiny_lm", "seed": "173"})

# sampling fixture: logits = torch.randn(1, 184704, generator=seed 7); for each config call
# distribution(logits, Sampling(...), history, step, phase, legacy_off) and store scores_expected.
```

Le script real (`real_fixtures.py`) suit la même structure avec `YuE2ForCausalLM.from_pretrained(models_dir/"YuE2-3B", torch_dtype=torch.bfloat16).to("mps")`, `YuE2TextTokenizer`, `token_prefixes` et `YuE2VAE.from_pretrained(..., decoder_only=True, device="cpu")`.
