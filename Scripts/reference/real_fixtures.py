"""Real-checkpoint parity fixtures (tier 2, gitignored, `$YUE2_MODELS_DIR/parity/`).

Unlike `tiny_fixtures.py` (tier 1, committed, random weights), these run the actual released
checkpoints on CPU fp32 and are regenerated locally, never committed (piège n°16).

Usage: `.venv-ref/bin/python Scripts/reference/real_fixtures.py vae --models-dir "$YUE2_MODELS_DIR"`

`lm` (T-2.9) runs the real checkpoint bf16 on MPS (matches `pipeline.py`'s `_load_model`);
`nar`, `song` land in T-3.5, T-3.6.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from _common import META, _YUE_ROOT, save_file_deterministic


def vae(models_dir: Path) -> None:
    from yue2.modeling_vae import YuE2VAE

    torch.set_num_threads(1)
    # decoder_only=False (T-5.1): also builds/loads the encoder, needed for the encode() parity
    # fixture below. Same fp32 CPU checkpoint either way.
    model = YuE2VAE.from_pretrained(models_dir / "YuE2-Vae", decoder_only=False, device="cpu")
    out = {}
    with torch.inference_mode():
        latent_40 = torch.randn(1, 64, 40, generator=torch.Generator().manual_seed(11))
        out["latent_40"] = latent_40
        out["audio_40"] = model.decode(latent_40)

        latent_1100 = torch.randn(1, 64, 1100, generator=torch.Generator().manual_seed(12))
        out["latent_1100"] = latent_1100
        out["audio_1100_full"] = model.decode(latent_1100)
        out["audio_1100_tiled"] = model.decode_tiled(latent_1100, core_frames=1024)

        # T-5.1: encode() parity. 2 s of random stereo audio (seeded, not a real song -- this is
        # a numeric-parity fixture, not the qualitative round-trip demo, which uses Vincent's own
        # file directly through the CLI, never committed/fixtured).
        audio_encode = torch.randn(1, 2, 96_000, generator=torch.Generator().manual_seed(13))
        out["audio_encode"] = audio_encode
        out["encoded_expected"] = model.encode(audio_encode)

    parity_dir = models_dir / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    destination = parity_dir / "vae.safetensors"
    save_file_deterministic(out, destination, metadata={**META, "kind": "real_vae"})
    print(f"real vae fixture written to {destination}")


def lm(models_dir: Path) -> None:
    """ABC-phase and semantic-phase prefixes (real `song.json`/`score.abc`), their last-token
    logits, layer-bisection taps, and 8-token greedy decodes — mirrors `plan/08-tests.md`'s
    `lm` row. `prefix_abc`/`prefix_semantic` are stored verbatim (not just the ABC ids used to
    build the semantic one) so `RealLMParityTests` never has to re-derive them from
    `reference/yue/examples/score.abc` (gitignored, Python-tooling-only) at Swift test time.
    """
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import Sampling, SongRequest, token_prefixes
    from yue2.sampling import generate_tokens
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    torch.set_num_threads(1)
    model_dir = models_dir / "YuE2-3B"
    model = YuE2ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("mps").eval()
    tokenizer = YuE2TextTokenizer(model_dir / "qwen.tiktoken")

    request = SongRequest(**json.loads((_YUE_ROOT / "examples" / "song.json").read_text()))
    score_abc = (_YUE_ROOT / "examples" / "score.abc").read_text()
    greedy_sampling = Sampling(temperature=0, top_p=1, top_k=1, repetition_penalty=1,
                                penalty_window=1, min_tokens=0, max_tokens=8)
    watched_layers = {0, 7, 14, 27}
    out = {}

    with torch.inference_mode():
        prefix_abc = token_prefixes(request, tokenizer)
        taps = {}
        hooks = [layer.register_forward_hook(
                     lambda _m, _i, output, idx=i: taps.__setitem__(idx, output.detach().float().cpu().clone()))
                 for i, layer in enumerate(model.model.layers) if i in watched_layers]
        logits_abc = model(torch.tensor([prefix_abc], device="mps"), use_cache=False).logits
        for hook in hooks:
            hook.remove()
        out["prefix_abc"] = torch.tensor(prefix_abc, dtype=torch.int32)
        out["logits_last_abc"] = logits_abc[:, -1, :].detach().float().cpu()
        for i, tap in taps.items():
            out[f"hidden_after_layer_{i}"] = tap

        greedy_abc, _, _ = generate_tokens(model, prefix_abc, greedy_sampling, seed=0,
                                           phase="abc", use_cuda_graph=False)
        out["greedy_abc"] = torch.tensor(greedy_abc, dtype=torch.int64)

        abc_ids = tokenizer.encode(score_abc)
        prefix_semantic = token_prefixes(request, tokenizer, abc_ids)
        logits_semantic = model(torch.tensor([prefix_semantic], device="mps"), use_cache=False).logits
        out["prefix_semantic"] = torch.tensor(prefix_semantic, dtype=torch.int32)
        out["logits_last_semantic"] = logits_semantic[:, -1, :].detach().float().cpu()

        greedy_semantic, _, _ = generate_tokens(model, prefix_semantic, greedy_sampling, seed=0,
                                                phase="semantic", use_cuda_graph=False)
        out["greedy_semantic"] = torch.tensor(greedy_semantic, dtype=torch.int64)

    parity_dir = models_dir / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    destination = parity_dir / "lm.safetensors"
    save_file_deterministic(out, destination, metadata={**META, "kind": "real_lm"})
    print(f"real lm fixture written to {destination}")


def nar(models_dir: Path) -> None:
    """Reuses lm.safetensors's semantic prefix and greedy tokens as this chunk's AR tokens,
    instead of reloading the model and re-generating a fresh one (saves a second ~1-3 min MPS
    load+prefill). `greedy_semantic`'s ids are already absolute (CODEC_OFFSET baked in, since
    they came out of the real RestrictedHead sampling path) so no round-trip through raw codec
    space is needed to rebuild `ar_tokens`. Only 8 tokens were generated for T-2.9's greedy
    fixture; they are duplicated to reach the 16 frames this fiche calls for.
    """
    from yue2 import nar as nar_mod
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import MUSIC_END

    torch.set_num_threads(1)
    model_dir = models_dir / "YuE2-3B"
    model = YuE2ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("mps").eval()

    lm_data = load_file(models_dir / "parity" / "lm.safetensors")
    prefix_semantic = lm_data["prefix_semantic"].to(torch.int64).tolist()
    greedy_semantic = lm_data["greedy_semantic"].to(torch.int64).tolist()
    codec_tokens = (greedy_semantic * 2)[:16]
    ar_tokens = prefix_semantic + codec_tokens + [MUSIC_END]
    noise = torch.randn((16, 64), dtype=torch.float32, generator=torch.Generator().manual_seed(42))
    chunk = nar_mod.Chunk(ar_tokens, noise)

    out = {"ar_tokens": torch.tensor(ar_tokens, dtype=torch.int64), "noise": noise}
    watched_layers = {0, 7, 14, 27}
    with torch.inference_mode():
        engine = nar_mod.CachedNAR(model, chunk)
        # `solve()` moves `chunk.noise` to the model's device/dtype internally; calling
        # `velocity()` directly (to record intermediate taps) needs that done up front.
        state = chunk.noise.to(device=engine.device, dtype=engine.dtype)
        for raw in (20.0, 0.0, -2.3):
            out[f"velocity_raw{raw}"] = engine.velocity(state, raw).float().cpu()

        # CachedNAR.velocity's body, copied to record bisection taps at raw=20.0 (mirrors
        # tiny_lm.py's tapped_velocity, T-1.3).
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model._shift_t_value(20.0, engine.device, engine.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(engine.nar_length))[None]
        x = x + engine.pos_emb
        out["nar_after_pos"] = x.float().cpu().clone()
        for i, (layer, (ar_k, ar_v)) in enumerate(zip(model.model.layers, engine.cache)):
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), engine.cos, engine.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = engine._attention(q[0], k, v)
            x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
            if i in watched_layers:
                out[f"nar_after_layer_{i}"] = x.float().cpu().clone()

        out["solve4_expected"] = engine.solve(steps=4)
        # T-6.2 (E2 sensitivity sweep): the production step count is 32, not 4 -- solve4 was a
        # fast parity smoke test. A quantized NAR's error compounds over all 64 velocity() calls
        # (32 steps x 2 midpoint evals), so judging a quantization preset needs the full trajectory.
        out["solve32_expected"] = engine.solve(steps=32)
        engine.close()

    parity_dir = models_dir / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    destination = parity_dir / "nar.safetensors"
    save_file_deterministic(out, destination, metadata={**META, "kind": "real_nar"})
    print(f"real nar fixture written to {destination}")


def song(models_dir: Path) -> None:
    """Runs the real end-to-end pipeline with the same short-run knobs as the Swift build's own
    smoke test/manual validation (`yue2 generate --abc-max-tokens 64 --semantic-max-tokens 100
    --semantic-min-tokens 0 --ode-steps 4`) on the same request and seed, for an optional
    side-by-side listen (G-4). Takes ~6-10 min on MPS for the full-length demo the fiche
    describes separately; this short variant (~1-2 min) is what CI/local validation actually runs.
    """
    from yue2.pipeline import YuE2Pipeline
    from yue2.protocol import GenerationConfig, Sampling

    torch.set_num_threads(1)
    generation_config = GenerationConfig(
        abc=Sampling(temperature=0.7, top_p=0.9, top_k=30, repetition_penalty=1.005,
                     penalty_window=100, min_tokens=0, max_tokens=64),
        semantic=Sampling(temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2,
                          penalty_window=50, min_tokens=0, max_tokens=100),
        ode_steps=4,
    )
    pipe = YuE2Pipeline.from_pretrained(
        models_dir / "YuE2-3B", vae=models_dir / "YuE2-Vae", device="mps",
        generation_config=generation_config, progress=False)
    request = json.loads((_YUE_ROOT / "examples" / "song.json").read_text())
    result = pipe(**request)
    destination = models_dir / "parity" / "song"
    result.save_artifacts(destination)
    print(f"real song fixture written to {destination}")


COMPONENTS = {"vae": vae, "lm": lm, "nar": nar, "song": song}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=sorted(COMPONENTS))
    parser.add_argument("--models-dir", required=True)
    args = parser.parse_args()
    COMPONENTS[args.component](Path(args.models_dir))
