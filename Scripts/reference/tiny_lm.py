"""Tiny YuE2ForCausalLM fixture (seed 173, config from tests/test_nar.py): full state
dict, AR bisection taps (per-layer hidden states, chunked KV cache, explicit positions,
RoPE), and NAR taps (velocity, midpoint solve, greedy/CFG token generation). Uses the
upstream YuE2ForCausalLM/StaticKVCache/nar classes unmodified; NAR taps are produced by
copying CachedNAR.velocity's body inline purely to record its intermediate tensors."""
import torch
import torch.nn.functional as F
from yue2 import nar
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM, StaticKVCache
from yue2.protocol import Sampling
import yue2.sampling as sampling_mod

from _common import META, save_file_deterministic


def tiny_config():
    return YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                       vocab_size=32, max_position_embeddings=128,
                       latent_dim=64, vae_latent_dim=64, max_latent_frames=128)


def tapped_velocity(engine, state, raw_t, out, prefix):
    """CachedNAR.velocity (yue2/nar.py), copied to record its intermediate taps."""
    model = engine.model
    x_nar = F.pad(state, (0, 0, 1, 1))
    shifted = model._shift_t_value(raw_t, engine.device, engine.dtype)
    x = model.vae2llm(x_nar[None])
    out[f"{prefix}.after_vae2llm"] = x.clone()
    x = x + model.time_embedder(shifted.expand(engine.nar_length))[None]
    out[f"{prefix}.after_time"] = x.clone()
    x = x + engine.pos_emb
    out[f"{prefix}.after_pos"] = x.clone()
    for i, (layer, (ar_k, ar_v)) in enumerate(zip(model.model.layers, engine.cache)):
        q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), engine.cos, engine.sin)
        k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
        h = engine._attention(q[0], k, v)
        x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
        x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        out[f"{prefix}.after_layer_{i}"] = x.clone()
    return model.llm2vae(model.model.norm(x))[0, 1:-1]


def greedy_and_cfg(model):
    names = ("EOD", "ABC_END", "MUSIC_END", "CODEC_OFFSET", "CODEC_SIZE")
    originals = {n: getattr(sampling_mod, n) for n in names}
    for n, v in zip(names, (5, 6, 7, 8, 20)):
        setattr(sampling_mod, n, v)
    try:
        sampling = Sampling(temperature=0, top_p=1, top_k=1, repetition_penalty=1,
                             penalty_window=1, min_tokens=0, max_tokens=8)
        tokens, _, truncated = sampling_mod.generate_tokens(
            model, [2, 3, 4], sampling, seed=0, phase="semantic", use_cuda_graph=False)
        cfg_tokens, _, _ = sampling_mod.generate_tokens(
            model, [2, 3, 4], sampling, seed=0, phase="semantic",
            negative=[2, 3], cfg_scale=1.5, use_cuda_graph=False)
    finally:
        for n, v in originals.items():
            setattr(sampling_mod, n, v)
    return tokens, truncated, cfg_tokens


def build():
    with torch.random.fork_rng():
        torch.manual_seed(173)
        model = YuE2ForCausalLM(tiny_config()).eval()
    out = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        ids = torch.tensor([[3, 5, 8, 7, 4, 9]])
        out["ids_full"] = ids
        hidden_taps = {}
        hooks = [layer.register_forward_hook(
            lambda _m, _i, output, idx=i: hidden_taps.__setitem__(idx, output.clone()))
            for i, layer in enumerate(model.model.layers)]
        out["logits_full"] = model(ids, use_cache=False).logits
        for hook in hooks:
            hook.remove()
        for i, tap in hidden_taps.items():
            out[f"hidden_after_layer_{i}"] = tap

        cache = StaticKVCache(num_layers=2, batch_size=1, num_kv_heads=2, max_seq_len=6,
                               head_dim=4, dtype=next(model.parameters()).dtype, device=ids.device)
        for i, (start, end) in enumerate(((0, 2), (2, 5), (5, 6))):
            cache_position = torch.arange(start, end)
            logits = model(ids[:, start:end], past_key_values=cache, use_cache=True,
                           cache_position=cache_position, logits_to_keep=0).logits
            out[f"logits_chunk_{i}"] = logits

        positions = torch.tensor([[1, 4, 9, 10]])
        out["positions_explicit"] = positions
        out["logits_explicit"] = model(ids[:, :4], position_ids=positions, use_cache=False,
                                       logits_to_keep=0).logits
        rope_positions = torch.tensor([[0, 1, 4, 9, 10]])
        cos, sin = model.model.rotary_emb(rope_positions)
        out["rope_cos"], out["rope_sin"] = cos, sin

        chunk = nar.Chunk([2, 3, 4, 5], torch.randn((5, 64), generator=torch.Generator().manual_seed(42)))
        engine = nar.CachedNAR(model, chunk)
        out["nar_ar_tokens"], out["nar_noise"] = torch.tensor(chunk.ar_tokens), chunk.noise
        for raw in (20.0, 0.0, -2.3):
            key = f"raw{raw}"
            out[f"velocity_{key}"] = engine.velocity(chunk.noise, raw)
            out[f"time_emb_{key}"] = model.time_embedder(
                model._shift_t_value(raw, "cpu", torch.float32).expand(1))
        tapped_velocity(engine, chunk.noise, 20.0, out, "nar")
        engine.close()

        chunk4 = nar.Chunk([2, 3, 4, 5], torch.randn((3, 64), generator=torch.Generator().manual_seed(381)))
        out["solve4_noise"] = chunk4.noise
        out["solve4_expected"] = nar.CachedNAR(model, chunk4).solve(steps=4)

        tokens, truncated, cfg_tokens = greedy_and_cfg(model)
        out["greedy_prefix"] = torch.tensor([2, 3, 4])
        out["greedy_tokens"] = torch.tensor(tokens, dtype=torch.long)
        out["greedy_truncated"] = torch.tensor([truncated])
        out["cfg_tokens"] = torch.tensor(cfg_tokens, dtype=torch.long)
    return out


def save(path="parity/tiny_lm.safetensors"):
    save_file_deterministic(build(), path, metadata={**META, "kind": "tiny_lm", "seed": "173"})


if __name__ == "__main__":
    save()
