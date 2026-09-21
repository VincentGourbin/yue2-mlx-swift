"""Exports the YuE2 NAR velocity field (28-layer MoT stack) to Core AI (.aimodel): int8 weights
on the NAR linear projections (`coreai::quantize`/`coreai::dequantize` custom ops, which
`coreai_torch`'s exporter lowers straight to native Core AI weight compression -- see
`_custom_to_core.py`'s `replace_dequantize`), fp16 activations everywhere else.

Usage: .venv-ref/bin/python Scripts/coreai/export_nar_stack.py --models-dir $YUE2_MODELS_DIR

`NARStack.forward` reproduces `Sources/YuE2Core/Synthesis/CachedNAR.swift`'s `velocity` exactly:
START/END zero-row padding, `vae2llm`, time embedder (sigmoid+shift computed *inside* this
module in fp16 -- piège n°4, this fiche's checklist item: compare against the reference's bf16
sigmoid), the audio latent position embedding recomputed analytically (matches the Swift port's
`computeLatentPositions` fallback, avoids bundling the checkpoint's ~100 MB `pe` buffer for a
handful of positions), then per NAR layer: RMSNorm, q/k/v projection, q/k-norm, RoPE with offset
`L_ar` (positions passed in are `[L_ar, L_ar+nar_length)`, matching the reference's cached-KV
convention), concatenation with the AR prefix's K/V, SDPA with **no causal mask** (piège n°12:
the NAR side attends bidirectionally over the whole prefix and all of its own positions), o_proj,
MLP; finally the backbone's last RMSNorm and `llm2vae`.

`torch.export`'s `dynamic_shapes` (`L` in `Dim(256..2048)`, `L_ar` in `Dim(256..8192)`) is tried
first, per the fiche. If it does not trace cleanly, this script falls back to the documented
enumerated-shape buckets (`L_BUCKETS` x `L_AR_BUCKETS`) with a boolean key-padding mask over the
AR-prefix keys (fresh NAR K/V is always fully valid, never masked) -- the same fallback pattern
`export_vae.py` (T-6.3) already used for the VAE decoder/encoder. The bucket set here is
deliberately small (a handful of (L, L_ar) pairs, not the full 512-to-8192 range plan/13-ios-
backend.md §13.7 mentions): this fiche's own closing criterion says the *final* verdict, including
the shape coverage the iPhone pack ships with, waits for G-9's on-device measurement -- this asset
only has to be correct and measurable on the Mac.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import coreai_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch import TorchConverter

# Registers `torch.ops.coreai.{quantize,dequantize}` (`torch.library.custom_op`) -- not imported
# by `coreai_torch/__init__.py` itself, but `_custom_to_core.py` (invoked from `get_decomp_table`/
# `to_coreai`) recognizes exactly these two ops and lowers them straight to native Core AI weight
# compression (`coreai.quantize`/`coreai.dequantize`), so this import is required before tracing.
import coreai_torch._compression.custom_layers  # noqa: F401

L_BUCKETS = (32, 256, 512, 1024, 1536)
L_AR_BUCKETS = (512, 1024)


# ══════════════════════════════════════════════════════════════════════════════
# int8 weight-only quantization for the NAR linear projections
# ══════════════════════════════════════════════════════════════════════════════


def _quantize_int8_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel (axis 0) symmetric int8 quantization -- `coreai::dequantize`'s scale
    has no group-size concept (only per-tensor or per-channel), so per-row is the finest
    granularity available; it is at least as fine as the MLX side's group-64 affine int8 (T-6.2)."""
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(w / scale.unsqueeze(1)), -127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


class Int8Linear(nn.Module):
    """Weight-only int8 `nn.Linear` replacement. `E2` (T-6.2) showed int4 breaks NAR parity (the
    error compounds over 64 `velocity` evaluations per chunk) -- this fiche requires int8, not
    `w4`, on every NAR linear projection. Bias stays fp16 (negligible size)."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        q, scale = _quantize_int8_per_channel(linear.weight)
        self.register_buffer("weight_q", q)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(linear.bias.detach().half()) if linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.ops.coreai.dequantize(self.weight_q, self.scale, axis=0)
        return F.linear(x, w, self.bias)


def _quantize_nar_linears(layer: nn.Module) -> None:
    attn = layer.nar_self_attn
    attn.q_proj = Int8Linear(attn.q_proj)
    attn.k_proj = Int8Linear(attn.k_proj)
    attn.v_proj = Int8Linear(attn.v_proj)
    attn.o_proj = Int8Linear(attn.o_proj)
    mlp = layer.nar_mlp
    mlp.gate_proj = Int8Linear(mlp.gate_proj)
    mlp.up_proj = Int8Linear(mlp.up_proj)
    mlp.down_proj = Int8Linear(mlp.down_proj)


# ══════════════════════════════════════════════════════════════════════════════
# NARStack module
# ══════════════════════════════════════════════════════════════════════════════


def _audio_position_embedding(positions: torch.Tensor, hidden_size: int, dtype: torch.dtype) -> torch.Tensor:
    """`AudioPositionEmbedding`'s analytic formula (`modeling_yue2.py`): `pe[pos,2i]=sin(pos*d_i)`,
    `pe[pos,2i+1]=cos(pos*d_i)`, `d_i=exp(-ln(10000)*2i/hidden_size)`, computed in fp32."""
    div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
    angles = positions.float().unsqueeze(-1) * div_term.unsqueeze(0)
    pe = torch.zeros(positions.shape[0], hidden_size, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe.to(dtype)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _sdpa_no_causal(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, num_kv_heads: int,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """`q`: `[T_q, num_heads, D]`, `k`/`v`: `[T_kv, num_kv_heads, D]` -> `[T_q, num_heads, D]`.
    `key_padding_mask` (`[T_kv]` bool, True = attendable), when given, is broadcast over every
    query row and every head -- a bucket's zero-padded AR-prefix keys must never receive nonzero
    attention weight even though their V is also zero (softmax normalizes over them regardless)."""
    groups = num_heads // num_kv_heads
    q_ = q.transpose(0, 1).unsqueeze(0)
    k_ = k.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
    v_ = v.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
    mask = None if key_padding_mask is None else key_padding_mask.view(1, 1, 1, -1)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask)
    return out[0].transpose(0, 1)


class NARStack(nn.Module):
    """Standalone traceable reproduction of `CachedNAR.velocity` for a whole chunk's 28 NAR
    layers. See the file docstring for the exact computation this mirrors.

    `forward(x_t, raw_t, k_ar, v_ar, key_padding_mask)`:
      - `x_t`: `[L, latent_dim]` fp16 -- the ODE state (unpadded).
      - `raw_t`: `[1]` fp32 -- pre-sigmoid logit-space `t` (`CachedNAR`'s `rawT`, computed in
        Double by the caller's midpoint solver); the sigmoid/timestep-shift formula runs inside
        this module in fp16.
      - `k_ar`/`v_ar`: `[num_layers, L_ar, num_kv_heads, head_dim]` fp16 -- the AR prefix's cached
        K/V (already past q/k-norm and RoPE, exactly `KVCache`'s convention), stacked once per
        chunk (never per ODE step).
      - `key_padding_mask`: `[L_ar + L + 2]` bool, True = real (attendable) key, False = bucket
        padding -- covers the *whole* concatenated key sequence (AR prefix, then this chunk's
        own NAR positions in START/frames/END order), not just the AR prefix: this fiche's own
        risk table (plan/13-ios-backend.md §13.7, L_ar) covers padding the AR prefix, but a
        bucketed `L` pads trailing NAR frames too, and this is a bidirectional (no-causal-mask)
        attention -- an unmasked padded NAR frame would inject a nonzero, non-garbage hidden
        state (`vae2llm(0) + time_embedder(t) + pos_emb[i]` is never actually zero) into every
        real frame's softmax. Pass all-True when both `L_ar` and `L` are exact.
      -> `[L, latent_dim]` fp16, the velocity field (`L` = this call's `x_t` row count, including
        any trailing bucket padding -- the caller crops to the real chunk length).
    """

    def __init__(
        self, *, vae2llm, llm2vae, time_embedder, rotary_emb, final_norm, layers,
        timestep_shift: float, hidden_size: int, num_heads: int, num_kv_heads: int,
        head_dim: int, max_latent_frames: int, quantize: bool = True,
    ) -> None:
        super().__init__()
        self.vae2llm = vae2llm
        self.llm2vae = llm2vae
        self.time_embedder = time_embedder
        self.rotary_emb = rotary_emb
        self.final_norm = final_norm
        self.layers = nn.ModuleList(layers)
        self.timestep_shift = timestep_shift
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_latent_frames = max_latent_frames
        # T-6.4b: `quantize=False` builds a plain fp16 variant (no `Int8Linear`, no
        # `torch.ops.coreai.dequantize` in the traced forward) -- isolates whether Core AI's
        # per-eval overhead comes from the custom dequantize op or is inherent to the runtime/
        # bridge regardless of weight format. There is no separate native int8 weight-compression
        # preset in this SDK build (`coreai-torch` 0.4.2 / `coreai-core` 1.0.0b2) to compare
        # against instead -- verified against `coreai_torch`'s full public API surface,
        # `coreai._compiler._transforms.passes`' pass/pipeline enums, `AIProgram.optimize()`
        # (generic CSE/DCE/canonicalize only), and `xcrun coreai-build compile --help` (no
        # quantize/compress flag); the only weight-compression mechanism this version exposes is
        # exactly the custom-op route already used for `quantize=True`.
        if quantize:
            for layer in self.layers:
                _quantize_nar_linears(layer)

    def _shift_t(self, raw_t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        t_sig = torch.sigmoid(raw_t.to(dtype))
        shift = self.timestep_shift
        if shift == 1.0:
            return t_sig
        shift_t = torch.tensor(shift, dtype=dtype)
        one = torch.tensor(1.0, dtype=dtype)
        return shift_t * t_sig / (one + (shift_t - one) * t_sig)

    def forward(
        self, x_t: torch.Tensor, raw_t: torch.Tensor, k_ar: torch.Tensor, v_ar: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        dtype = x_t.dtype
        L = x_t.shape[0]
        L_ar_bucket = k_ar.shape[1]
        nar_length = L + 2

        zero_row = torch.zeros(1, x_t.shape[1], dtype=dtype)
        x_nar = torch.cat([zero_row, x_t, zero_row], dim=0).unsqueeze(0)  # [1, nar_length, latent]
        x = self.vae2llm(x_nar)  # [1, nar_length, hidden]

        shifted = self._shift_t(raw_t, dtype).expand(nar_length)
        x = x + self.time_embedder(shifted).unsqueeze(0)

        positions_pe = torch.arange(nar_length, dtype=torch.long).clamp(max=self.max_latent_frames - 1)
        x = x + _audio_position_embedding(positions_pe, self.hidden_size, dtype).unsqueeze(0)

        # RoPE anchors NAR positions right after the *real* AR prefix -- `L_ar_bucket` (`k_ar`'s
        # static shape) counts bucket padding too, so the offset must come from the mask's AR
        # slice instead (piège n°1-adjacent: an offset off by the padding amount silently rotates
        # every NAR query/key by the wrong angle -- this is exactly the bug this fiche's own
        # bucket-padding numeric check below caught before it ever reached Swift).
        l_ar_real = key_padding_mask[:L_ar_bucket].sum()
        rope_positions = (l_ar_real + torch.arange(nar_length, dtype=torch.long)).unsqueeze(0)
        cos, sin = self.rotary_emb(rope_positions)  # [1, nar_length, head_dim // 2]
        rc, rs = cos.unsqueeze(2).to(dtype), sin.unsqueeze(2).to(dtype)  # -> [1, nar_length, 1, half]

        for i, layer in enumerate(self.layers):
            attn = layer.nar_self_attn
            ln = layer.nar_input_layernorm(x)
            q = attn.q_proj(ln).view(1, nar_length, self.num_heads, self.head_dim)
            k = attn.k_proj(ln).view(1, nar_length, self.num_kv_heads, self.head_dim)
            v = attn.v_proj(ln).view(1, nar_length, self.num_kv_heads, self.head_dim)
            q, k = attn.q_norm(q), attn.k_norm(k)
            q = _apply_rotary(q, rc, rs)
            k = _apply_rotary(k, rc, rs)

            k_cat = torch.cat([k_ar[i], k[0]], dim=0)
            v_cat = torch.cat([v_ar[i], v[0]], dim=0)
            h = _sdpa_no_causal(q[0], k_cat, v_cat, self.num_heads, self.num_kv_heads, key_padding_mask)
            o = attn.o_proj(h.reshape(1, nar_length, self.num_heads * self.head_dim))
            x = x + o
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))

        out = self.llm2vae(self.final_norm(x))  # [1, nar_length, latent]
        return out[0, 1 : nar_length - 1]


# ══════════════════════════════════════════════════════════════════════════════
# AR prefix K/V (test-only: builds example/reference inputs, never traced/exported)
# ══════════════════════════════════════════════════════════════════════════════


def _prefill_ar_kv(model, ar_tokens: list[int], num_layers: int) -> torch.Tensor:
    """Runs the AR path once (causal, bf16) and returns the post-norm/RoPE K/V every layer
    cached, stacked `[num_layers, L_ar, num_kv_heads, head_dim]` -- mirrors `nar.py`'s
    `CachedNAR._prefill` (this port's own `TokenGenerator.prefill`/`KVCache`)."""
    backbone = model.model
    ids = torch.tensor([ar_tokens], dtype=torch.long)
    positions = torch.arange(len(ar_tokens))[None]
    cos, sin = backbone.rotary_emb(positions)
    x = backbone.embed_tokens(ids)
    keys, values = [], []
    for layer in backbone.layers[:num_layers]:
        q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
        keys.append(k[0])
        values.append(v[0])
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        h = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
            enable_gqa=(q.shape[2] != k.shape[2]),
        )
        x = x + layer.self_attn.o_proj(h.transpose(1, 2).reshape(1, len(ar_tokens), -1))
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return torch.stack(keys), torch.stack(values)


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════


def _build_module(model, num_layers: int, *, quantize: bool = True) -> NARStack:
    """Deep-copies only the NAR-relevant submodules (never the AR backbone's `embed_tokens`/
    `lm_head`/`self_attn`/`mlp`, nor the original `model`), casts them to fp16, and (if
    `quantize`) quantizes the NAR linears -- so the fp32/bf16 reference model used for building
    test fixtures is untouched."""
    layers = copy.deepcopy(model.model.layers[:num_layers])
    aux = copy.deepcopy(
        nn.ModuleDict({
            "vae2llm": model.vae2llm, "llm2vae": model.llm2vae,
            "time_embedder": model.time_embedder, "rotary_emb": model.model.rotary_emb,
            "final_norm": model.model.norm,
        })
    )
    config = model.config
    module = NARStack(
        vae2llm=aux["vae2llm"], llm2vae=aux["llm2vae"], time_embedder=aux["time_embedder"],
        rotary_emb=aux["rotary_emb"], final_norm=aux["final_norm"], layers=layers,
        timestep_shift=float(config.timestep_shift), hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim, max_latent_frames=config.max_latent_frames, quantize=quantize,
    )
    return module.half().eval()


def _example_inputs(latent_dim: int, num_kv_heads: int, head_dim: int, num_layers: int, L: int, L_ar: int):
    x_t = torch.zeros(L, latent_dim, dtype=torch.float16)
    raw_t = torch.zeros(1, dtype=torch.float32)
    k_ar = torch.zeros(num_layers, L_ar, num_kv_heads, head_dim, dtype=torch.float16)
    v_ar = torch.zeros(num_layers, L_ar, num_kv_heads, head_dim, dtype=torch.float16)
    mask = torch.ones(L_ar + L + 2, dtype=torch.bool)
    return (x_t, raw_t, k_ar, v_ar, mask)


def _export_dynamic(module: NARStack, example, out_path: Path) -> bool:
    L_dim = torch.export.Dim("L", min=32, max=2048)
    L_ar_dim = torch.export.Dim("L_ar", min=32, max=8192)
    dynamic_shapes = {
        # `key_padding_mask`'s length (`L_ar + L + 2`) is a sum of two independent dynamic dims,
        # which `torch.export.Dim` cannot express as a derived dim (only affine forms of *one*
        # dim) -- left static here, harmless since this whole dynamic attempt is a best-effort
        # try-first per the fiche, not the path this asset actually ships with.
        "x_t": {0: L_dim}, "raw_t": None, "k_ar": {1: L_ar_dim}, "v_ar": {1: L_ar_dim},
        "key_padding_mask": None,
    }
    try:
        ep = torch.export.export(module, example, dynamic_shapes=dynamic_shapes)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        converter = TorchConverter().add_exported_program(
            ep, input_names=["x_t", "raw_t", "k_ar", "v_ar", "key_padding_mask"],
            output_names=["v"], entrypoint_name="dynamic",
        )
        program = converter.to_coreai()
        program.optimize()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        program.save_asset(out_path)
        print(f"exported nar_stack -> {out_path} (dynamic_shapes: L in [32,2048], L_ar in [32,8192])")
        return True
    except Exception as exc:  # noqa: BLE001 -- fiche-mandated fallback, not a silent catch-all
        print(f"dynamic_shapes export failed, falling back to enumerated buckets: {exc}", file=sys.stderr)
        return False


def _export_buckets(
    module: NARStack, out_path: Path, latent_dim: int, num_kv_heads: int, head_dim: int, num_layers: int,
) -> None:
    converter = TorchConverter()
    entrypoints = []
    for L in L_BUCKETS:
        for L_ar in L_AR_BUCKETS:
            example = _example_inputs(latent_dim, num_kv_heads, head_dim, num_layers, L, L_ar)
            ep = torch.export.export(module, example)
            ep = ep.run_decompositions(coreai_torch.get_decomp_table())
            name = f"L{L}_Lar{L_ar}"
            converter = converter.add_exported_program(
                ep, input_names=["x_t", "raw_t", "k_ar", "v_ar", "key_padding_mask"],
                output_names=["v"], entrypoint_name=name,
            )
            entrypoints.append(name)
    program = converter.to_coreai()
    program.optimize()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(out_path)
    print(f"exported nar_stack -> {out_path} (buckets: {', '.join(entrypoints)})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument(
        "--variant", choices=["int8", "fp16"], default="int8",
        help="int8 (default, production path, weight-only int8 via torch.ops.coreai.dequantize) "
        "or fp16 (T-6.4b diagnostic: no quantization at all, isolates dequantize-op overhead from "
        "Core AI's own per-eval runtime cost). Written to a different .aimodel filename so the "
        "default int8 asset loadNARBackend expects is never disturbed by an fp16 export.",
    )
    args = parser.parse_args()

    from safetensors.torch import load_file
    from yue2.modeling_yue2 import YuE2ForCausalLM

    model = YuE2ForCausalLM.from_pretrained(
        args.models_dir / "YuE2-3B", local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()

    num_layers = args.num_layers
    latent_dim = model.config.latent_dim
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim

    quantize = args.variant == "int8"
    module = _build_module(model, num_layers, quantize=quantize)

    asset_name = "YuE2NarStack.aimodel" if quantize else "YuE2NarStackFp16.aimodel"
    out_path = args.models_dir / "YuE2-3B" / "coreai" / asset_name
    example = _example_inputs(latent_dim, num_kv_heads, head_dim, num_layers, L=16, L_ar=374)
    if not _export_dynamic(module, example, out_path):
        _export_buckets(module, out_path, latent_dim, num_kv_heads, head_dim, num_layers)

    # ── Python-side numeric verification against parity/nar.safetensors ─────
    fixture = load_file(str(args.models_dir / "parity" / "nar.safetensors"))
    ar_tokens = fixture["ar_tokens"].tolist()
    noise = fixture["noise"]
    k_ar_bf16, v_ar_bf16 = _prefill_ar_kv(model, ar_tokens, num_layers)
    k_ar, v_ar = k_ar_bf16.half(), v_ar_bf16.half()
    L_ar = k_ar.shape[1]
    L = noise.shape[0]
    exact_mask = torch.ones(L_ar + L + 2, dtype=torch.bool)

    # Bucketed path: pad x_t/k_ar/v_ar up to this fiche's smallest covering bucket and build the
    # real padding mask (AR-prefix padding + trailing NAR-frame padding, see `forward`'s
    # docstring) -- exercises exactly what the exported `.aimodel` entrypoints run, not just the
    # unpadded eager module the three `velocity_raw*` checks above use.
    L_bucket = next(b for b in L_BUCKETS if b >= L)
    L_ar_bucket = next(b for b in L_AR_BUCKETS if b >= L_ar)
    x_t_padded = F.pad(noise.half(), (0, 0, 0, L_bucket - L))
    k_ar_padded = F.pad(k_ar, (0, 0, 0, 0, 0, L_ar_bucket - L_ar))
    v_ar_padded = F.pad(v_ar, (0, 0, 0, 0, 0, L_ar_bucket - L_ar))
    bucket_mask = torch.cat([
        torch.ones(L_ar, dtype=torch.bool), torch.zeros(L_ar_bucket - L_ar, dtype=torch.bool),
        torch.ones(L + 2, dtype=torch.bool), torch.zeros(L_bucket - L, dtype=torch.bool),
    ])

    worst_rel = 0.0
    for raw in (20.0, 0.0, -2.3):
        raw_t = torch.tensor([raw], dtype=torch.float32)
        reference = fixture[f"velocity_raw{raw}"]
        with torch.inference_mode():
            v = module(noise.half(), raw_t, k_ar, v_ar, exact_mask)
        rel = (v.float() - reference).abs().mean() / reference.abs().mean().clamp_min(1e-8)
        worst_rel = max(worst_rel, rel.item())
        print(f"COREAI NAR velocity_raw{raw}: rel={rel.item():.4f}")

        with torch.inference_mode():
            v_bucket = module(x_t_padded, raw_t, k_ar_padded, v_ar_padded, bucket_mask)[:L]
        rel_bucket = (v_bucket.float() - reference).abs().mean() / reference.abs().mean().clamp_min(1e-8)
        worst_rel = max(worst_rel, rel_bucket.item())
        print(f"COREAI NAR velocity_raw{raw}_bucketL{L_bucket}Lar{L_ar_bucket}: rel={rel_bucket.item():.4f}")

    verdict = f"rel={worst_rel:.4f}"
    if worst_rel >= 5e-2:
        print(f"COREAI NAR FAILED {verdict}")
        return 1
    print(f"COREAI NAR OK {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
