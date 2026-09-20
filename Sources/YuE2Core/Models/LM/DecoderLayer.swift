// DecoderLayer.swift - Mixture-of-Transformers decoder layer, AR path (plan §2.2)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXFast
import MLXNN

/// One backbone layer. Declares **both** the AR and NAR sub-modules (Mixture-of-Transformers)
/// for checkpoint coverage, but only exercises the AR path here — the NAR path is driven
/// separately by `CachedNAR` (Jalon 3) and is not called from `forwardAR`.
final class DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: YuE2Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MLP

    @ModuleInfo(key: "nar_input_layernorm") var narInputLayernorm: RMSNorm
    @ModuleInfo(key: "nar_self_attn") var narSelfAttn: YuE2Attention
    @ModuleInfo(key: "nar_pre_mlp_layernorm") var narPreMlpLayernorm: RMSNorm
    @ModuleInfo(key: "nar_mlp") var narMlp: MLP

    init(config: YuE2Config) {
        _inputLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "input_layernorm")
        _selfAttn = ModuleInfo(wrappedValue: YuE2Attention(config: config), key: "self_attn")
        _postAttentionLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "post_attention_layernorm")
        _mlp = ModuleInfo(wrappedValue: MLP(config: config), key: "mlp")
        _narInputLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "nar_input_layernorm")
        _narSelfAttn = ModuleInfo(wrappedValue: YuE2Attention(config: config), key: "nar_self_attn")
        _narPreMlpLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "nar_pre_mlp_layernorm")
        _narMlp = ModuleInfo(wrappedValue: MLP(config: config), key: "nar_mlp")
        super.init()
    }

    /// AR path: `input_layernorm → self_attn (+cache) → +res → post_attention_layernorm → mlp → +res`.
    /// `positions`, when given, overrides the cache-offset-derived contiguous RoPE positions
    /// with arbitrary absolute per-token positions (T-2.4).
    func forwardAR(
        _ x: MLXArray, cache: KVCache?, layer: Int, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        positions: MLXArray? = nil
    ) -> MLXArray {
        let residual = x
        let normed = inputLayernorm(x)
        let (q, k0, v0): (MLXArray, MLXArray, MLXArray)
        if let positions {
            (q, k0, v0) = selfAttn.projectQKV(normed, positions: positions)
        } else {
            (q, k0, v0) = selfAttn.projectQKV(normed, offset: cache?.offset ?? 0)
        }
        var (k, v) = (k0, v0)
        if let cache {
            (k, v) = cache.append(layer: layer, k: k, v: v)
        }
        var h = residual + selfAttn.attend(q, k, v, mask: mask)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}
