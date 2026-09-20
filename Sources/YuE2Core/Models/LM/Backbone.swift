// Backbone.swift - embed_tokens + decoder layers + final norm (plan §2.2)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXFast
import MLXNN

/// The AR transformer stack: `embed_tokens → layers → norm`.
final class Backbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: YuE2Config) {
        _embedTokens = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize), key: "embed_tokens")
        layers = (0..<config.numHiddenLayers).map { _ in DecoderLayer(config: config) }
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "norm")
        super.init()
    }

    /// Runs the AR path over `tokens`, appending to `cache` if given, and returns the final
    /// normalized hidden states `[B, T, hiddenSize]`. `taps(layerIndex, hidden)` is called with
    /// the hidden state right after each layer (pre-`norm`), for bisection. `positions`, when
    /// given, overrides cache-offset-derived RoPE positions with arbitrary absolute ones.
    ///
    /// The mask is always `.causal`: `MLXFast.scaledDotProductAttention`'s native causal mode
    /// already aligns bottom-right from the actual key/query lengths (`offset = kL - qL`), which
    /// is exactly the cache's pre-step offset once K/V are appended — verified against
    /// `parity/tiny_lm.safetensors`' chunked-cache fixtures (T-2.4) rather than building a
    /// redundant explicit boolean mask by hand.
    func forwardAR(
        tokens: MLXArray, cache: KVCache?, positions: MLXArray? = nil,
        taps: ((Int, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        var hidden = embedTokens(tokens)
        let t = tokens.dim(1)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = t == 1 ? .none : .causal
        for (i, layer) in layers.enumerated() {
            hidden = layer.forwardAR(hidden, cache: cache, layer: i, mask: mask, positions: positions)
            taps?(i, hidden)
        }
        cache?.advance(by: t)
        return norm(hidden)
    }
}
