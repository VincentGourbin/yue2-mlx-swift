// CompiledDecodeStep.swift - single-token AR forward with the per-layer math compiled (dispatch-bound decode)
// Copyright 2026 Vincent Gourbin
//
// The AR decode loop is bound by dispatch, not compute (2026-09-26: one core at 60-96 %, GPU at
// 70 %, ≈ 5-16 ms of CPU per token; 8-bit pack at 114 tok/s where memory bandwidth would allow
// ≈ 270). Every token walks 28 layers × ≈ 20 small ops built from Swift and encoded one by one.
// This path compiles, per layer, the two op sequences that carry no Swift `Int` (norm →
// q/k/v projections → q/k norm → explicit RoPE ; o_proj → residual → norm → MLP → residual) into
// one fused graph each, replayed every token; the cache write (`slice_update` at a Swift
// offset) and the fused SDPA stay outside. ≈ 7 dispatch units per layer instead of ≈ 20.
//
// Closures are built per generation call (they capture the layer's current weight arrays as
// compile-time constants: never keep them across `releaseWeights`/`loadWeights`), for the
// single-token shape only — prefill keeps the plain path.

import MLX
import MLXFast
import MLXNN

/// Compiled single-token step over a `YuE2ForCausalLM`'s AR branch. `forward` is a drop-in for
/// `Backbone.forwardAR(tokens:cache:)` when `tokens` is `[1, 1]`.
final class CompiledDecodeStep {
    private let model: YuE2ForCausalLM
    private let pre: [([MLXArray]) -> [MLXArray]]
    private let post: [([MLXArray]) -> [MLXArray]]

    init(model: YuE2ForCausalLM) {
        self.model = model
        var pre: [([MLXArray]) -> [MLXArray]] = []
        var post: [([MLXArray]) -> [MLXArray]] = []
        for layer in model.model.layers {
            let attention = layer.selfAttn
            let (heads, kvHeads, headDim) = (attention.numHeads, attention.numKVHeads, attention.headDim)
            pre.append(compile { args in
                let x = args[0], positions = args[1]
                let normed = layer.inputLayernorm(x)
                let (b, t) = (normed.dim(0), normed.dim(1))
                let q = attention.qNorm(attention.qProj(normed).reshaped(b, t, heads, headDim)).transposed(0, 2, 1, 3)
                let k = attention.kNorm(attention.kProj(normed).reshaped(b, t, kvHeads, headDim)).transposed(0, 2, 1, 3)
                let v = attention.vProj(normed).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
                return [attention.rotateExplicit(q, positions: positions), attention.rotateExplicit(k, positions: positions), v]
            })
            post.append(compile { args in
                let attended = args[0], residual = args[1]
                let (b, t) = (residual.dim(0), residual.dim(1))
                let h = residual + attention.oProj(attended.transposed(0, 2, 1, 3).reshaped(b, t, heads * headDim))
                return [h + layer.mlp(layer.postAttentionLayernorm(h))]
            })
        }
        self.pre = pre
        self.post = post
    }

    /// One token (`[1, 1]`) through the AR branch, appending to `cache` and advancing it —
    /// same contract and numerics (up to fusion order) as `Backbone.forwardAR`.
    func forward(tokens: MLXArray, cache: KVCache) -> MLXArray {
        precondition(tokens.dim(1) == 1, "CompiledDecodeStep is for single-token steps; prefill uses forwardAR")
        let positions = MLXArray([Int32(cache.offset)])
        var hidden = model.model.embedTokens(tokens)
        for (i, layer) in model.model.layers.enumerated() {
            let qkv = pre[i]([hidden, positions])
            let (k, v) = cache.append(layer: i, k: qkv[1], v: qkv[2])
            let attended = MLXFast.scaledDotProductAttention(
                queries: qkv[0], keys: k, values: v, scale: layer.selfAttn.scale, mask: .none)
            hidden = post[i]([attended, hidden])[0]
        }
        cache.advance(by: 1)
        return model.model.norm(hidden)
    }
}
