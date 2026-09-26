// Attention.swift - GQA attention, q/k-norm before RoPE, RoPE non-interleaved (plan §2.2, pitfall #1)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXFast
import MLXNN

/// Qwen3-style attention: `q_norm`/`k_norm` (RMSNorm per head) applied **before** RoPE,
/// RoPE non-interleaved (`traditional: false`), GQA handled natively by
/// `MLXFast.scaledDotProductAttention` (no manual K/V repeat).
final class YuE2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: MLXNN.RoPE
    private let ropeBase: Float

    init(config: YuE2Config) {
        numHeads = config.numAttentionHeads
        numKVHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = 1.0 / Float(headDim).squareRoot()
        ropeBase = config.ropeTheta

        _qProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, numHeads * headDim, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, numKVHeads * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, numKVHeads * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, config.hiddenSize, bias: false), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps), key: "k_norm")
        rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    /// Projects, normalizes (per head, before RoPE) and rotates Q/K; returns `[B, H, T, D]`.
    func projectQKV(_ x: MLXArray, offset: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (b, t) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, t, numHeads, headDim))
        var k = kNorm(kProj(x).reshaped(b, t, numKVHeads, headDim))
        let v = vProj(x).reshaped(b, t, numKVHeads, headDim).transposed(0, 2, 1, 3)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        return (q, k, v)
    }

    /// Same as `projectQKV(_:offset:)`, but for arbitrary (non-contiguous) absolute positions
    /// per token — `MLXFast.RoPE`'s `offset` only expresses `arange(T) + offset` (optionally one
    /// offset per batch element), which cannot express e.g. `positions = [1, 4, 9, 10]`. Mirrors
    /// the same math (plan §2.2: `x1·cos − x2·sin, x2·cos + x1·sin`) by hand instead.
    func projectQKV(_ x: MLXArray, positions: MLXArray) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (b, t) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, t, numHeads, headDim))
        var k = kNorm(kProj(x).reshaped(b, t, numKVHeads, headDim))
        let v = vProj(x).reshaped(b, t, numKVHeads, headDim).transposed(0, 2, 1, 3)
        q = rotateExplicit(q.transposed(0, 2, 1, 3), positions: positions)
        k = rotateExplicit(k.transposed(0, 2, 1, 3), positions: positions)
        return (q, k, v)
    }

    func rotateExplicit(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let half = headDim / 2
        let i = MLXArray(Array(0..<half)).asType(.float32)
        let invFreq = MLX.exp(i * (-Foundation.log(ropeBase) / Float(half)))
        let theta = positions.asType(.float32).expandedDimensions(axis: -1) * invFreq
        let cos = MLX.cos(theta).asType(x.dtype).reshaped([1, 1, theta.dim(0), half])
        let sin = MLX.sin(theta).asType(x.dtype).reshaped([1, 1, theta.dim(0), half])
        let x1 = x[.ellipsis, 0..<half]
        let x2 = x[.ellipsis, half..<headDim]
        return concatenated([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis: -1)
    }

    /// Scaled dot-product attention over `[B, H, T, D]` inputs, back to `[B, T, hiddenSize]`.
    func attend(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (b, t) = (q.dim(0), q.dim(2))
        let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(b, t, numHeads * headDim))
    }
}
