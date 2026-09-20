// RMSNorm.swift - RMS normalization, variance in fp32 (plan §2.2)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXFast
import MLXNN

/// `x * rsqrt(mean(x.float()^2) + eps).to(x.dtype) * weight` via `MLXFast.rmsNorm`
/// (internal computation in fp32, result cast back to `x`'s dtype).
final class RMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([dimensions]), key: "weight")
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight.asType(x.dtype), eps: eps)
    }
}
