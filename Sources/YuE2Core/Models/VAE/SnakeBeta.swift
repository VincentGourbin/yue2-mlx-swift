// SnakeBeta.swift - per-channel Snake activation (plan §2.5; adapted from ltx-video-swift-mlx)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Kill-switch for `compile(shapeless:)` in the VAE decoder (O7/T-4.4), mirroring
/// `Flux2FusedKernels.disabled`'s environment-variable pattern for A/B comparison.
enum YuE2Compile {
    nonisolated(unsafe) static let disabled = ProcessInfo.processInfo.environment["YUE2_DISABLE_COMPILE"] != nil
}

/// `y = x + (1 / (exp(beta) + eps)) * sin(exp(alpha) * x)^2`, alpha/beta stored in log scale,
/// one value per channel. `x` is NLC, so the channel axis is last and the per-channel
/// parameters broadcast without reshaping.
final class SnakeBeta: Module, UnaryLayer {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    private let eps: Float = 1e-9
    /// Traced once (on first call, by which point weight loading has already fixed `alpha`/
    /// `beta` for inference) and reused for every subsequent call at any input shape
    /// (`shapeless: true` — frame count varies per chunk/tile). O7/T-4.4.
    private lazy var compiledActivation: (MLXArray) -> MLXArray = MLX.compile(shapeless: true, activation)

    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.zeros([channels])
        _beta.wrappedValue = MLXArray.zeros([channels])
    }

    private func activation(_ x: MLXArray) -> MLXArray {
        let a = MLX.exp(alpha)
        let b = MLX.exp(beta)
        let s = MLX.sin(x * a)
        return x + (1.0 / (b + eps)) * (s * s)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        YuE2Compile.disabled ? activation(x) : compiledActivation(x)
    }
}
