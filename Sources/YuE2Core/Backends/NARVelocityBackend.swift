// NARVelocityBackend.swift - backend-agnostic NAR velocity-field protocol (T-6.4, E4)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// A NAR velocity-field backend for a single chunk: MLX (`CachedNAR.velocity` itself, always
/// available) or Core AI (`CoreAINARStack`, GPU/Neural Engine, iOS 27+/macOS 27+ only). Unlike
/// `VAEDecoding`, this never replaces `CachedNAR.velocity`/`solve` (the hot path every existing
/// caller — `Pipeline`, `RemixCommand`, the generation smoke tests — already depends on staying
/// synchronous and MLX-only); it only backs the new `solve(steps:backend:...)` overload that
/// `yue2 parity nar --backend coreai-*` and `yue2 bench-coreai-nar` opt into explicitly.
public protocol NARVelocityBackend: Sendable {
    /// `state`: `[L, latentDim]`, the ODE state for this evaluation. `rawT`: pre-sigmoid
    /// logit-space `t` (`CachedNAR`'s own `rawT`, computed in `Double`). Returns `[L, latentDim]`.
    func velocity(state: MLXArray, rawT: Double) async throws -> MLXArray
}

/// The NAR velocity backends this port can run.
public enum NARBackendKind: String, CaseIterable, Sendable {
    case mlx
    case coreaiGPU = "coreai-gpu"
    case coreaiANE = "coreai-ane"
}
