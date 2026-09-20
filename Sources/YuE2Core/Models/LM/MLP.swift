// MLP.swift - SwiGLU feed-forward (plan §2.2)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// `down(silu(gate(x)) * up(x))`, no bias.
final class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: YuE2Config) {
        self._gateProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize, bias: false), key: "gate_proj")
        self._upProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize, bias: false), key: "up_proj")
        self._downProj = ModuleInfo(wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: false), key: "down_proj")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
