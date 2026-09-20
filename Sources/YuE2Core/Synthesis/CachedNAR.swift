// CachedNAR.swift - one acoustic chunk's AR prefix cache + NAR velocity field (plan §2.4)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXFast

/// One original acoustic chunk: the AR prefix's K/V is prefilled once (pitfall #2 — cached
/// after q/k-norm and RoPE) and reused, unchanged, for every ODE step's NAR forward pass.
public final class CachedNAR {
    private let model: YuE2ForCausalLM
    private let cache: KVCache
    private let dtype: DType
    private let posEmb: MLXArray
    private let noise: MLXArray
    let arLength: Int
    let narLength: Int

    /// If `cache` is omitted, prefills `chunk.arTokens` through the AR path (chunked via
    /// `TokenGenerator.prefill`, so a long prefix never materializes one giant attention/lm_head)
    /// and keeps the resulting cache. Passing a cache lets O2 (T-4.2) reuse a KV cache the AR
    /// generation phase already built for this exact prefix, instead of prefilling twice.
    public init(model: YuE2ForCausalLM, chunk: NARChunk, cache: KVCache? = nil) {
        self.model = model
        self.dtype = model.vae2llm.weight.dtype
        self.arLength = chunk.arTokens.count
        self.narLength = chunk.noise.dim(0) + 2
        self.noise = chunk.noise
        self.cache = cache ?? TokenGenerator.prefill(model: model, tokens: chunk.arTokens).cache
        self.posEmb = model.latentPositions(count: narLength).expandedDimensions(axis: 0)
    }

    /// `v_theta(state, raw_t)`: one NAR forward pass over `narLength` positions (the chunk's
    /// frames padded with one zero row on each side), attending over the AR prefix's cached K/V
    /// plus this step's freshly computed NAR K/V — bidirectionally, with **no mask** (pitfall
    /// #12: the NAR side sees the whole prefix and all of its own positions, never causal).
    /// `taps`, when given, is called with `("after_vae2llm"|"after_time"|"after_pos"|
    /// "after_layer_<i>", hidden)` for bisection against the tiny fixture.
    public func velocity(state: MLXArray, rawT: Double, taps: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let zeroRow = MLXArray.zeros([1, 64]).asType(dtype)
        let xNar = concatenated([zeroRow, state.asType(dtype), zeroRow], axis: 0)
        var x = model.vae2llm(xNar.expandedDimensions(axis: 0))
        taps?("after_vae2llm", x)

        let shifted = MLX.broadcast(model.shiftT(raw: rawT), to: [narLength])
        x = x + model.timeEmbedding(tShifted: shifted).expandedDimensions(axis: 0)
        taps?("after_time", x)

        x = x + posEmb
        taps?("after_pos", x)

        for (i, layer) in model.model.layers.enumerated() {
            guard let (arK, arV) = cache.keyValue(layer: i) else {
                preconditionFailure("CachedNAR: AR cache is missing layer \(i)")
            }
            let (q, k, v) = layer.narSelfAttn.projectQKV(layer.narInputLayernorm(x), offset: arLength)
            let K = concatenated([arK, k], axis: 2)
            let V = concatenated([arV, v], axis: 2)
            x = x + layer.narSelfAttn.attend(q, K, V, mask: .none)
            x = x + layer.narMlp(layer.narPreMlpLayernorm(x))
            taps?("after_layer_\(i)", x)
        }

        let out = model.llm2vae(model.model.norm(x))
        return out[0, 1..<(narLength - 1), 0...]
    }

    /// Midpoint ODE solve from `t = 1` (pure noise) to `t = 0` (data), in `steps` steps, entirely
    /// in the model's compute dtype (bf16 in real weights, fp32 in tiny — pitfall #5: casting up
    /// mid-solve instead of only at the very end is a different, wrong trajectory). `raw = logit(t)`
    /// is computed in `Double`, clamped to `±20` (`logit(1) = +∞ → 20`, pitfall #4).
    /// `onVelocityCall`, when given, fires once per `velocity` call (2 per step) — for tests that
    /// need to count evaluations without instrumenting `velocity` itself.
    ///
    /// `initialState`/`startStep` (default: pure noise from `step 0`, i.e. ordinary generation)
    /// let a caller resume the same schedule from an arbitrary point — e.g. SDEdit-style editing,
    /// where `initialState = t·noise + (1-t)·realData` at `t = 1 - startStep/steps` (the reference
    /// training path is exactly this linear interpolant, `nar.py`'s `solve`), continuing with a
    /// *different* velocity field (this `CachedNAR`'s own cache/conditioning) for the remaining
    /// steps. This is experimental, unvalidated territory — no fixture covers it.
    public func solve(
        steps: Int = 32,
        initialState: MLXArray? = nil,
        startStep: Int = 0,
        onProgress: ((Int, Int) -> Void)? = nil,
        cancel: (() -> Bool)? = nil,
        onVelocityCall: (() -> Void)? = nil
    ) throws -> MLXArray {
        guard steps >= 1 else {
            throw YuE2Error.invalidRequest("steps must be a positive integer")
        }
        guard (0..<steps).contains(startStep) else {
            throw YuE2Error.invalidRequest("startStep must be in 0..<steps")
        }
        var state = (initialState ?? noise).asType(dtype)
        let dt = 1.0 / Double(steps)
        for step in startStep..<steps {
            if cancel?() == true {
                throw YuE2Error.cancelled
            }
            let t = 1.0 - Double(step) * dt
            onVelocityCall?()
            let first = velocity(state: state, rawT: Self.clampedLogit(t))
            let mid = state - first * MLXArray(Float(dt / 2)).asType(dtype)
            if cancel?() == true {
                throw YuE2Error.cancelled
            }
            onVelocityCall?()
            let second = velocity(state: mid, rawT: Self.clampedLogit(t - dt / 2))
            state = state - second * MLXArray(Float(dt)).asType(dtype)
            eval(state)
            onProgress?(step + 1, steps)
        }
        let result = state.asType(.float32)
        eval(result)
        guard MLX.all(MLX.isFinite(result)).item(Bool.self) else {
            throw YuE2Error.nonFiniteLatents
        }
        return result
    }

    /// `clamp(logit(t), -20, 20)` in `Double` (`logit(1) = log(1/0) = +∞ → clamps to 20`).
    private static func clampedLogit(_ t: Double) -> Double {
        let raw = Foundation.log(t / (1 - t))
        return Swift.min(Swift.max(raw, -20), 20)
    }
}
