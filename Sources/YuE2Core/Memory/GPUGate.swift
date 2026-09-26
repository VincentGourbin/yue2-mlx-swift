// GPUGate.swift - pause GPU submissions while the host app is not in the foreground (iOS)
// Copyright 2026 Vincent Gourbin
//
// iOS refuses every GPU submission from a backgrounded app, immediately and without a grace
// period (`kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`); mlx-core raises
// the resulting error inside the Metal completion handler, where nothing can catch it
// (`std::terminate`). MLX commits a command buffer every 20-50 ops *inside* an `eval`, so the
// only place the engine can stop submitting is between evals. This gate is that place: the app
// closes it as soon as the scene goes `.inactive`, reopens it on `.active`; the engine waits on
// it before every unit of GPU work (a token, a NAR layer, a VAE tile).

import Foundation

/// Process-wide gate the engine checks before each unit of GPU work. Open by default.
public final class YuE2GPUGate: @unchecked Sendable {
    public static let shared = YuE2GPUGate()

    private let condition = NSCondition()
    private var suspended = false

    public init() {}

    /// Closes the gate: every `wait` blocks until `resume()`.
    public func suspend() {
        condition.lock()
        suspended = true
        condition.unlock()
    }

    /// Opens the gate and wakes every waiter.
    public func resume() {
        condition.lock()
        suspended = false
        condition.broadcast()
        condition.unlock()
    }

    public var isSuspended: Bool {
        condition.lock()
        defer { condition.unlock() }
        return suspended
    }

    /// Blocks the calling thread while the gate is closed. `cancel`, when given, is polled every
    /// 100 ms; returns `false` when it fired (the caller then throws `.cancelled`), `true` once
    /// the gate is open. A caller that cancels a run from the app must also `resume()` the gate,
    /// or pass the same cancel closure here, so no worker stays parked forever.
    @discardableResult
    public func wait(cancel: (() -> Bool)? = nil) -> Bool {
        condition.lock()
        defer { condition.unlock() }
        while suspended {
            if cancel?() == true { return false }
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.1))
        }
        return true
    }
}

/// Execution knobs that trade a little synchronization for a shorter distance between two gate
/// checks.
public enum YuE2ExecutionPolicy {
    /// How the NAR branch's quantized projections are computed during the solve.
    public enum NARCompute: String, CaseIterable, Sendable {
        /// `quantizedMatmul` on the packed weights (memory-lean; the iPhone default).
        case packed
        /// Dequantized once to bf16/fp16 before the solve (`YuE2ForCausalLM.dequantizeWeights(of:
        /// .nar)`): +1.4 GB for the 8-bit NAR, ≈ 15 % faster on a Mac (large-M matmuls).
        case dequantized
    }

    /// Single-token AR steps through `CompiledDecodeStep` (per-layer fused graphs) instead of
    /// the op-by-op forward. Default off; `YUE2_COMPILED_DECODE=1` or `--compiled-decode`.
    nonisolated(unsafe) public static var compiledDecode: Bool =
        ProcessInfo.processInfo.environment["YUE2_COMPILED_DECODE"] == "1"

    /// Default `.packed`; `YUE2_NAR_COMPUTE=dequantized` overrides (measurements, Mac).
    nonisolated(unsafe) public static var narCompute: NARCompute = {
        ProcessInfo.processInfo.environment["YUE2_NAR_COMPUTE"].flatMap(NARCompute.init(rawValue:)) ?? .packed
    }()

    /// Evaluate the NAR velocity graph after every layer instead of once per ODE step. Turns a
    /// 10-20 s in-flight graph (iPhone, 1 500 frames) into 28 units of ≈ 0.2-0.4 s, so
    /// `YuE2GPUGate` can park the worker before iOS withdraws the GPU. Never changes the numbers
    /// (same ops, same order). Default: on for the mobile memory profile, off on Mac;
    /// `YUE2_EVAL_PER_LAYER=1|0` overrides.
    nonisolated(unsafe) public static var evalPerLayer: Bool = {
        switch ProcessInfo.processInfo.environment["YUE2_EVAL_PER_LAYER"] {
        case "1": return true
        case "0": return false
        default: return YuE2MemoryManager.profile == .mobile
        }
    }()
}
