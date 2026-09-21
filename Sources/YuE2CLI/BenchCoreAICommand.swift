// BenchCoreAICommand.swift - MLX vs Core AI GPU/ANE NAR velocity timing (T-6.4, E4)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import MLXRandom
import YuE2Core

/// `yue2 bench-coreai-nar`: per-shape, per-backend timing of one `velocity` evaluation, plus the
/// once-per-chunk K/V conversion cost, extrapolated to a full 32-step midpoint solve (64 velocity
/// evaluations) — CLAUDE.md's performance-measurement discipline (cool down, A/B/B/A, judge
/// against spread) is for *durable* findings recorded in `docs/knowledge/log.md`; this command's
/// own wall-clock (`Date`) timing is the same convention `parity vae --backend`'s `ms_per_tile`
/// already uses for a quick in-session number, not a substitute for that profiler-based rigor.
struct BenchCoreAICommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-coreai-nar",
        abstract: "Time MLX vs Core AI GPU/ANE NAR velocity evaluations (T-6.4, E4)."
    )

    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Comma-separated chunk frame counts (L) to benchmark.", transform: Self.parseInts)
    var frames: [Int] = [256, 512, 1024, 1536]

    @Option(name: .long, help: "Comma-separated backends to benchmark.", transform: Self.parseBackends)
    var backends: [NARBackendKind] = [.mlx, .coreaiGPU, .coreaiANE]

    @Option(name: .long, help: "Synthetic AR prefix length (L_ar) — any valid token id works for timing.")
    var lAr: Int = 512

    @Option(name: .long, help: "velocity() evaluations per (shape, backend) point.")
    var iterations: Int = 20

    @Option(name: .long, help: "MLX quantization preset for the LM (the MLX NAR path's weights): none (bf16, default), int4-mixed, …")
    var quant: ModelSession.Quantization = .none

    /// T-6.4b diagnostic only: points `.coreaiGPU`/`.coreaiANE` at an arbitrary `.aimodel`
    /// instead of `loadNARBackend`'s normal derived path — e.g.
    /// `$YUE2_MODELS_DIR/YuE2-3B/coreai/YuE2NarStackFp16.aimodel` (`export_nar_stack.py
    /// --variant fp16`), to compare against the production int8 asset's throughput without a
    /// second `NARBackendKind` case. Never used by the default path.
    @Option(name: .long, help: "T-6.4b: override the Core AI NAR asset path (diagnostic only, e.g. to compare the fp16 variant).")
    var narAssetOverride: String?

    static func parseInts(_ value: String) throws -> [Int] {
        try value.split(separator: ",").map {
            guard let v = Int($0) else { throw ValidationError("not an integer: \($0)") }
            return v
        }
    }

    static func parseBackends(_ value: String) throws -> [NARBackendKind] {
        try value.split(separator: ",").map {
            guard let v = NARBackendKind(rawValue: String($0)) else {
                throw ValidationError("unknown backend: \($0)")
            }
            return v
        }
    }

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let config = try YuE2Config.load(from: dir.appendingPathComponent("YuE2-3B/config.json"))
        let model = try LMWeightLoader.load(
            directory: dir.appendingPathComponent("YuE2-3B"), config: config, quantization: quant)
        // Memory axis (question raised after T-6.4b): what does each backend add to the process
        // footprint (`phys_footprint`, iOS jetsam's number) on top of the MLX LM already loaded?
        // `footprint_load_mb` is the resting footprint after every model/asset is in memory,
        // `footprint_peak_mb` the peak during the timing loop; `footprint_base_mb` is the common
        // starting point (LM weights + AR prefill) both are measured against.
        let baseFootprintMB = Int(FootprintSampler.current() / (1024 * 1024))
        print("BENCH base: quant=\(quant.rawValue) footprint_base_mb=\(baseFootprintMB) mlx_active_mb=\(YuE2MemoryManager.snapshot().activeMB)")

        let arTokens = (0..<lAr).map { _ in Int.random(in: 0..<config.vocabSize) }
        let prefill = TokenGenerator.prefill(model: model, tokens: arTokens)

        // mlx=<s> gpu=<s> ane=<s> in `E4 VERDICT` below, aggregated over every requested shape's
        // extrapolated 64-velocity full-solve estimate (mean across `frames`).
        var totals: [NARBackendKind: [Double]] = [:]

        for L in frames {
            let key = MLXRandom.key(UInt64(700_000 + L))
            let noise = MLXRandom.normal([L, config.latentDim], key: key)
            let chunk = NARChunk(arTokens: arTokens, noise: noise)
            let nar = CachedNAR(model: model, chunk: chunk, cache: prefill.cache.clone())

            for kind in backends {
                do {
                    let footprint = FootprintSampler()
                    footprint.start()
                    let (msPerEval, msConversion, loadFootprintMB) = try await Self.time(
                        kind: kind, nar: nar, noise: noise, modelsDir: dir,
                        cache: prefill.cache.clone(), arLength: lAr, narLength: L + 2,
                        iterations: iterations,
                        assetOverride: narAssetOverride.map { URL(fileURLWithPath: $0) })
                    let peakFootprintMB = Int(footprint.stop() / (1024 * 1024))
                    let extrapolated64 = msPerEval * 64
                    totals[kind, default: []].append(extrapolated64)
                    print(
                        "BENCH nar L=\(L) backend=\(kind.rawValue): ms_per_eval=\(msPerEval) "
                            + "ms_conversion=\(msConversion) extrapolated_64=\(extrapolated64)"
                            + " footprint_load_mb=\(loadFootprintMB) footprint_peak_mb=\(peakFootprintMB)"
                            + " mlx_peak_mb=\(YuE2MemoryManager.snapshot().peakMB)")
                } catch {
                    print("BENCH nar L=\(L) backend=\(kind.rawValue): FAILED \(error)")
                }
            }
        }

        func summary(_ kind: NARBackendKind) -> String {
            guard let values = totals[kind], !values.isEmpty else { return "n/a" }
            let meanSeconds = (values.reduce(0, +) / Double(values.count)) / 1000
            return String(format: "%.3f", meanSeconds)
        }
        print(
            "E4 VERDICT: mlx=\(summary(.mlx)) gpu=\(summary(.coreaiGPU)) ane=\(summary(.coreaiANE))")
    }

    /// Returns `(msPerEval, msConversion)`. `msConversion` is the once-per-chunk cost
    /// (`CoreAINARStack.init`'s K/V stacking/padding/NDArray conversion, or the MLX path's cache
    /// clone, which is comparatively free — reported as `0` there since `CachedNAR` never does a
    /// discrete "conversion" step) measured once, outside the `iterations` timing loop.
    private static func time(
        kind: NARBackendKind, nar: CachedNAR, noise: MLXArray, modelsDir: URL, cache: KVCache,
        arLength: Int, narLength: Int, iterations: Int, assetOverride: URL? = nil
    ) async throws -> (Double, Double, Int) {
        switch kind {
        case .mlx:
            let loadFootprintMB = Int(FootprintSampler.current() / (1024 * 1024))
            let start = Date()
            for _ in 0..<iterations {
                let v = nar.velocity(state: noise, rawT: 0.0)
                eval(v)
            }
            let elapsedMS = Date().timeIntervalSince(start) * 1000
            return (elapsedMS / Double(iterations), 0, loadFootprintMB)

        case .coreaiGPU, .coreaiANE:
            let convertStart = Date()
            guard let backend = try await loadNARBackend(
                kind, modelsDir: modelsDir, cache: cache, arLength: arLength, narLength: narLength,
                assetOverride: assetOverride)
            else {
                throw YuE2Error.backendUnavailable("loadNARBackend returned nil for \(kind)")
            }
            let msConversion = Date().timeIntervalSince(convertStart) * 1000
            let loadFootprintMB = Int(FootprintSampler.current() / (1024 * 1024))

            let start = Date()
            for _ in 0..<iterations {
                let v = try await backend.velocity(state: noise, rawT: 0.0)
                eval(v)
            }
            let elapsedMS = Date().timeIntervalSince(start) * 1000
            return (elapsedMS / Double(iterations), msConversion, loadFootprintMB)
        }
    }
}
