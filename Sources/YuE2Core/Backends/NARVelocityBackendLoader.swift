// NARVelocityBackendLoader.swift - selects/loads a NARVelocityBackend by kind (T-6.4, E4)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Loads `kind` for one chunk. Returns `nil` for `.mlx` — that path is `CachedNAR`'s own
/// synchronous `velocity`/`solve`, never a `NARVelocityBackend` value (see that protocol's doc
/// comment for why MLX never goes through this indirection). Core AI kinds read
/// `modelsDir/YuE2-3B/coreai/YuE2NarStack.aimodel` (or `.aimodelc` if present) --
/// `Scripts/coreai/export_nar_stack.py`'s output path, unless `assetOverride` is given (T-6.4b
/// diagnostic only: e.g. `yue2 bench-coreai-nar --nar-asset-override` pointed at
/// `YuE2NarStackFp16.aimodel` to compare the unquantized variant's throughput — never used by the
/// default path, no new `NARBackendKind` case needed). Throws `.backendUnavailable`/
/// `.missingFile` (never crashes, never silently substitutes MLX) on any Core AI problem, same
/// contract as `loadVAEBackend` (T-6.3).
public func loadNARBackend(
    _ kind: NARBackendKind, modelsDir: URL, cache: KVCache, arLength: Int, narLength: Int,
    assetOverride: URL? = nil
) async throws -> (any NARVelocityBackend)? {
    switch kind {
    case .mlx:
        return nil

    case .coreaiGPU, .coreaiANE:
        #if canImport(CoreAI)
        guard #available(macOS 27.0, iOS 27.0, *) else {
            throw YuE2Error.backendUnavailable("Core AI requires macOS 27 / iOS 27 or later")
        }
        let assetURL: URL
        if let assetOverride {
            assetURL = assetOverride
        } else {
            let coreaiDir = modelsDir.appendingPathComponent("YuE2-3B").appendingPathComponent("coreai")
            let compiled = coreaiDir.appendingPathComponent("YuE2NarStack.aimodelc")
            let raw = coreaiDir.appendingPathComponent("YuE2NarStack.aimodel")
            assetURL = FileManager.default.fileExists(atPath: compiled.path) ? compiled : raw
        }
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            throw YuE2Error.missingFile(
                "no Core AI NAR stack asset at \(assetURL.path) — run Scripts/coreai/export_nar_stack.py first")
        }
        let unit: CoreAINARStack.ComputeUnit = kind == .coreaiGPU ? .gpu : .neuralEngine
        return try await CoreAINARStack(
            assetURL: assetURL, unit: unit, cache: cache, arLength: arLength, narLength: narLength)
        #else
        throw YuE2Error.backendUnavailable("Core AI is not available in this SDK build")
        #endif
    }
}
