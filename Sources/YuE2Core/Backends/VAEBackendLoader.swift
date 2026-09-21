// VAEBackendLoader.swift - selects/loads a VAEDecoding backend by name (T-6.3, E3)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// The VAE decoder backends this port can run: MLX (always available, any OS this package
/// targets) or Core AI (GPU/Neural Engine, iOS 27+/macOS 27+ only — see `CoreAIVAEDecoder`'s
/// doc comment for the compute-unit-safety limitation of this beta SDK).
public enum VAEBackendKind: String, CaseIterable, Sendable {
    case mlx
    case coreaiGPU = "coreai-gpu"
    case coreaiANE = "coreai-ane"
}

/// Loads `kind` for `directory` (a `YuE2-Vae`-shaped checkpoint directory). Core AI kinds read
/// `directory/coreai/YuE2VaeDecoder.aimodel` (or `.aimodelc` if present, preferred since it
/// skips on-device specialization) — `Scripts/coreai/export_vae.py`'s output path.
///
/// Throws `.backendUnavailable` (never crashes, never silently substitutes MLX) when: the
/// framework isn't in this SDK (`#if canImport(CoreAI)` false), the running OS is below 27, or
/// the asset file is missing. Callers that want an MLX fallback on Core AI failure (the pipeline,
/// per `plan/13-ios-backend.md`) must catch this error themselves and log the fallback — this
/// function never does it implicitly, so a caller that wants strict Core AI (e.g. `yue2 parity
/// vae --backend coreai-gpu`, which must fail loudly on any Core AI problem) gets that too.
public func loadVAEBackend(
    _ kind: VAEBackendKind, directory: URL, precision: VAEPrecision = .fp32
) async throws -> any VAEDecoding {
    switch kind {
    case .mlx:
        return try YuE2VAE.load(directory: directory, precision: precision)

    case .coreaiGPU, .coreaiANE:
        #if canImport(CoreAI)
        guard #available(macOS 27.0, iOS 27.0, *) else {
            throw YuE2Error.backendUnavailable("Core AI requires macOS 27 / iOS 27 or later")
        }
        let config = try VAEConfig.load(from: directory.appendingPathComponent("config.json"))
        let coreaiDir = directory.appendingPathComponent("coreai")
        let compiled = coreaiDir.appendingPathComponent("YuE2VaeDecoder.aimodelc")
        let raw = coreaiDir.appendingPathComponent("YuE2VaeDecoder.aimodel")
        let assetURL = FileManager.default.fileExists(atPath: compiled.path) ? compiled : raw
        guard FileManager.default.fileExists(atPath: assetURL.path) else {
            throw YuE2Error.missingFile(
                "no Core AI VAE decoder asset at \(raw.path) or \(compiled.path) — run Scripts/coreai/export_vae.py first")
        }
        let unit: CoreAIVAEDecoder.ComputeUnit = kind == .coreaiGPU ? .gpu : .neuralEngine
        return try await CoreAIVAEDecoder(assetURL: assetURL, config: config, unit: unit)
        #else
        throw YuE2Error.backendUnavailable("Core AI is not available in this SDK build")
        #endif
    }
}
