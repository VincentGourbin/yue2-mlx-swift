// ModelPicker.swift - discovers checkpoint variants present on disk (T-4.7, O10)
// Copyright 2026 Vincent Gourbin

import Foundation
import YuE2Core

/// One AR checkpoint variant this machine can actually load right now, discovered by looking at
/// `$modelsDir/YuE2-3B/{model.safetensors,mlx-prequantized/<preset>/model.safetensors}` — never
/// hand-listed, so a variant that isn't really on disk can't appear as a false choice.
struct AvailableVariant: Identifiable, Hashable {
    let id: String
    let quant: YuE2Quantization
    let quantizeHead: Bool
    let label: String
    let sizeBytes: Int64?

    var sizeLabel: String {
        guard let sizeBytes else { return "on-the-fly" }
        return ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }
}

/// The two VAE checkpoints this port ships (`plan/02.5-vae.md`): the standard decoder and the
/// legacy one kept for parity comparisons.
enum VAEChoice: String, CaseIterable, Identifiable {
    case standard = "YuE2-Vae"
    case legacy = "YuE2-Vae-legacy"

    var id: String { rawValue }
    var label: String { self == .standard ? "standard" : "legacy" }
    var directoryName: String { rawValue }
}

enum ModelDiscovery {
    static let modelsDirDefaultsKey = "YuE2BenchUI.modelsDir"

    /// One semantic token is one 40 ms latent frame (plan §2.1) — used to turn a GUI "target
    /// duration in seconds" into a semantic-phase token count, nowhere else in this port.
    static let semanticFramesPerSecond = 25.0

    /// `$YUE2_MODELS_DIR` (or the GUI's own override) resolved once per call — no CLI-only
    /// `resolveModelsDir` here since `YuE2BenchUI` doesn't depend on `YuE2CLI`.
    static func defaultModelsDir() -> String {
        ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"] ?? ""
    }

    /// The field's initial value at launch: whatever the user last picked in this app (survives
    /// across launches, including a double-clicked `.app` that never sees `$YUE2_MODELS_DIR`),
    /// else the environment variable, else empty (the picker still works either way).
    static func rememberedOrDefaultModelsDir() -> String {
        if let remembered = UserDefaults.standard.string(forKey: modelsDirDefaultsKey), !remembered.isEmpty {
            return remembered
        }
        return defaultModelsDir()
    }

    private static func fileSize(_ url: URL) -> Int64? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? nil
    }

    /// AR variants this machine can load without downloading anything: bf16 if
    /// `model.safetensors` exists, any prequantized export already on disk
    /// (`mlx-prequantized/<preset>[-head]/model.safetensors`), and — whenever bf16 is present —
    /// the on-the-fly quantized presets too (`LMWeightLoader` derives them from bf16 at load
    /// time and exports them for next time, T-4.5/O5).
    static func availableVariants(modelsDir: URL) -> [AvailableVariant] {
        var result: [AvailableVariant] = []
        let lmDir = modelsDir.appendingPathComponent("YuE2-3B")
        let bf16 = lmDir.appendingPathComponent("model.safetensors")
        let hasBF16 = FileManager.default.fileExists(atPath: bf16.path)
        if hasBF16 {
            result.append(
                AvailableVariant(id: "none", quant: .none, quantizeHead: false, label: "bf16 (full precision)", sizeBytes: fileSize(bf16)))
        }

        let prequantDir = lmDir.appendingPathComponent("mlx-prequantized")
        for quant in YuE2Quantization.allCases where quant != .none {
            for quantizeHead in [false, true] {
                let name = quantizeHead ? "\(quant.rawValue)-head" : quant.rawValue
                let file = prequantDir.appendingPathComponent(name).appendingPathComponent("model.safetensors")
                guard FileManager.default.fileExists(atPath: file.path) else { continue }
                result.append(
                    AvailableVariant(
                        id: name, quant: quant, quantizeHead: quantizeHead,
                        label: "\(quant.rawValue)\(quantizeHead ? " + head" : "") (prequantized)",
                        sizeBytes: fileSize(file)))
            }
        }

        if hasBF16 {
            for quant: YuE2Quantization in [.qint8, .int4] {
                guard !result.contains(where: { $0.quant == quant && !$0.quantizeHead }) else { continue }
                result.append(
                    AvailableVariant(
                        id: "\(quant.rawValue)-on-the-fly", quant: quant, quantizeHead: false,
                        label: "\(quant.rawValue) (from bf16, on-the-fly)", sizeBytes: nil))
            }
        }
        return result
    }

    /// VAE variants whose `model.safetensors` is actually present under `modelsDir`.
    static func availableVAEs(modelsDir: URL) -> [VAEChoice] {
        VAEChoice.allCases.filter {
            FileManager.default.fileExists(
                atPath: modelsDir.appendingPathComponent($0.directoryName).appendingPathComponent("model.safetensors").path)
        }
    }
}
