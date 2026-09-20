// PrequantizedCheckpoint.swift - export/load a quantized AR checkpoint (plan §7 O5, T-4.5)
// Copyright 2026 Vincent Gourbin
//
// On-the-fly quantization still reads the full bf16 `model.safetensors` (≈ 7.3 GB) before
// shrinking the AR projections. `LMWeightLoader` exports the quantized parameters once, to
// `$YUE2_MODELS_DIR/YuE2-3B/mlx-prequantized/<quant>[-head]/model.safetensors`, so every later
// load with the same preset can skip straight to the smaller file. Metadata + a strict two-way
// key check (via `WeightLoader.apply`) guard against a stale or mismatched export.
import Foundation
import MLX

enum YuE2PrequantizedCheckpoint {
    private static let format = "yue2-mlx-swift-prequantized-v1"

    private static func presetName(_ quantization: YuE2Quantization, quantizeHead: Bool) -> String {
        quantizeHead ? "\(quantization.rawValue)-head" : quantization.rawValue
    }

    static func url(lmDirectory: URL, quantization: YuE2Quantization, quantizeHead: Bool) -> URL {
        lmDirectory
            .appendingPathComponent("mlx-prequantized")
            .appendingPathComponent(presetName(quantization, quantizeHead: quantizeHead))
            .appendingPathComponent("model.safetensors")
    }

    static func exists(lmDirectory: URL, quantization: YuE2Quantization, quantizeHead: Bool) -> Bool {
        quantization != .none
            && FileManager.default.fileExists(
                atPath: url(lmDirectory: lmDirectory, quantization: quantization, quantizeHead: quantizeHead).path)
    }

    /// Saves `model`'s current parameters (already quantized by `YuE2QuantizationFilter.apply`)
    /// atomically, tagged with the preset that produced them.
    static func export(
        model: YuE2ForCausalLM, lmDirectory: URL, quantization: YuE2Quantization, quantizeHead: Bool
    ) throws {
        guard let descriptor = quantization.descriptor else { return }
        let destination = url(lmDirectory: lmDirectory, quantization: quantization, quantizeHead: quantizeHead)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

        var arrays = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() { arrays[key] = value }
        // Piège n°8: eval tensor by tensor before writing, never the whole tree at once.
        for (_, value) in arrays { eval(value) }

        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".tmp-\(destination.lastPathComponent)")
        try MLX.save(
            arrays: arrays,
            metadata: [
                "format": format,
                "quantization": quantization.rawValue,
                "quantize_head": quantizeHead ? "1" : "0",
                "bits": "\(descriptor.bits)",
                "group_size": "\(descriptor.groupSize)",
            ],
            url: temporary
        )
        _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary)
        YuE2Debug.log("exported quantized AR checkpoint [\(presetName(quantization, quantizeHead: quantizeHead))] -> \(destination.path)")
    }

    /// Loads a prequantized file into `model`, whose structure must already have been quantized
    /// with the *same* preset (`YuE2QuantizationFilter.apply`) — this only replaces arrays, it
    /// never changes which modules are `QuantizedLinear`.
    static func load(
        into model: YuE2ForCausalLM, from fileURL: URL, quantization: YuE2Quantization, quantizeHead: Bool
    ) throws {
        let (arrays, metadata) = try loadArraysAndMetadata(url: fileURL)
        guard metadata["format"] == format,
              metadata["quantization"] == quantization.rawValue,
              metadata["quantize_head"] == (quantizeHead ? "1" : "0")
        else {
            throw YuE2Error.weightMismatch(
                "\(fileURL.lastPathComponent): metadata mismatch (\(metadata)) for preset "
                    + presetName(quantization, quantizeHead: quantizeHead))
        }
        try WeightLoader.apply(
            arrays, to: model,
            component: "YuE2ForCausalLM [prequantized \(presetName(quantization, quantizeHead: quantizeHead))]")
    }
}
