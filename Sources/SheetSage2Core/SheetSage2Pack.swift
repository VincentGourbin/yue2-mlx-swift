// SheetSage2Pack.swift - prequantized packs (`weights_format: mlx-quantized`), plan/14 + ASK Q10
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// A prequantized SheetSage2 pack: the parameters of a model already configured for a quantized
/// profile, saved in MLX layout (packed `uint32` weights + `scales` + `biases` for the Conformer's
/// linear layers, decoder float16, ConvNeXt front and mel buffers float32). Loading one rebuilds
/// the quantized structure first and applies the tensors as stored: no float16 model is ever
/// materialized, so neither the load time nor the load peak includes a quantization.
public enum SheetSage2Pack {
    public static let format = "sheetsage2-mlx-quantized-v1"
    public static let weightsFormat = "mlx-quantized"
    public static let groupSize = 64

    /// The quantization every profile applies (`SheetSage2Profile.quantizationBits`).
    static func quantizeEncoder(_ model: SheetSage2Model, bits: Int, groupSize: Int) {
        quantize(model: model, groupSize: groupSize, bits: bits) { path, module in
            module is Linear && path.hasPrefix("encoder.layers.")
        }
    }

    /// Writes `model` (loaded and quantized for a `bits`-bit profile) as a pack into `directory`:
    /// `config.json` and `model.safetensors`. Run it on the CPU device if the GPU is busy.
    public static func save(_ model: SheetSage2Model, bits: Int, groupSize: Int = groupSize, to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var config = model.config
        config.weightsFormat = weightsFormat
        config.quantization = .init(bits: bits, groupSize: groupSize)
        config.loraAlpha = nil
        config.loraRank = nil
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(config).write(to: directory.appendingPathComponent("config.json"))
        let arrays = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(
            arrays: arrays, metadata: ["format": format, "bits": String(bits), "group_size": String(groupSize)],
            url: directory.appendingPathComponent("model.safetensors"))
    }

    /// Loads a pack written by `save`: the quantized structure is built first, then the stored
    /// tensors are applied with a strict key check, in their stored dtypes.
    static func load(directory: URL, config: SheetSage2Config) throws -> SheetSage2Model {
        guard let quantization = config.quantization else {
            throw SheetSage2Error.weightMismatch("mlx-quantized pack without a quantization entry")
        }
        let model = SheetSage2Model(config: config)
        quantizeEncoder(model, bits: quantization.bits, groupSize: quantization.groupSize)
        var weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        let expected = Set(model.parameters().flattened().map(\.0))
        let provided = Set(weights.keys)
        guard expected == provided else {
            let missing = expected.subtracting(provided).sorted().prefix(6)
            let unexpected = provided.subtracting(expected).sorted().prefix(6)
            throw SheetSage2Error.weightMismatch("pack: missing=\(Array(missing)) unexpected=\(Array(unexpected))")
        }
        var loaded = [String: MLXArray]()
        for key in provided.sorted() {
            let value = weights.removeValue(forKey: key)!
            eval(value)
            loaded[key] = value
        }
        model.update(parameters: ModuleParameters.unflattened(loaded))
        Memory.clearCache()
        return model
    }
}
