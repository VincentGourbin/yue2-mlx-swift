// SheetSage2Weights.swift - merged snapshot → MLX parameter tree (plan/14-sheetsage2.md)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Maps the upstream `state_dict` (merged snapshot from `save_pretrained`) onto `SheetSage2Model`:
/// `nn.Sequential` indices become named children, Conv1d weights go from PyTorch's
/// `[out, in/groups, k]` to MLX's `[out, k, in/groups]`, and the two tied copies of the token
/// embedding (`token_embedding`, `output_projection`) are dropped in favour of
/// `decoder.embed_tokens`.
public enum SheetSage2Weights {
    private static let renames: [(String, String, isConv: Bool)] = [
        (".resampling_layer.0.", ".resample_norm.", false),
        (".resampling_layer.2.", ".resample_conv.", true),
        (".convnext_layers.", ".layers.", false),
        (".depthwise_block.1.", ".depthwise.", true),
        (".pointwise_block.0.", ".norm.", false),
        (".pointwise_block.1.", ".expand.", false),
        (".pointwise_block.3.", ".grn.", false),
        (".pointwise_block.4.", ".project.", false),
        (".conv_block.1.", ".pointwise_in.", false),
        (".conv_block.3.", ".depthwise.", true),
        (".conv_block.4.1.", ".depthwise_norm.", false),
        (".conv_block.6.", ".pointwise_out.", false),
    ]

    /// Renames and converts every tensor; `nil` keys are dropped.
    public static func sanitize(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in raw {
            if key == "token_embedding.weight" || key == "output_projection.weight" { continue }
            var mapped = key
            var isConv = false
            for (from, to, conv) in renames where mapped.contains(from) {
                mapped = mapped.replacingOccurrences(of: from, with: to)
                isConv = isConv || conv
            }
            if mapped.contains(".pointwise_in.") || mapped.contains(".pointwise_out.") {
                result[mapped] = value.squeezed(axis: -1)  // kernel-1 Conv1d [out, in, 1] → Linear [out, in]
            } else {
                result[mapped] = isConv && mapped.hasSuffix(".weight") ? value.transposed(0, 2, 1) : value
            }
        }
        // `save_pretrained` keeps a single copy of tied tensors; which name survives depends on
        // the tie order, so take whichever copy the file has.
        if result["decoder.embed_tokens.weight"] == nil {
            result["decoder.embed_tokens.weight"] = raw["token_embedding.weight"] ?? raw["output_projection.weight"]
        }
        return result
    }

    /// Applies `weights` to `model`, refusing any missing or unexpected key, casting floating
    /// tensors to `dtype` except the mel front-end's buffers (float32 always, as upstream).
    public static func apply(
        _ weights: consuming [String: MLXArray], to model: SheetSage2Model, dtype: DType, subsamplingFloat32: Bool = false
    ) throws {
        let expected = Set(model.parameters().flattened().map(\.0))
        let provided = Set(weights.keys)
        let missing = expected.subtracting(provided).sorted()
        let unexpected = provided.subtracting(expected).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw SheetSage2Error.weightMismatch(
                "missing=[\(missing.prefix(8).joined(separator: ", "))] unexpected=[\(unexpected.prefix(8).joined(separator: ", "))]")
        }
        // Tensor by tensor: each float32 source is dropped as soon as its cast is evaluated, so the
        // load never holds the whole float32 checkpoint next to its float16 copy.
        var cast = [String: MLXArray]()
        for key in provided.sorted() {
            guard let value = weights.removeValue(forKey: key) else { continue }
            let keepFloat32 = key.hasPrefix("encoder.feature_extractor.") || key == "layer_weight"
                || (subsamplingFloat32 && key.hasPrefix("encoder.subsampling_module."))
            let converted = keepFloat32 || !value.dtype.isFloatingPoint ? value.asType(.float32) : value.asType(dtype)
            eval(converted)
            cast[key] = converted
        }
        model.update(parameters: ModuleParameters.unflattened(cast))
    }
}

extension SheetSage2Weights {
    static let projections = ["query_proj", "key_proj", "value_proj", "out_proj"]

    /// Upstream `merge_lora`: the MERT-v2 parent's tensors under `encoder.`, the adapter file's
    /// own tensors, and `W += (B·A)·α/r` on every attention projection, in float32 on the CPU
    /// like upstream. Throws if an adapter tensor is missing or left unused.
    public static func mergeAdapters(
        adapter: [String: MLXArray], parent: [String: MLXArray], config: SheetSage2Config
    ) throws -> [String: MLXArray] {
        guard let alpha = config.loraAlpha, let rank = config.loraRank else {
            throw SheetSage2Error.weightMismatch("adapter config without lora_alpha / lora_rank")
        }
        let scale = Float(alpha / Double(rank))
        var merged = [String: MLXArray]()
        for (key, value) in parent { merged["encoder." + key] = value }
        var adapters = [String: MLXArray]()
        for (key, value) in adapter {
            if key.hasPrefix("adapter.") { adapters[key] = value } else { merged[key] = value }
        }
        for layer in 0..<config.backbone.numHiddenLayers {
            for projection in projections {
                let base = "adapter.layers.\(layer).attn.\(projection)"
                let key = "encoder.layers.\(layer).attn.\(projection).weight"
                guard let a = adapters.removeValue(forKey: base + ".lora_A.weight"),
                    let b = adapters.removeValue(forKey: base + ".lora_B.weight"), let weight = merged[key]
                else { throw SheetSage2Error.weightMismatch("missing LoRA factors or parent weight for \(key)") }
                let update = matmul(b.asType(.float32), a.asType(.float32), stream: .cpu) * scale
                merged[key] = add(weight.asType(.float32), update, stream: .cpu)
            }
        }
        guard adapters.isEmpty else {
            throw SheetSage2Error.weightMismatch("unused adapter tensors: \(adapters.keys.sorted().prefix(4))")
        }
        return merged
    }
}

extension SheetSage2Model {
    /// Loads a merged snapshot directory (`config.json` + `model.safetensors`, written by
    /// `Scripts/reference/sheetsage2_fixtures.py convert`).
    /// Default float16: tokens and ABC identical to the float32 reference on both parity excerpts,
    /// where bfloat16 diverges; on a 500 s song greedy decoding drifts in float16 too, but less than
    /// a change of input resampler does in float32 (docs/knowledge/log.md, 2026-10-03). `subsamplingFloat32` keeps the
    /// small ConvNeXt front (1.2 M parameters, 30 000 frames) in float32: its GRN sums x² over the
    /// whole window and overflows float16, so it is on by default.
    ///
    /// `directory` holds either a merged snapshot (`weights_format: merged`) or the upstream
    /// adapter release (`weights_format: adapter`, as `yue2 download --model sheetsage2` fetches
    /// it), whose MERT-v2 parent is read from `parentDirectory` (default: the sibling
    /// `MERT-v2-FullSong` folder) and merged here.
    public static func load(
        directory: URL, parentDirectory: URL? = nil, dtype: DType = .float16, subsamplingFloat32: Bool = true
    ) throws -> SheetSage2Model {
        let config = try SheetSage2Config.load(directory.appendingPathComponent("config.json"))
        if config.weightsFormat == SheetSage2Pack.weightsFormat {
            return try SheetSage2Pack.load(directory: directory, config: config)
        }
        let model = SheetSage2Model(config: config)
        let weights: [String: MLXArray]
        switch config.weightsFormat {
        case "merged":
            weights = SheetSage2Weights.sanitize(try loadArrays(url: directory.appendingPathComponent("model.safetensors")))
        case "adapter":
            let parent = parentDirectory ?? directory.deletingLastPathComponent().appendingPathComponent("MERT-v2-FullSong")
            weights = SheetSage2Weights.sanitize(try SheetSage2Weights.mergeAdapters(
                adapter: try loadArrays(url: directory.appendingPathComponent("model.safetensors")),
                parent: try loadArrays(url: parent.appendingPathComponent("model.safetensors")), config: config))
        default:
            throw SheetSage2Error.weightMismatch("unknown weights_format \(config.weightsFormat)")
        }
        try SheetSage2Weights.apply(consume weights, to: model, dtype: dtype, subsamplingFloat32: subsamplingFloat32)
        Memory.clearCache()
        return model
    }
}

extension DType {
    var isFloatingPoint: Bool {
        self == .float32 || self == .float16 || self == .bfloat16 || self == .float64
    }
}
