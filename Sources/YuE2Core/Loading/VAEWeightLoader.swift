// VAEWeightLoader.swift - VAE decoder: weight-norm merge + torch-to-MLX conv layouts (plan §2.5)
// Copyright 2026 Vincent Gourbin
//
// Every conv in the Oobleck decoder is weight-normed (WNConv1d / WNConvTranspose1d): the
// checkpoint never has a plain `.weight` under `decoder.*`, only `weight_g`/`weight_v` pairs
// (plus `bias`, and `alpha`/`beta` for SnakeBeta). Merge happens BEFORE the layout conversion
// (piège n°9): norm is taken over every dimension but 0, torch `weight_norm(dim=0)` semantics,
// regardless of whether dim 0 means "output channel" (Conv1d) or "input channel"
// (ConvTranspose1d, since torch stores its weight as `[in, out, k]`).

import Foundation
import MLX
import MLXNN

public enum VAEWeightLoader {
    /// `w = g * v / ||v||`, norm over every axis but 0 (torch `weight_norm(dim=0)`).
    public static func mergeWeightNorm(g: MLXArray, v: MLXArray) -> MLXArray {
        let v32 = v.asType(.float32)
        let g32 = g.asType(.float32)
        let reduceAxes = Array(1..<v32.ndim)
        let norm = MLX.sqrt(MLX.sum(v32 * v32, axes: reduceAxes, keepDims: true))
        return g32 * v32 / norm
    }

    /// Torch Conv1d `[out, in, k]` -> MLX `[out, k, in]`.
    public static func conv1dToMLX(_ w: MLXArray) -> MLXArray {
        w.transposed(0, 2, 1)
    }

    /// Torch ConvTranspose1d `[in, out, k]` -> MLX `[out, k, in]`.
    public static func convTransposed1dToMLX(_ w: MLXArray) -> MLXArray {
        w.transposed(1, 2, 0)
    }

    /// The transpose lives at exactly `decoder.layers.{1..6}.layers.1` (`DecoderBlock.layers`
    /// index 1, plan §2.5's topology table); every other weight-normed conv is a plain Conv1d.
    private static let transposedConvBase = try! NSRegularExpression(
        pattern: #"^decoder\.layers\.[1-6]\.layers\.1$"#
    )

    private static func isConvTransposedBase(_ base: String) -> Bool {
        let range = NSRange(base.startIndex..., in: base)
        return transposedConvBase.firstMatch(in: base, range: range) != nil
    }

    /// Loads `decoder.*` from `directory/model.safetensors`: merges every `weight_g`/`weight_v`
    /// pair into a single `.weight`, converts conv layouts to MLX, casts to float32 (VAE stays
    /// fp32 throughout — piège n°11), and evaluates tensor by tensor (piège n°8).
    public static func loadDecoderWeights(directory: URL) throws -> [String: MLXArray] {
        let raw = try WeightLoader.load(url: directory.appendingPathComponent("model.safetensors")) { key in
            key.hasPrefix("decoder.") ? key : nil
        }

        var result = [String: MLXArray]()
        result.reserveCapacity(raw.count)

        for (key, value) in raw where key.hasSuffix(".weight_v") {
            let base = String(key.dropLast(".weight_v".count))
            guard let gain = raw[base + ".weight_g"] else {
                throw YuE2Error.weightMismatch("vae decoder: missing weight_g for \(key)")
            }
            let merged = mergeWeightNorm(g: gain, v: value)
            let converted = isConvTransposedBase(base)
                ? convTransposed1dToMLX(merged)
                : conv1dToMLX(merged)
            let tensor = converted.asType(.float32)
            eval(tensor)
            result[base + ".weight"] = tensor
        }

        for (key, value) in raw where !key.hasSuffix(".weight_v") && !key.hasSuffix(".weight_g") {
            let tensor = value.asType(.float32)
            eval(tensor)
            result[key] = tensor
        }

        return result
    }

    /// Loads `encoder.*` from `directory/model.safetensors` (T-5.1): same weight-norm merge and
    /// float32 cast as the decoder, but every conv is a plain Conv1d (`EncoderBlock` never uses
    /// `ConvTranspose1d`), so no transposed-layout branch is needed here.
    public static func loadEncoderWeights(directory: URL) throws -> [String: MLXArray] {
        let raw = try WeightLoader.load(url: directory.appendingPathComponent("model.safetensors")) { key in
            key.hasPrefix("encoder.") ? key : nil
        }

        var result = [String: MLXArray]()
        result.reserveCapacity(raw.count)

        for (key, value) in raw where key.hasSuffix(".weight_v") {
            let base = String(key.dropLast(".weight_v".count))
            guard let gain = raw[base + ".weight_g"] else {
                throw YuE2Error.weightMismatch("vae encoder: missing weight_g for \(key)")
            }
            let merged = mergeWeightNorm(g: gain, v: value)
            let tensor = conv1dToMLX(merged).asType(.float32)
            eval(tensor)
            result[base + ".weight"] = tensor
        }

        for (key, value) in raw where !key.hasSuffix(".weight_v") && !key.hasSuffix(".weight_g") {
            let tensor = value.asType(.float32)
            eval(tensor)
            result[key] = tensor
        }

        return result
    }
}
