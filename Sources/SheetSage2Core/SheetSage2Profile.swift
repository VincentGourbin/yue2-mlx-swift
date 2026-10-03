// SheetSage2Profile.swift - reference profiles `<bits>bit-fast|lean`, as for YuE2 (docs/References.md)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// A reference transcription configuration: every setting that matters is pinned, so a profile
/// gives the same score for the same audio and comparable time on comparable hardware.
public struct SheetSage2Profile: Sendable, Identifiable, Equatable {
    public enum Kind: String, CaseIterable, Sendable { case fast, lean }

    /// `16bit-fast`, `8bit-lean`, …
    public let id: String
    /// Weight bits of the Conformer's linear layers (16 = float16, unquantized). The decoder is never
    /// quantized: its linear layers weigh ≈ 50 MB, and the greedy loop compounds its error.
    public let bits: Int
    public let kind: Kind

    init(bits: Int, kind: Kind) {
        self.bits = bits
        self.kind = kind
        id = "\(bits)bit-\(kind.rawValue)"
    }

    /// Affine quantization of the Conformer's linear layers (group size 64), `nil` for 16 bits. The ConvNeXt
    /// front stays float32 and the embeddings float16 in every profile.
    public var quantizationBits: Int? { bits == 16 ? nil : bits }
    /// Lean: STFT and ConvNeXt computed by chunks of this many frames (the ConvNeXt's global
    /// normalization in two passes), so no stage materializes a whole-window activation.
    public var chunkFrames: Int? { kind == .lean ? 2048 : nil }
    /// Lean: every window is encoded first, then the encoder's weights are released before
    /// decoding (the decoder needs 88 MB of weights and 8 MB of encoder memory per window).
    public var releaseEncoderAfterEncoding: Bool { kind == .lean }
    /// MLX buffer-cache limit while transcribing.
    public var cacheLimitBytes: Int { (kind == .lean ? 128 : 512) * 1024 * 1024 }

    public static let all: [SheetSage2Profile] = [16, 8, 4].flatMap { bits in
        Kind.allCases.map { SheetSage2Profile(bits: bits, kind: $0) }
    }

    public static func named(_ id: String) -> SheetSage2Profile? { all.first { $0.id == id } }

    public static let `default` = named("16bit-fast")!
}

extension SheetSage2Model {
    /// Loads `directory` (merged snapshot or upstream adapter release) configured for `profile`.
    public static func load(directory: URL, parentDirectory: URL? = nil, profile: SheetSage2Profile) throws -> SheetSage2Model {
        let model = try load(directory: directory, parentDirectory: parentDirectory, dtype: .float16, subsamplingFloat32: true)
        if let packed = model.config.quantization {
            // A prequantized pack (ASK Q10): it must be the profile's own quantization.
            guard packed.bits == profile.bits else {
                throw SheetSage2Error.weightMismatch("\(packed.bits)-bit pack loaded for profile \(profile.id)")
            }
        } else if let bits = profile.quantizationBits {
            // Quantized on the fly from float16: works from any snapshot, but the load then pays
            // the quantization and keeps its float16 transient (prefer the matching pack).
            SheetSage2Pack.quantizeEncoder(model, bits: bits, groupSize: SheetSage2Pack.groupSize)
            for (_, value) in model.parameters().flattened() { eval(value) }
            Memory.clearCache()
        }
        model.chunkFrames = profile.chunkFrames
        return model
    }
}

extension SheetSage2Transcriber {
    /// A transcriber applying `profile`'s runtime settings (cache limit, encoder release).
    public init(model: SheetSage2Model, profile: SheetSage2Profile) {
        self.init(model: model)
        cacheLimitBytes = profile.cacheLimitBytes
        releaseEncoderAfterEncoding = profile.releaseEncoderAfterEncoding
    }
}
