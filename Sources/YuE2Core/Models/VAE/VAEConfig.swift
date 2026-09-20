// VAEConfig.swift - YuE2VAEConfig decode (plan §2.5, encoder_config added T-5.1)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Encoder topology parameters (`OobleckEncoder.__init__` keyword arguments). `latentDim` here
/// is the encoder's raw output width (128 on the released checkpoint: mean ‖ scale concatenated,
/// each half the top-level `VAEConfig.latentDim`), not the same field as the decoder's.
public struct VAEEncoderConfig: Codable, Sendable {
    public var inChannels: Int
    public var channels: Int
    public var cMults: [Int]
    public var strides: [Int]
    public var latentDim: Int
    public var useSnake: Bool

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case channels
        case cMults = "c_mults"
        case strides
        case latentDim = "latent_dim"
        case useSnake = "use_snake"
    }
}

/// Decoder-only topology parameters (`OobleckDecoder.__init__` keyword arguments).
public struct VAEDecoderConfig: Codable, Sendable {
    public var channels: Int
    public var cMults: [Int]
    public var strides: [Int]
    public var latentDim: Int
    public var outChannels: Int
    public var useSnake: Bool
    public var snakeType: String
    public var finalTanh: Bool

    enum CodingKeys: String, CodingKey {
        case channels
        case cMults = "c_mults"
        case strides
        case latentDim = "latent_dim"
        case outChannels = "out_channels"
        case useSnake = "use_snake"
        case snakeType = "snake_type"
        case finalTanh = "final_tanh"
    }
}

/// `YuE2VAEConfig` (`config.json`). `encoderConfig` is optional so a config missing it (none on
/// a released checkpoint, but decoder-only exports are plausible) still decodes for `YuE2VAE`.
public struct VAEConfig: Codable, Sendable {
    public var encoderConfig: VAEEncoderConfig?
    public var decoderConfig: VAEDecoderConfig
    public var latentDim: Int
    public var downsamplingRatio: Int
    public var sampleRate: Int
    public var decodeCoreFrames: Int
    public var decodeHaloFrames: Int
    public var releaseVariant: String

    enum CodingKeys: String, CodingKey {
        case encoderConfig = "encoder_config"
        case decoderConfig = "decoder_config"
        case latentDim = "latent_dim"
        case downsamplingRatio = "downsampling_ratio"
        case sampleRate = "sample_rate"
        case decodeCoreFrames = "decode_core_frames"
        case decodeHaloFrames = "decode_halo_frames"
        case releaseVariant = "release_variant"
    }

    public static func load(from url: URL) throws -> VAEConfig {
        let config = try JSONDecoder().decode(VAEConfig.self, from: Data(contentsOf: url))
        precondition(
            config.decoderConfig.strides.reduce(1, *) == config.downsamplingRatio,
            "decoder strides do not multiply to downsampling_ratio"
        )
        precondition(
            config.decoderConfig.latentDim == config.latentDim,
            "decoder latent_dim does not match the top-level latent_dim"
        )
        return config
    }
}
