// YuE2VAEEncoder.swift - encoder-only VAE: load + encode (T-5.1)
// Copyright 2026 Vincent Gourbin
//
// Kept separate from YuE2VAE (decoder-only) so nothing about the existing decode path changes:
// generation never needs the encoder, and this class exists purely for T-5.2's `yue2 encode`.

import Foundation
import MLX
import MLXNN

/// Encoder-only port of `YuE2VAE.encode` (`modeling_vae.py:488-513`), deterministic mode only
/// (posterior mean; `sample=True` stochastic reparameterization is not exposed here -- nothing
/// in this port trains or needs it, cf. T-5.1's known limitations).
public final class YuE2VAEEncoder: Module {
    public let config: VAEConfig
    public let encoderConfig: VAEEncoderConfig
    @ModuleInfo(key: "encoder") var encoder: OobleckEncoder

    public init(config: VAEConfig) {
        guard let encoderConfig = config.encoderConfig else {
            preconditionFailure("VAEConfig has no encoder_config; this checkpoint cannot be encoded with")
        }
        self.config = config
        self.encoderConfig = encoderConfig
        _encoder.wrappedValue = OobleckEncoder(config: encoderConfig)
        super.init()
    }

    /// Loads `directory/config.json` and `directory/model.safetensors` (`encoder.*` only).
    public static func load(directory: URL) throws -> YuE2VAEEncoder {
        let config = try VAEConfig.load(from: directory.appendingPathComponent("config.json"))
        guard config.encoderConfig != nil else {
            throw YuE2Error.invalidRequest("\(directory.lastPathComponent): config.json has no encoder_config")
        }
        let model = YuE2VAEEncoder(config: config)
        let weights = try VAEWeightLoader.loadEncoderWeights(directory: directory)
        try WeightLoader.apply(weights, to: model, component: "vae-encoder")
        return model
    }

    /// `audio`: `[1, S, audioChannels]` (NLC), fp32, `S >= downsamplingRatio` (at least one
    /// latent frame). Returns the posterior mean, `[1, T, latentDim]` fp32, `T = S / downsamplingRatio`
    /// truncated toward zero by the encoder's own stride arithmetic.
    ///
    /// Unlike `YuE2VAE.decode` (an internal invariant: its input always comes from this port's own
    /// NAR pipeline, so a violation is a bug worth crashing on), `audio` here is a system boundary
    /// -- arbitrary user-supplied audio via `yue2 encode` (T-5.2) -- so bad input is rejected with
    /// a thrown error, never a `precondition` crash.
    public func encode(_ audio: MLXArray, taps: ((Int, MLXArray) -> Void)? = nil) throws -> MLXArray {
        guard audio.ndim == 3, audio.dim(0) == 1 else {
            throw YuE2Error.invalidRequest("expected [1, S, channels] audio, got \(audio.shape)")
        }
        guard audio.dim(2) == encoderConfig.inChannels else {
            throw YuE2Error.invalidRequest("expected \(encoderConfig.inChannels) audio channels, got \(audio.dim(2))")
        }
        guard audio.dim(1) >= config.downsamplingRatio else {
            throw YuE2Error.invalidRequest("audio too short: need at least \(config.downsamplingRatio) samples for one latent frame")
        }
        guard audio.dtype == .float32 else {
            throw YuE2Error.invalidRequest("VAE encoder input must be float32")
        }
        guard MLX.all(MLX.isFinite(audio)).item(Bool.self) else {
            throw YuE2Error.invalidRequest("VAE encoder input contains non-finite values")
        }

        let pre = encoder(audio, taps: taps) // [1, T, 2*latentDim]
        let halfWidth = pre.dim(2) / 2
        let mean = pre[0..., 0..., 0..<halfWidth]
        return mean
    }
}
