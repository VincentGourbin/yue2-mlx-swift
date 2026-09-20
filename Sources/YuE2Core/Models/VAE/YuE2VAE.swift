// YuE2VAE.swift - decoder-only VAE: load + decode (plan §2.5)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Decoder-only port of `YuE2VAE` (`decoder_only=True` in the reference): fp32 throughout,
/// no clipping (the pipeline clamps to `[-1, 1]` only at WAV export — piège n°11).
public final class YuE2VAE: Module {
    public let config: VAEConfig
    @ModuleInfo(key: "decoder") var decoder: OobleckDecoder

    public init(config: VAEConfig) {
        self.config = config
        _decoder.wrappedValue = OobleckDecoder(config: config.decoderConfig)
        super.init()
    }

    /// Loads `directory/config.json` and `directory/model.safetensors`.
    public static func load(directory: URL) throws -> YuE2VAE {
        let config = try VAEConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = YuE2VAE(config: config)
        let weights = try VAEWeightLoader.loadDecoderWeights(directory: directory)
        try WeightLoader.apply(weights, to: model, component: "vae")
        return model
    }

    /// `z`: `[1, T, latentDim]` (NLC — the NAR's own output layout, no transpose needed).
    /// Returns `[1, naturalOutputLength(frames: T), outChannels]`, fp32, unclipped.
    public func decode(_ z: MLXArray) -> MLXArray {
        precondition(
            z.ndim == 3 && z.dim(0) == 1 && z.dim(1) >= 1 && z.dim(2) == config.decoderConfig.latentDim,
            "expected nonempty [1, T, \(config.decoderConfig.latentDim)] latents, got \(z.shape)"
        )
        precondition(z.dtype == .float32, "VAE latents must be float32")
        precondition(MLX.all(MLX.isFinite(z)).item(Bool.self), "VAE latents contain non-finite values")
        return decoder(z)
    }

    /// `1920 * T - 64` for the released topology (kernel/stride/padding fixed by config;
    /// verified against `parity/tiny_vae.safetensors`' `expected_full_T*` fixtures).
    public func naturalOutputLength(frames: Int) -> Int {
        config.downsamplingRatio * frames - 64
    }

    /// Decodes `z` tile by tile, bounding peak memory, with results bit-for-bit consistent with
    /// `decode` (no crossfade: each crop's halo already supplies its full dependency support).
    public func decodeTiled(
        _ z: MLXArray,
        coreFrames: Int = 1024,
        haloFrames: Int = 16,
        onProgress: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        guard coreFrames >= 1 else {
            throw YuE2Error.invalidRequest("coreFrames must be a positive integer")
        }
        // The reference derives a per-config minimum halo from the decoder's dependency
        // graph; every released checkpoint's `decode_halo_frames` is 16, so this port checks
        // against that fixed floor rather than re-deriving the dependency interval.
        guard haloFrames >= 16 else {
            throw YuE2Error.invalidRequest("halo_frames must be at least 16 for this decoder")
        }
        let frames = z.dim(1) // z is NLC: [1, T, latentDim]
        let ratio = config.downsamplingRatio
        let total = naturalOutputLength(frames: frames)
        let tiles = (frames + coreFrames - 1) / coreFrames

        var crops: [MLXArray] = []
        var tileIndex = 0
        var start = 0
        while start < frames {
            let end = min(frames, start + coreFrames)
            let left = max(0, start - haloFrames)
            let right = min(frames, end + haloFrames)
            let tile = decode(z[0..., left..<right, 0...])
            let outStart = start * ratio
            let outEnd = min(end * ratio, total)
            let cropStart = (start - left) * ratio
            let crop = tile[0..., cropStart..<(cropStart + outEnd - outStart), 0...]
            eval(crop) // materialize and let `tile`'s larger graph be released before the next tile
            crops.append(crop)
            tileIndex += 1
            onProgress?(tileIndex, tiles)
            start += coreFrames
        }
        let audio = concatenated(crops, axis: 1)
        eval(audio)
        return audio
    }
}
