// VAEDecoding.swift - backend-agnostic VAE decoding protocol (T-6.3, E3)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// A VAE decoder backend: MLX (`YuE2VAE`, always available) or Core AI (`CoreAIVAEDecoder`,
/// GPU/Neural Engine, iOS 27+/macOS 27+ only). `decode` is `async throws` so a synchronous,
/// non-throwing implementation (`YuE2VAE.decode`) can satisfy it as-is (Swift allows a sync
/// function to fulfill an async protocol requirement) while Core AI's genuinely-async
/// `InferenceFunction.run` also fits without a wrapper.
public protocol VAEDecoding {
    var config: VAEConfig { get }

    /// Latent frame counts this backend can decode in one call when it is restricted to a fixed
    /// set (Core AI's enumerated shapes, `CoreAIVAEDecoder.enumeratedFrames`); `nil` for a
    /// backend that accepts any length (MLX). `decodeTiled(using:)` uses it to size every tile's
    /// window to an exact entry instead of letting the backend zero-pad (see `planVAETiles`).
    var enumeratedFrameCounts: [Int]? { get }

    /// `z`: `[1, T, latentDim]` NLC. Returns `[1, naturalOutputLength(frames: T), outChannels]`,
    /// fp32, unclipped — same contract as `YuE2VAE.decode`.
    func decode(_ z: MLXArray) async throws -> MLXArray
}

extension VAEDecoding {
    public var enumeratedFrameCounts: [Int]? { nil }

    /// `1920 * T - 64` for the released topology (same formula as `YuE2VAE.naturalOutputLength`).
    public func naturalOutputLength(frames: Int) -> Int {
        config.downsamplingRatio * frames - 64
    }
}

extension YuE2VAE: VAEDecoding {}

/// One tile of a tiled decode: the latent window `[windowStart, windowEnd)` handed to the
/// backend and the core `[coreStart, coreEnd)` whose output is kept. `paddedFrames` is what the
/// backend will still have to zero-pad (only when the whole signal is shorter than the smallest
/// enumerated shape — every other case is absorbed by widening the window).
public struct VAETileWindow: Equatable, Sendable {
    public let windowStart: Int
    public let windowEnd: Int
    public let coreStart: Int
    public let coreEnd: Int
    public let paddedFrames: Int

    public var windowFrames: Int { windowEnd - windowStart }
}

/// Splits `frames` latent frames into `coreFrames` cores with `haloFrames` of context on each
/// side (`YuE2VAE.decodeTiled`'s crop-only algorithm). For a backend restricted to
/// `enumerated` frame counts, each window is then widened — first to the right, then to the
/// left, never past the signal — until it matches the smallest entry that fits, so the backend
/// receives real latents instead of zero padding. That matters (T-6.3 follow-up): zero *latent*
/// frames are not the decoder's own zero activation padding — a zero latent still goes through
/// conv bias and SnakeBeta, and contaminates the last ≈14 real frames before it (the decoder's
/// dependency interval), which is exactly where the final, short tile lost 34 dB when it was
/// padded from 92 to 288 frames. Widening leaves the crop untouched: the extra frames are only
/// ever more context, never output. `enumerated == nil` reproduces the original plan exactly.
public func planVAETiles(frames: Int, coreFrames: Int, haloFrames: Int, enumerated: [Int]?) -> [VAETileWindow] {
    var windows: [VAETileWindow] = []
    var start = 0
    while start < frames {
        let end = min(frames, start + coreFrames)
        var left = max(0, start - haloFrames)
        var right = min(frames, end + haloFrames)
        var padded = 0
        if let enumerated, let need = enumerated.sorted().first(where: { $0 >= right - left }) {
            var extra = need - (right - left)
            let growRight = min(extra, frames - right)
            right += growRight
            extra -= growRight
            let growLeft = min(extra, left)
            left -= growLeft
            extra -= growLeft
            padded = extra
        }
        windows.append(VAETileWindow(
            windowStart: left, windowEnd: right, coreStart: start, coreEnd: end, paddedFrames: padded))
        start += coreFrames
    }
    return windows
}

/// Decodes `z` tile by tile through any `VAEDecoding` backend, bounding each call's latent
/// frame count — mirrors `YuE2VAE.decodeTiled`'s algorithm (crop-only, no crossfade) but async
/// and backend-agnostic. Windows come from `planVAETiles`, so a backend with enumerated shapes
/// (`CoreAIVAEDecoder`, 288/544/1056) gets exact-shape windows whenever the signal is long enough.
public func decodeTiled(
    using backend: any VAEDecoding,
    _ z: MLXArray,
    coreFrames: Int = 1024,
    haloFrames: Int = 16,
    onProgress: ((Int, Int) -> Void)? = nil
) async throws -> MLXArray {
    guard coreFrames >= 1 else {
        throw YuE2Error.invalidRequest("coreFrames must be a positive integer")
    }
    guard haloFrames >= 16 else {
        throw YuE2Error.invalidRequest("halo_frames must be at least 16 for this decoder")
    }
    let frames = z.dim(1)
    let ratio = backend.config.downsamplingRatio
    let total = backend.naturalOutputLength(frames: frames)
    let windows = planVAETiles(
        frames: frames, coreFrames: coreFrames, haloFrames: haloFrames,
        enumerated: backend.enumeratedFrameCounts)

    var crops: [MLXArray] = []
    for (index, window) in windows.enumerated() {
        let tile = try await backend.decode(z[0..., window.windowStart..<window.windowEnd, 0...])
        let outStart = window.coreStart * ratio
        let outEnd = min(window.coreEnd * ratio, total)
        let cropStart = (window.coreStart - window.windowStart) * ratio
        let crop = tile[0..., cropStart..<(cropStart + outEnd - outStart), 0...]
        eval(crop)
        crops.append(crop)
        onProgress?(index + 1, windows.count)
    }
    let audio = concatenated(crops, axis: 1)
    eval(audio)
    return audio
}
