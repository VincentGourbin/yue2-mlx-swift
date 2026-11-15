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

/// Issue #4: MLX's fp16 `convTransposed1d` emits non-finite outputs for particular decode
/// window shapes (a 704...1280 core-frame band was measured on one M5 / MLX 0.31.6 host).
/// Rather than hard-coding that host-specific band into the general decode path, `decodeTiled`
/// detects a non-finite **crop** and re-decodes all tiles at half the core size. Halving changes
/// no sample on a clean window — the halo already supplies each crop's full dependency support,
/// which is why `decodeTiled` output is bit-for-bit independent of `coreFrames`. The clean path
/// therefore costs only one finiteness reduction per tile, with no extra decode and no extra
/// memory; only a genuinely bad shape pays for a second pass. Retries stop here: a non-finite
/// result at that size comes from the data (e.g. an fp16-overflowing latent), not from a kernel
/// shape, and is left to the exporter's non-finite sanitisation.
public let vaeMinRetryCoreFrames = 64

/// Whether an already-materialised crop is entirely finite. Checking the crop (not the full
/// tile) matters: NaNs confined to the halo are discarded by the crop and never reach the WAV.
/// Issue #4 instrumentation showed that forcing materialisation *inside* the decoder can change
/// which window produces NaNs, so this reduction is deliberately done after `eval(crop)`.
public func vaeCropIsFinite(_ crop: MLXArray) -> Bool {
    MLX.all(MLX.isFinite(crop)).item(Bool.self)
}

/// One step of the issue #4 retry policy: `nil` means the current pass is final. Pure and
/// backend-independent so both `decodeTiled` entry points (and their tests) share one decision.
public func nextVAECoreFrames(current: Int, foundNonFinite: Bool) -> Int? {
    guard foundNonFinite, current > vaeMinRetryCoreFrames else { return nil }
    return max(vaeMinRetryCoreFrames, current / 2)
}

/// A fallback is rare (only a bad kernel shape) and otherwise invisible, so report it on stderr
/// instead of hiding it behind `YUE2_DEBUG`. Enabling `YUE2_DEBUG` also turns on debug-only
/// `isFinite` assertions inside `SnakeBeta`, which trap before the crop is ever checked — the
/// reason verification runs without it.
public func logVAETileRetry(from current: Int, to next: Int) {
    FileHandle.standardError.write(
        Data("yue2: non-finite VAE tile at \(current) core frames; re-decoding at \(next)\n".utf8))
}

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

    // Issue #4: if a window lands on an fp16 `convTransposed1d` shape that emits NaNs, discard
    // the whole pass and re-decode at half the core size. Every backend goes through this path,
    // including callers that request an explicit `--vae-core-frames` / `--core-frames`.
    var effectiveCore = coreFrames
    var crops: [MLXArray] = []
    while true {
        // At the minimum tile a non-finite result is data-side, not shape-side: keep those crops
        // (the exporter sanitises them) instead of retrying forever.
        let finalPass = effectiveCore <= vaeMinRetryCoreFrames
        crops.removeAll(keepingCapacity: true)
        let windows = planVAETiles(
            frames: frames, coreFrames: effectiveCore, haloFrames: haloFrames,
            enumerated: backend.enumeratedFrameCounts)
        var foundNonFinite = false
        for (index, window) in windows.enumerated() {
            YuE2GPUGate.shared.wait()
            let tile = try await backend.decode(z[0..., window.windowStart..<window.windowEnd, 0...])
            let outStart = window.coreStart * ratio
            let outEnd = min(window.coreEnd * ratio, total)
            let cropStart = (window.coreStart - window.windowStart) * ratio
            let crop = tile[0..., cropStart..<(cropStart + outEnd - outStart), 0...]
            eval(crop)
            if !finalPass, !vaeCropIsFinite(crop) {
                foundNonFinite = true
                break
            }
            crops.append(crop)
            onProgress?(index + 1, windows.count)
        }
        guard let next = nextVAECoreFrames(current: effectiveCore, foundNonFinite: foundNonFinite) else {
            break
        }
        logVAETileRetry(from: effectiveCore, to: next)
        effectiveCore = next
    }
    let audio = concatenated(crops, axis: 1)
    eval(audio)
    return audio
}
