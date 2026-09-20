// NARChunk.swift - one original acoustic chunk: AR prefix tokens + its noise slice (plan §2.4)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXRandom

/// One original acoustic chunk fed to `CachedNAR` (T-3.2): the full AR prefix (semantic prefix
/// plus this chunk's codec tokens, offset into the semantic id space, plus `MUSIC_END`) and this
/// chunk's slice of the song-wide noise draw.
public struct NARChunk {
    public let arTokens: [Int]
    /// `[T_c, 64]`, float32.
    public let noise: MLXArray

    /// Builds a chunk directly from already-known AR tokens and noise — for callers (parity
    /// commands, tests) that already have both, rather than deriving them via `songChunks`.
    public init(arTokens: [Int], noise: MLXArray) {
        self.arTokens = arTokens
        self.noise = noise
    }
}

extension NARChunk {
    /// Splits `codec` into `chunkRanges`-sized pieces and pairs each with `prefix + that piece's
    /// codec ids (offset by `codecOffset`) + MUSIC_END`, plus its slice of one song-wide noise
    /// draw. Mirrors `song_chunks` in `nar.py`.
    ///
    /// The noise is drawn **once for the whole song** (`[codec.count, 64]`), then sliced per
    /// chunk — never redrawn per chunk. PyTorch's CPU RNG isn't reproducible from MLX, so parity
    /// tests inject the reference's exact noise via `noise:` instead of letting this draw one.
    public static func songChunks(
        prefix: [Int], codec: [Int], seed: UInt64, context: Int = YuE2Token.context,
        noise: MLXArray? = nil
    ) throws -> [NARChunk] {
        guard !prefix.isEmpty, prefix.allSatisfy({ $0 >= 0 }) else {
            throw YuE2Error.invalidRequest("prefix must be a nonempty sequence of nonnegative token IDs")
        }
        guard !codec.isEmpty, codec.allSatisfy({ $0 >= 0 && $0 < YuE2Token.codecSize }) else {
            throw YuE2Error.invalidRequest("codec must be a nonempty sequence of IDs in [0, codecSize)")
        }
        guard (1...YuE2Token.context).contains(context) else {
            throw YuE2Error.invalidRequest("context must be an integer in 1...\(YuE2Token.context)")
        }
        let fullNoise: MLXArray
        if let noise {
            guard noise.dim(0) == codec.count, noise.dim(1) == 64 else {
                throw YuE2Error.invalidRequest("injected noise must be shaped [codec.count, 64]")
            }
            fullNoise = noise
        } else {
            fullNoise = MLXRandom.normal([codec.count, 64], type: Float.self, key: MLXRandom.key(seed))
        }
        let ranges = try Prefixes.chunkRanges(frames: codec.count, prefixTokens: prefix.count, context: context)
        return ranges.map { a, b in
            let arTokens = prefix + codec[a..<b].map { $0 + YuE2Token.codecOffset } + [YuE2Token.musicEnd]
            return NARChunk(arTokens: arTokens, noise: fullNoise[a..<b])
        }
    }
}
