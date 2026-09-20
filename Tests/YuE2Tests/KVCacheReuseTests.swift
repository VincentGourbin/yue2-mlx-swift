// KVCacheReuseTests.swift - O2 (T-4.2): NAR reuses the semantic AR cache instead of reprefilling
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a - b)).item(Float.self)
}

/// Forwards `tokens` through a fresh `KVCache`, simulating what `TokenGenerator.generate` would
/// have left behind after processing exactly that many tokens (semantic AR generation, possibly
/// stopped early).
private func cachePrefilled(with tokens: [Int], model: YuE2ForCausalLM) -> KVCache {
    let cache = KVCache()
    guard !tokens.isEmpty else { return cache }
    let block = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
    _ = model.model.forwardAR(tokens: block, cache: cache)
    cache.evalAll()
    return cache
}

@Suite("KVCacheReuse")
struct KVCacheReuseTests {
    private static let arTokens = [2, 3, 4, 5]

    private func referenceVelocity(model: YuE2ForCausalLM, noise: MLXArray) -> MLXArray {
        let chunk = NARChunk(arTokens: Self.arTokens, noise: noise)
        let fresh = CachedNAR(model: model, chunk: chunk)
        return fresh.velocity(state: noise, rawT: 0)
    }

    @Test func reusedCacheMatchesFreshPrefillWhenComplete() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])

        // The semantic loop's own cache already holds every token in `arTokens` (the common
        // case: the end token's lookahead forward pass already ran before the loop broke).
        let cache = cachePrefilled(with: Self.arTokens, model: model)
        #expect(cache.offset == Self.arTokens.count)
        let toppedUp = Synthesizer.topUp(cache: cache, to: Self.arTokens, model: model)
        #expect(toppedUp.offset == Self.arTokens.count)

        let chunk = NARChunk(arTokens: Self.arTokens, noise: noise)
        let reused = CachedNAR(model: model, chunk: chunk, cache: toppedUp)
        let expected = referenceVelocity(model: model, noise: noise)
        #expect(maxAbsDiff(reused.velocity(state: noise, rawT: 0), expected) <= 1e-6)
    }

    @Test func truncatedCacheIsCaughtUpBeforeReuse() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])

        // The semantic loop stopped mid-`arTokens` (truncated: the end token, and possibly one
        // trailing content token, never got their own forward pass into the cache).
        let cache = cachePrefilled(with: Array(Self.arTokens.dropLast(2)), model: model)
        #expect(cache.offset == Self.arTokens.count - 2)
        let toppedUp = Synthesizer.topUp(cache: cache, to: Self.arTokens, model: model)
        #expect(toppedUp.offset == Self.arTokens.count)

        let chunk = NARChunk(arTokens: Self.arTokens, noise: noise)
        let reused = CachedNAR(model: model, chunk: chunk, cache: toppedUp)
        let expected = referenceVelocity(model: model, noise: noise)
        #expect(maxAbsDiff(reused.velocity(state: noise, rawT: 0), expected) <= 1e-6)
    }
}
