// BackboneParityTests.swift - LM backbone parity against parity/tiny_lm.safetensors (T-2.x)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import Testing
@testable import YuE2Core

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a - b)).item(Float.self)
}

@Suite("BackboneParity")
struct BackboneParityTests {
    @Test func loadsTinyWithFullCoverage() throws {
        _ = try loadedTinyModel()
    }

    @Test func logitsFullMatchesFixture() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let ids = raw["ids_full"]!.asType(.int32)
        let logits = model.forwardAR(tokens: ids, cache: nil, keepLast: false)
        #expect(maxAbsDiff(logits, raw["logits_full"]!) <= 1e-4)
    }

    @Test func hiddenAfterLayerBisection() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let ids = raw["ids_full"]!.asType(.int32)
        var taps: [Int: MLXArray] = [:]
        _ = model.forwardAR(tokens: ids, cache: nil, keepLast: false) { i, h in taps[i] = h }
        for i in 0..<2 {
            #expect(maxAbsDiff(taps[i]!, raw["hidden_after_layer_\(i)"]!) <= 1e-4)
        }
    }

    /// `rope_cos`/`rope_sin` in the fixture are the raw half-size frequencies (head_dim/2 = 2),
    /// reused for both halves of `rotate_half`: for `x = ones(4)`,
    /// `rotated = [cos0 - sin0, cos1 - sin1, cos0 + sin0, cos1 + sin1]` (plan §2.2).
    @Test func ropeMatchesFixture() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let cos = raw["rope_cos"]!
        let sin = raw["rope_sin"]!
        let positions = [0, 1, 4, 9, 10]

        let rope = MLXNN.RoPE(dimensions: 4, traditional: false, base: 1_000_000)
        for (i, position) in positions.enumerated() {
            let rotated = rope(MLXArray.ones([1, 1, 1, 4]), offset: position)
            let got = rotated.reshaped([4])
            let c0 = cos[0, i, 0].item(Float.self)
            let c1 = cos[0, i, 1].item(Float.self)
            let s0 = sin[0, i, 0].item(Float.self)
            let s1 = sin[0, i, 1].item(Float.self)
            let expected = MLXArray([c0 - s0, c1 - s1, c0 + s0, c1 + s1])
            #expect(maxAbsDiff(got, expected) <= 1e-4)
        }
    }

    /// Chunked prefill with a real cache reproduces the full-sequence logits exactly — i.e.
    /// `.causal` alone (no hand-built bottom-right mask) is already correct for `T > 1` against
    /// a non-empty cache (see `Backbone.forwardAR`'s doc comment).
    @Test func chunkedPrefillMatchesFullLogits() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let ids = raw["ids_full"]!.asType(.int32)
        let cache = KVCache()
        var chunks: [MLXArray] = []
        for (i, bounds) in [(0, 2), (2, 5), (5, 6)].enumerated() {
            let block = ids[0..., bounds.0..<bounds.1]
            let logits = model.forwardAR(tokens: block, cache: cache, keepLast: false)
            #expect(maxAbsDiff(logits, raw["logits_chunk_\(i)"]!) <= 1e-4)
            chunks.append(logits)
        }
        #expect(maxAbsDiff(concatenated(chunks, axis: 1), raw["logits_full"]!) <= 1e-4)
    }

    @Test func explicitPositionsMatchFixture() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let ids = raw["ids_full"]!.asType(.int32)[0..., 0..<4]
        let positions = raw["positions_explicit"]!.asType(.int32).reshaped([4])
        let logits = model.forwardAR(tokens: ids, cache: nil, keepLast: false, positions: positions)
        #expect(maxAbsDiff(logits, raw["logits_explicit"]!) <= 1e-4)
    }

    @Test func decodeOneTokenAfterPrefillMatchesFullLogitsColumn() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let ids = raw["ids_full"]!.asType(.int32)
        let cache = KVCache()
        _ = model.forwardAR(tokens: ids[0..., 0..<5], cache: cache, keepLast: false)
        let logits = model.forwardAR(tokens: ids[0..., 5..<6], cache: cache, keepLast: false)
        let expectedColumn = raw["logits_full"]![0..., 5..<6, 0...]
        #expect(maxAbsDiff(logits, expectedColumn) <= 1e-4)
    }

    @Test func tokenGeneratorPrefillMatchesLastLogitsColumn() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let ids = raw["ids_full"]!.asType(.int32).asArray(Int32.self).map(Int.init)
        let result = TokenGenerator.prefill(model: model, tokens: ids, stepSize: 2)
        let expected = raw["logits_full"]![0..., (ids.count - 1)..<ids.count, 0...]
        #expect(maxAbsDiff(result.lastLogits, expected) <= 1e-4)
        #expect(result.cache.offset == ids.count)
    }
}
