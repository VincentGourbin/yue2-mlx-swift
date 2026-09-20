// TokenGeneratorTests.swift - generate() parity against parity/tiny_lm.safetensors (T-2.7)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

@Suite("TokenGenerator")
struct TokenGeneratorTests {
    /// Matches `Scripts/reference/tiny_lm.py`'s `greedy_and_cfg`: `EOD`/`ABC_END`/`MUSIC_END`/
    /// `CODEC_OFFSET`/`CODEC_SIZE` monkeypatched to `(5, 6, 7, 8, 20)` for that fixture only,
    /// phase `"semantic"` ⇒ bounds = `([CODEC_OFFSET, CODEC_OFFSET+CODEC_SIZE), MUSIC_END)`
    /// = `(8..<28, 7)` — `MUSIC_END` deliberately sits *below* `allowedRange`.
    private static let bounds = LogitsBounds(allowedRange: 8..<28, endToken: 7)

    private static let baseSampling = try! Sampling(
        temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1,
        penaltyWindow: 1, minTokens: 0, maxTokens: 8)

    private func expectedTokens(_ raw: [String: MLXArray], _ key: String) -> [Int] {
        raw[key]!.asType(.int32).asArray(Int32.self).map(Int.init)
    }

    @Test func greedyMatchesFixture() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let head = RestrictedHead(lmHead: model.lmHead, bounds: Self.bounds)
        let result = try TokenGenerator.generate(
            model: model, head: head, prefix: [2, 3, 4], sampling: Self.baseSampling, seed: 0,
            bounds: Self.bounds)
        #expect(result.tokens == expectedTokens(raw, "greedy_tokens"))
        let expectedTruncated = raw["greedy_truncated"]!.asType(.int32).item(Int32.self) != 0
        #expect(result.truncated == expectedTruncated)
    }

    @Test func cfgMatchesFixture() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let head = RestrictedHead(lmHead: model.lmHead, bounds: Self.bounds)
        let result = try TokenGenerator.generate(
            model: model, head: head, prefix: [2, 3, 4], sampling: Self.baseSampling, seed: 0,
            bounds: Self.bounds, negative: [2, 3], cfgScale: 1.5)
        #expect(result.tokens == expectedTokens(raw, "cfg_tokens"))
    }

    /// With `min_tokens: 3`, `MUSIC_END` (local bound `7`) cannot appear among the first 3
    /// emitted tokens — `LogitsProcessor` masks it to `-inf` while `step < min_tokens`.
    @Test func minTokensBlocksEndBeforeIndex() throws {
        let model = try loadedTinyModel()
        let head = RestrictedHead(lmHead: model.lmHead, bounds: Self.bounds)
        let sampling = try Sampling(
            temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1,
            penaltyWindow: 1, minTokens: 3, maxTokens: 8)
        var seenTokens: [Int] = []
        _ = try TokenGenerator.generate(
            model: model, head: head, prefix: [2, 3, 4], sampling: sampling, seed: 0,
            bounds: Self.bounds, onToken: { seenTokens.append($0) })
        #expect(!seenTokens.prefix(3).contains(Self.bounds.endToken))
    }

    /// `cancel` returning `true` on its 2nd call (i.e. during step 1) must raise
    /// `.cancelled` before any further generation happens.
    @Test func cancellationStopsAtSecondStep() throws {
        let model = try loadedTinyModel()
        let head = RestrictedHead(lmHead: model.lmHead, bounds: Self.bounds)
        var callCount = 0
        do {
            _ = try TokenGenerator.generate(
                model: model, head: head, prefix: [2, 3, 4], sampling: Self.baseSampling, seed: 0,
                bounds: Self.bounds,
                cancel: {
                    callCount += 1
                    return callCount >= 2
                })
            Issue.record("expected YuE2Error.cancelled")
        } catch YuE2Error.cancelled {
            // expected
        } catch {
            Issue.record("unexpected error: \(error)")
        }
        #expect(callCount == 2)
    }
}
