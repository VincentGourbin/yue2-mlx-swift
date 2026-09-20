// SamplingTests.swift - LogitsProcessor parity against parity/sampling.safetensors (T-2.5)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let samplingFixture = repoRoot.appendingPathComponent("parity/sampling.safetensors")

/// Mirrors the per-`cfg{i}` JSON blob `Scripts/reference/tiny_sampling.py` writes into the
/// safetensors metadata (one dictionary per configuration, decoded rather than hardcoded so
/// the test tracks the fixture instead of duplicating its literals).
private struct FixtureSamplingConfig: Decodable {
    let phase: String
    let legacyOff: Bool
    let dtype: String
    let temperature: Double
    let topP: Double
    let topK: Int
    let repetitionPenalty: Double
    let penaltyWindow: Int
    let minTokens: Int
    let maxTokens: Int

    enum CodingKeys: String, CodingKey {
        case phase
        case legacyOff = "legacy_off"
        case dtype
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case penaltyWindow = "penalty_window"
        case minTokens = "min_tokens"
        case maxTokens = "max_tokens"
    }
}

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
}

@Suite("Sampling")
struct SamplingTests {
    /// One test per `cfg{1...6}` of `parity/sampling.safetensors`: abc vs semantic bounds,
    /// temperature 0 (argmax-only) vs >0, top_p 1 (disabled) vs <1, and `legacy_off` (bf16,
    /// no fp32 upcast, top-p keeps 3 instead of 1) — see `tiny_sampling.py` for the exact
    /// six configurations.
    @Test(arguments: ["cfg1", "cfg2", "cfg3", "cfg4", "cfg5", "cfg6"])
    func matchesFixture(_ name: String) throws {
        let (raw, metadata) = try loadArraysAndMetadata(url: samplingFixture)
        let json = try #require(metadata[name]?.data(using: .utf8))
        let cfg = try JSONDecoder().decode(FixtureSamplingConfig.self, from: json)

        let sampling = try Sampling(
            temperature: cfg.temperature, topP: cfg.topP, topK: cfg.topK,
            repetitionPenalty: cfg.repetitionPenalty, penaltyWindow: cfg.penaltyWindow,
            minTokens: cfg.minTokens, maxTokens: cfg.maxTokens)
        let bounds = cfg.phase == "abc" ? LogitsBounds.abc : LogitsBounds.semantic
        let processor = LogitsProcessor(
            sampling: sampling, bounds: bounds, vocabSize: YuE2Token.vocabSize, legacyOff: cfg.legacyOff)

        let history = raw["\(name)_history"]!.asArray(Int32.self).map(Int.init)
        for token in history { processor.push(token: token) }
        let step = Int(raw["\(name)_step"]!.asArray(Int32.self)[0])

        var logits = raw["logits"]!
        if cfg.dtype.contains("bfloat16") { logits = logits.asType(.bfloat16) }
        let scores = processor.scores(logits: logits, step: step)

        // "positions -inf identiques": the finite-entry set must match, up to one boundary
        // flip in bf16 (legacy_off/cfg5) — MLX's and PyTorch's bf16 cumsum/softmax round
        // differently right at the top-p 0.95 cumulative-probability cutoff, which can move
        // a single low-magnitude entry across the threshold; fp32 configs match exactly.
        let finiteIndices = raw["\(name)_finite_indices"]!.asArray(Int32.self).map(Int.init)
        let expectedIndexSet = Set(finiteIndices)
        let gotMask = MLX.isFinite(scores).reshaped([YuE2Token.vocabSize]).asArray(Bool.self)
        let gotIndexSet = Set(gotMask.indices.filter { gotMask[$0] })
        let allowedSymmetricDiff = cfg.legacyOff ? 1 : 0
        #expect(gotIndexSet.symmetricDifference(expectedIndexSet).count <= allowedSymmetricDiff)

        let commonIndices = gotIndexSet.intersection(expectedIndexSet).sorted()
        let idx = MLXArray(commonIndices.map { Int32($0) }).expandedDimensions(axis: 0)
        let gotValues = takeAlong(scores, idx, axis: -1)
        let expectedByIndex = Dictionary(
            uniqueKeysWithValues: zip(finiteIndices, raw["\(name)_finite_values"]!.asArray(Float.self)))
        let expectedValues = MLXArray(commonIndices.map { expectedByIndex[$0]! }).expandedDimensions(axis: 0)
        // bf16 (cfg5) carries ~2 decimal digits of precision; fp32 configs hold to 1e-5.
        let tolerance: Float = cfg.dtype.contains("bfloat16") ? 5e-2 : 1e-5
        #expect(maxAbsDiff(gotValues, expectedValues) <= tolerance)

        let expectedArgmax = Int(raw["\(name)_argmax"]!.asArray(Int32.self)[0])
        let gotArgmax = Int(MLX.argMax(scores, axis: -1).item(Int32.self))
        #expect(gotArgmax == expectedArgmax)
    }

    /// Repetition penalty vs a naive CPU loop (3 windows with duplicate ids), isolated from
    /// top-k/top-p/temperature by using `temperature: 0` (short-circuits `scores()` right
    /// after the penalty step — see `sampling.py`'s `distribution()`).
    @Test(arguments: [
        [1, 1, 3, 5, 5, 5],
        [0, 0, 0, 0],
        [2, 7, 2, 7, 2, 7, 9],
    ])
    func windowPenaltyMatchesNaiveCPU(_ window: [Int]) throws {
        let vocabSize = 12
        let bounds = LogitsBounds(allowedRange: 0..<vocabSize, endToken: vocabSize - 1)
        let sampling = try Sampling(
            temperature: 0, topP: 1, topK: vocabSize, repetitionPenalty: 1.3,
            penaltyWindow: 10, minTokens: 0, maxTokens: 1)
        let processor = LogitsProcessor(sampling: sampling, bounds: bounds, vocabSize: vocabSize, legacyOff: false)
        for token in window { processor.push(token: token) }

        let logitsValues: [Float] = [-2.5, -1.0, -0.3, 0.1, 0.4, 1.2, 2.0, -0.7, 0.9, -1.8, 0.2, 1.5]
        let logits = MLXArray(logitsValues).reshaped([1, vocabSize])
        let got = processor.scores(logits: logits, step: 0)

        var freq = [Int](repeating: 0, count: vocabSize)
        for token in window { freq[token] += 1 }
        var expected = logitsValues
        for i in 0..<vocabSize {
            let alpha = pow(1.3, Double(freq[i]))
            expected[i] = expected[i] < 0
                ? Float(Double(expected[i]) * alpha)
                : Float(Double(expected[i]) / alpha)
        }
        #expect(maxAbsDiff(got, MLXArray(expected).reshaped([1, vocabSize])) <= 1e-5)
    }

    /// A tie at the k-th largest value must be kept by both entries (the reference masks
    /// with `scores < threshold`, a strict inequality, so ties at the threshold survive).
    @Test func topKKeepsTies() throws {
        let vocabSize = 8
        let bounds = LogitsBounds(allowedRange: 0..<vocabSize, endToken: vocabSize - 1)
        let sampling = try Sampling(
            temperature: 1, topP: 1, topK: 3, repetitionPenalty: 1.0,
            penaltyWindow: 1, minTokens: 0, maxTokens: 1)
        let processor = LogitsProcessor(sampling: sampling, bounds: bounds, vocabSize: vocabSize, legacyOff: false)

        // Top 3 by value would be indices 7 (3.0), 6 (2.0), then a 3-way tie at 1.0
        // (indices 0, 3, 5) for the 3rd slot: all three must survive.
        let logitsValues: [Float] = [1.0, -1.0, 0.0, 1.0, -2.0, 1.0, 2.0, 3.0]
        let logits = MLXArray(logitsValues).reshaped([1, vocabSize])
        let got = processor.scores(logits: logits, step: 0)
        let finite = MLX.isFinite(got).reshaped([vocabSize]).asArray(Bool.self)
        #expect(finite == [true, false, false, true, false, true, true, true])
    }

    // MARK: - RestrictedHead (T-2.6, O1)

    /// Two phase-shaped bounds over the tiny model's 32-id vocabulary: one abc-like (a
    /// contiguous low range, end token past it) and one semantic-like (a contiguous high
    /// range, end token *before* it) — the endToken deliberately sits outside, and on both
    /// sides of, its `allowedRange` so ordering assumptions can't sneak by unexercised.
    private static let abcLikeBounds = LogitsBounds(allowedRange: 0..<20, endToken: 25)
    private static let semanticLikeBounds = LogitsBounds(allowedRange: 10..<30, endToken: 5)

    /// `[1, T, hiddenSize]` final hidden states from the tiny checkpoint, and its full
    /// `[1, T, 32]` `lm_head` logits — the ground truth `RestrictedHead` is checked against
    /// (self-consistency with `model.lmHead`, not a Python fixture: T-2.6 has no ⛔ gate).
    private func tinyHiddenAndFullLogits() throws -> (hidden: MLXArray, fullLogits: MLXArray) {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let ids = raw["ids_full"]!.asType(.int32)
        let hidden = model.model.forwardAR(tokens: ids, cache: nil)
        return (hidden, model.lmHead(hidden))
    }

    @Test(arguments: [abcLikeBounds, semanticLikeBounds])
    func globalIDsAreAllowedRangeThenEndToken(_ bounds: LogitsBounds) throws {
        let head = RestrictedHead(lmHead: try loadedTinyModel().lmHead, bounds: bounds)
        #expect(Array(head.globalIDs.dropLast()) == Array(bounds.allowedRange))
        #expect(head.globalIDs.last == bounds.endToken)
        #expect(head.globalIDs.count == bounds.allowedRange.count + 1)
    }

    /// The whole point of O1: the gathered weight has `n` rows, not `vocabSize` — `logits`
    /// comes out `[..., n]`, never `[..., vocabSize]`.
    @Test(arguments: [abcLikeBounds, semanticLikeBounds])
    func restrictedLogitsAreVocabSizedDown(_ bounds: LogitsBounds) throws {
        let (hidden, _) = try tinyHiddenAndFullLogits()
        let head = RestrictedHead(lmHead: try loadedTinyModel().lmHead, bounds: bounds)
        let got = head.logits(hidden: hidden)
        #expect(got.shape.last == head.globalIDs.count)
        #expect(head.globalIDs.count < 32)
    }

    @Test(arguments: [abcLikeBounds, semanticLikeBounds])
    func restrictedLogitsMatchFullHeadGather(_ bounds: LogitsBounds) throws {
        let (hidden, fullLogits) = try tinyHiddenAndFullLogits()
        let head = RestrictedHead(lmHead: try loadedTinyModel().lmHead, bounds: bounds)
        let got = head.logits(hidden: hidden)
        let idx = MLXArray(head.globalIDs.map { Int32($0) })
        let expected = MLX.take(fullLogits, idx, axis: -1)
        #expect(maxAbsDiff(got, expected) <= 1e-6)
    }

    @Test(arguments: [abcLikeBounds, semanticLikeBounds])
    func globalAndLocalRoundTrip(_ bounds: LogitsBounds) throws {
        let head = RestrictedHead(lmHead: try loadedTinyModel().lmHead, bounds: bounds)
        for local in [0, head.globalIDs.count / 2, head.globalIDs.count - 1] {
            let global = head.global(local)
            #expect(head.local(global) == local)
        }
        // A vocabulary id the head does not carry at all has no local index.
        let outside = Set(0..<32).subtracting(head.globalIDs).first!
        #expect(head.local(outside) == nil)
    }

    /// `LogitsBounds.local(count:)` allows every row except the last (the phase's end token,
    /// which stands last in `RestrictedHead.globalIDs`).
    @Test func localBoundsAllowsEveryRowButTheLast() throws {
        let n = 21
        let bounds = LogitsBounds.local(count: n)
        #expect(bounds.allowedRange == 0..<(n - 1))
        #expect(bounds.endToken == n - 1)
    }

    /// The restricted (local-space) chain and the full (global-space) chain must agree on
    /// `argmax`: masking first-then-computing and computing-only-the-legal-rows discard the
    /// same rows, so the highest surviving score — and hence its id, once mapped back through
    /// `global(_:)` — is identical either way. `temperature: 0` isolates the mask + min_tokens
    /// + penalty stages (no top-k/top-p) so the comparison is exact end to end.
    @Test(arguments: [abcLikeBounds, semanticLikeBounds])
    func restrictedChainMatchesGlobalChainArgmax(_ bounds: LogitsBounds) throws {
        let (hidden, fullLogits) = try tinyHiddenAndFullLogits()
        let lastLogits = fullLogits[0..., (fullLogits.dim(1) - 1)..., 0...].reshaped([1, 32])
        let lastHidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        let head = RestrictedHead(lmHead: try loadedTinyModel().lmHead, bounds: bounds)
        let restrictedLogits = head.logits(hidden: lastHidden).reshaped([1, head.globalIDs.count])

        let sampling = try Sampling(
            temperature: 0, topP: 1, topK: 32, repetitionPenalty: 1.0,
            penaltyWindow: 1, minTokens: 0, maxTokens: 1)

        let globalProcessor = LogitsProcessor(sampling: sampling, bounds: bounds, vocabSize: 32, legacyOff: false)
        let globalArgmax = Int(MLX.argMax(globalProcessor.scores(logits: lastLogits, step: 0), axis: -1).item(Int32.self))

        let localProcessor = LogitsProcessor(
            sampling: sampling, bounds: .local(count: head.globalIDs.count), vocabSize: head.globalIDs.count,
            legacyOff: false)
        let localArgmax = Int(
            MLX.argMax(localProcessor.scores(logits: restrictedLogits, step: 0), axis: -1).item(Int32.self))

        #expect(head.global(localArgmax) == globalArgmax)
    }

    /// The same equivalence holds with top-k and top-p active (not just the mask/temperature-0
    /// short circuit): the top-ranked id always survives both stages (top-k's threshold and
    /// top-p's `keepFirst` both preserve rank 0), so restricting first cannot change it.
    @Test func restrictedChainMatchesGlobalChainArgmaxWithTopKTopP() throws {
        let bounds = SamplingTests.semanticLikeBounds
        let (hidden, fullLogits) = try tinyHiddenAndFullLogits()
        let lastLogits = fullLogits[0..., (fullLogits.dim(1) - 1)..., 0...].reshaped([1, 32])
        let lastHidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        let head = RestrictedHead(lmHead: try loadedTinyModel().lmHead, bounds: bounds)
        let restrictedLogits = head.logits(hidden: lastHidden).reshaped([1, head.globalIDs.count])

        let sampling = try Sampling(
            temperature: 0.8, topP: 0.9, topK: 10, repetitionPenalty: 1.0,
            penaltyWindow: 1, minTokens: 0, maxTokens: 1)

        let globalProcessor = LogitsProcessor(sampling: sampling, bounds: bounds, vocabSize: 32, legacyOff: false)
        let globalArgmax = Int(MLX.argMax(globalProcessor.scores(logits: lastLogits, step: 0), axis: -1).item(Int32.self))

        let localProcessor = LogitsProcessor(
            sampling: sampling, bounds: .local(count: head.globalIDs.count), vocabSize: head.globalIDs.count,
            legacyOff: false)
        let localArgmax = Int(
            MLX.argMax(localProcessor.scores(logits: restrictedLogits, step: 0), axis: -1).item(Int32.self))

        #expect(head.global(localArgmax) == globalArgmax)
    }

    /// Pushing the same generated tokens as local ids (into a local-space processor) or as
    /// their global equivalents (into a global-space processor) must apply the identical
    /// repetition penalty: the window-penalty math only cares about identity/frequency, and
    /// `globalIDs`/`local(_:)` are a bijection, so relabeling changes nothing.
    @Test func repetitionPenaltyIsInvariantUnderLocalGlobalRelabeling() throws {
        let bounds = SamplingTests.abcLikeBounds
        let head = RestrictedHead(lmHead: try loadedTinyModel().lmHead, bounds: bounds)
        let n = head.globalIDs.count

        let sampling = try Sampling(
            temperature: 0, topP: 1, topK: n, repetitionPenalty: 1.3,
            penaltyWindow: 10, minTokens: 0, maxTokens: 1)

        let localHistory = [0, 3, 3, n - 1, 5]
        let globalHistory = localHistory.map { head.global($0) }

        let localLogitsValues = (0..<n).map { Float($0) - Float(n) / 2 }
        let localProcessor = LogitsProcessor(sampling: sampling, bounds: .local(count: n), vocabSize: n, legacyOff: false)
        for token in localHistory { localProcessor.push(token: token) }
        let localScores = localProcessor.scores(logits: MLXArray(localLogitsValues).reshaped([1, n]), step: 0)

        let globalProcessor = LogitsProcessor(sampling: sampling, bounds: bounds, vocabSize: 32, legacyOff: false)
        for token in globalHistory { globalProcessor.push(token: token) }
        var globalLogitsValues = [Float](repeating: 0, count: 32)
        for (local, global) in head.globalIDs.enumerated() { globalLogitsValues[global] = localLogitsValues[local] }
        let globalScores = globalProcessor.scores(logits: MLXArray(globalLogitsValues).reshaped([1, 32]), step: 0)

        let idx = MLXArray(head.globalIDs.map { Int32($0) }).expandedDimensions(axis: 0)
        let globalAtLocalPositions = takeAlong(globalScores, idx, axis: -1)
        #expect(maxAbsDiff(localScores, globalAtLocalPositions) <= 1e-5)
    }
}
