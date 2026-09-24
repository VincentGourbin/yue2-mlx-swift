// LogitsProcessor.swift - reproduces sampling.py's distribution() exactly (plan §2.3)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXRandom

/// Which token ids are legal for a phase, and which id ends it.
public struct LogitsBounds: Sendable {
    public let allowedRange: Range<Int>
    public let endToken: Int

    public init(allowedRange: Range<Int>, endToken: Int) {
        self.allowedRange = allowedRange
        self.endToken = endToken
    }

    public static let abc = LogitsBounds(allowedRange: 0..<YuE2Token.eod, endToken: YuE2Token.abcEnd)
    public static let semantic = LogitsBounds(
        allowedRange: YuE2Token.codecOffset..<(YuE2Token.codecOffset + YuE2Token.codecSize),
        endToken: YuE2Token.musicEnd
    )

    /// Local-space bounds for a `RestrictedHead` with `n` rows (`globalIDs.count`, O1, plan
    /// §2.3): every row is allowed except the last, which stands in for the phase's end token.
    public static func local(count n: Int) -> LogitsBounds {
        LogitsBounds(allowedRange: 0..<(n - 1), endToken: n - 1)
    }
}

/// Reproduces `sampling.py`'s `distribution()`/`window_penalty()` exactly, in MLX (no host
/// round-trip). Scores stay fp32 unless `legacyOff` (cot off), which stays bf16 (piège n°6).
public final class LogitsProcessor {
    private let sampling: Sampling
    private let bounds: LogitsBounds
    private let vocabSize: Int
    private let legacyOff: Bool
    private let allowedMask: MLXArray
    private(set) var history: [Int] = []

    public init(sampling: Sampling, bounds: LogitsBounds, vocabSize: Int, legacyOff: Bool) {
        self.sampling = sampling
        self.bounds = bounds
        self.vocabSize = vocabSize
        self.legacyOff = legacyOff
        var mask = [Float](repeating: -.infinity, count: vocabSize)
        for i in bounds.allowedRange { mask[i] = 0 }
        mask[bounds.endToken] = 0
        allowedMask = MLXArray(mask).asType(legacyOff ? .bfloat16 : .float32)
    }

    /// `logits`: `[1, V]`. `step` is 0-based (piège n°7: compared against `min_tokens` as-is,
    /// and `history` never includes the end token).
    public func scores(logits: MLXArray, step: Int) -> MLXArray {
        var scores = (legacyOff ? logits : logits.asType(.float32)) + allowedMask
        if step < sampling.minTokens {
            scores[0..., bounds.endToken..<(bounds.endToken + 1)] =
                MLXArray(-Float.infinity).asType(scores.dtype)
        }
        scores = Self.windowPenalty(
            scores, window: Array(history.suffix(sampling.penaltyWindow)),
            penalty: sampling.repetitionPenalty)
        if sampling.temperature == 0 {
            return scores
        }
        if sampling.temperature != 1 {
            scores = scores / Float(sampling.temperature)
        }
        let k = min(sampling.topK, vocabSize)
        let threshold = MLX.min(top(scores, k: k, axis: -1), axis: -1, keepDims: true)
        scores = MLX.where(scores .< threshold, MLXArray(-Float.infinity).asType(scores.dtype), scores)
        if sampling.topP < 1 {
            scores = Self.topP(
                scores, threshold: Float(sampling.topP), keepFirst: legacyOff ? 3 : 1)
        }
        return scores
    }

    /// `alpha = penalty ** freq(id)`; `logit < 0 → logit·alpha`, else `logit/alpha` (frequency,
    /// not mere presence — piège n°7). Duplicate ids in `window` all resolve to the same
    /// updated value, so writing them back via advanced indexing is safe despite the repeats.
    private static func windowPenalty(_ scores: MLXArray, window: [Int], penalty: Double) -> MLXArray {
        guard penalty != 1.0, !window.isEmpty else { return scores }
        let ids = MLXArray(window.map { Int32($0) })
        let counts = MLX.sum(
            (ids.expandedDimensions(axis: 1) .== ids.expandedDimensions(axis: 0)).asType(.float32),
            axis: 1
        )
        let alpha = MLX.pow(MLXArray(Float(penalty)), counts).asType(scores.dtype)
        let values = takeAlong(scores, ids.expandedDimensions(axis: 0), axis: -1)
        let updated = MLX.where(values .< 0, values * alpha, values / alpha)
        let result = scores
        result[0..., ids] = updated
        return result
    }

    /// Descending order, cumulative-probability cutoff, always keeping `keepFirst` entries,
    /// scattered back to the original (unsorted) order.
    private static func topP(_ scores: MLXArray, threshold: Float, keepFirst: Int) -> MLXArray {
        let order = argSort(-scores, axis: -1)
        let sortedValues = takeAlong(scores, order, axis: -1)
        let probabilities = softmax(sortedValues, axis: -1)
        let removed = (probabilities.cumsum(axis: -1) - probabilities) .> threshold
        removed[0..., 0..<keepFirst] = MLXArray(false)
        let masked = MLX.where(removed, MLXArray(-Float.infinity).asType(scores.dtype), sortedValues)
        return takeAlong(masked, argSort(order, axis: -1), axis: -1)
    }

    /// `argmax` if `temperature == 0`, else a categorical draw from `scores` (unnormalized).
    /// Lazy — no `.item()` — so the caller can `asyncEval` it and build the next step's forward
    /// pass on top of it before reading its value (T-2.7's pipelining).
    public func sampleArray(scores: MLXArray, key: MLXArray? = nil) -> MLXArray {
        sampling.temperature == 0
            ? MLX.argMax(scores, axis: -1)
            : MLXRandom.categorical(scores, axis: -1, key: key)
    }

    /// `sampleArray`, materialized.
    public func sample(scores: MLXArray, key: MLXArray? = nil) -> Int {
        Int(sampleArray(scores: scores, key: key).item(UInt32.self))
    }

    /// Appends a generated (non-end) token to the repetition-penalty window.
    public func push(token: Int) {
        history.append(token)
    }
}
