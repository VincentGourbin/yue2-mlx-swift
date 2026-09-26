// TokenGenerator.swift - chunked AR prefill (part 1: T-2.4; sampling loop arrives T-2.7)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Result of prefilling a prompt: the logits to sample the next token from, the cache to
/// continue generation with, and how long it took.
public struct PrefillResult {
    public let lastLogits: MLXArray
    public let cache: KVCache
    public let seconds: Double
}

/// Chunked AR prefill: splits a long prompt into blocks of at most `stepSize` tokens so peak
/// memory stays bounded by one block's activations instead of the whole prompt's (attention
/// scores grow with T², `lm_head` output with T·vocab). `eval` runs after each block to
/// materialize the cache and free the block's intermediate graph (piège n°8).
public enum TokenGenerator {
    /// `keepLastOnly`: when true (the common case — a prompt is prefilled only to sample the
    /// next token), every block computes `lm_head` for its last position only. When false, every
    /// block computes full per-token logits (used to compare chunked vs. full-sequence logits).
    /// Either way, `PrefillResult.lastLogits` ends up holding just the final token's logits.
    public static func prefill(
        model: YuE2ForCausalLM, tokens: [Int], cache: KVCache = KVCache(),
        stepSize: Int = 2048, keepLastOnly: Bool = true
    ) -> PrefillResult {
        let start = Date()
        var lastLogits = MLXArray.zeros([1, 1, 1])
        var index = 0
        while index < tokens.count {
            YuE2GPUGate.shared.wait()
            let end = min(index + stepSize, tokens.count)
            let block = MLXArray(tokens[index..<end].map { Int32($0) }).reshaped([1, end - index])
            let logits = model.forwardAR(tokens: block, cache: cache, keepLast: keepLastOnly)
            lastLogits = keepLastOnly ? logits : logits[0..., (logits.dim(1) - 1)..., 0...]
            eval(lastLogits)
            cache.evalAll()
            index = end
        }
        return PrefillResult(lastLogits: lastLogits, cache: cache, seconds: Date().timeIntervalSince(start))
    }
}
