// SemanticGenerator.swift - semantic-phase AR generation (plan §2.3, pipeline.py's `generate_semantic`)
// Copyright 2026 Vincent Gourbin

import Foundation

/// One `generateSemantic()` call's result: codec ids with `CODEC_OFFSET` subtracted back out
/// (∈ `[0, codecSize)`, ready for the NAR), timing, truncation, and the filled positive-branch
/// `KVCache` — kept around so the NAR can reuse the AR's K/V instead of recomputing them (O2,
/// Jalon 3+); `nil` only if generation never ran (it always does here, but the field stays
/// optional to mirror `TokenGenerator.generate`'s own `cache` being caller-supplied/absent).
public struct SemanticResult {
    public let plan: SymbolicPlan
    public let tokens: [Int]
    public let timing: GenerationTiming
    public let truncated: Bool
    public let cache: KVCache?

    /// For callers outside `YuE2Core` (e.g. `yue2 synthesize`, T-3.6) rebuilding a result from
    /// saved artifacts (`plan.json`/`semantic.npy`/`semantic.json`) rather than a fresh generation.
    public init(plan: SymbolicPlan, tokens: [Int], timing: GenerationTiming, truncated: Bool, cache: KVCache? = nil) {
        self.plan = plan
        self.tokens = tokens
        self.timing = timing
        self.truncated = truncated
        self.cache = cache
    }
}

/// Runs the semantic AR phase from a `SymbolicPlan`, mirroring `pipeline.py`'s
/// `generate_semantic()`: re-derives the prefix from `plan.request`/`plan.abcIDs` and refuses to
/// proceed if it disagrees with `plan.prefix` (the plan may have been loaded from disk, edited,
/// or built against a different tokenizer state), builds the CFG negative prefix when
/// `guidance != 1`, and runs `TokenGenerator.generate` with the semantic `RestrictedHead`.
public struct SemanticGenerator {
    private let model: YuE2ForCausalLM
    private let tokenizer: TextEncoding
    private let config: GenerationConfig

    public init(model: YuE2ForCausalLM, tokenizer: TextEncoding, config: GenerationConfig) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    }

    public func generateSemantic(
        plan: SymbolicPlan, sampling: Sampling? = nil,
        onToken: ((Int) -> Void)? = nil, cancel: (() -> Bool)? = nil
    ) throws -> SemanticResult {
        let expected = try Prefixes.tokenPrefixes(request: plan.request, tokenizer: tokenizer, abcIDs: plan.abcIDs)
        guard expected == plan.prefix else {
            throw YuE2Error.invalidRequest("Plan prefix disagrees with request/exact ABC IDs")
        }

        let resolvedSampling = resolveSampling(sampling, default: config.semantic)
        let negative: [Int]? =
            plan.request.guidance != 1
            ? try Prefixes.negativePrefix(request: plan.request, tokenizer: tokenizer, abcIDs: plan.abcIDs)
            : nil
        let legacyOff = plan.request.cot == .off
        let head = RestrictedHead(lmHead: model.lmHead, bounds: .semantic)
        let cache = KVCache()

        let result = try TokenGenerator.generate(
            model: model, head: head, prefix: plan.prefix, sampling: resolvedSampling,
            seed: UInt64(plan.request.seed), bounds: .semantic, negative: negative,
            cfgScale: plan.request.guidance, legacyOff: legacyOff, cache: cache,
            onToken: onToken, cancel: cancel)

        let codecTokens = result.tokens.map { $0 - YuE2Token.codecOffset }
        return SemanticResult(plan: plan, tokens: codecTokens, timing: result.timing, truncated: result.truncated, cache: cache)
    }
}
