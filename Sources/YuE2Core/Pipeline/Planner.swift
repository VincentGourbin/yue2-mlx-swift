// Planner.swift - ABC planning phase (plan §2.3, pipeline.py's `plan()`)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Produces a `SymbolicPlan` from a `SongRequest`: no generation at all (`cot == .off`), the
/// externally provided score re-tokenized (`request.abc != nil`), or a fresh ABC-phase
/// generation otherwise.
public struct Planner {
    private let model: YuE2ForCausalLM
    private let tokenizer: TextDecoding
    private let config: GenerationConfig

    public init(model: YuE2ForCausalLM, tokenizer: TextDecoding, config: GenerationConfig) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    }

    public func plan(
        request: SongRequest, abcSampling: Sampling? = nil,
        onToken: ((Int) -> Void)? = nil, cancel: (() -> Bool)? = nil
    ) throws -> SymbolicPlan {
        if request.cot == .off {
            let prefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer)
            return SymbolicPlan(request: request, abc: nil, abcIDs: [], prefix: prefix)
        }
        if let abc = request.abc {
            let ids = tokenizer.encode(abc)
            let prefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer, abcIDs: ids)
            return SymbolicPlan(request: request, abc: abc, abcIDs: ids, prefix: prefix)
        }

        let sampling = resolveSampling(abcSampling, default: config.abc)
        let prefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer)
        let head = RestrictedHead(lmHead: model.lmHead, bounds: .abc)
        let result = try TokenGenerator.generate(
            model: model, head: head, prefix: prefix, sampling: sampling, seed: UInt64(request.seed),
            bounds: .abc, legacyOff: false, onToken: onToken, cancel: cancel)
        let abcText = tokenizer.decode(result.tokens)
        let finalPrefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer, abcIDs: result.tokens)
        return SymbolicPlan(
            request: request, abc: abcText, abcIDs: result.tokens, prefix: finalPrefix,
            timing: result.timing, truncated: result.truncated)
    }
}
