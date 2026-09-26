// TokenGenerator+CFG.swift - sampling loop, CFG, asyncEval pipelining (plan §2.3, part 2)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXProfiler

/// Wall-clock timings for one `generate()` call, mirroring `generate_tokens()`'s `timing` dict.
public struct GenerationTiming: Codable, Equatable, Sendable {
    public let seconds: Double
    public let prefillSeconds: Double
    /// Time to the first sampled token (0 if `maxTokens == 0`, which never samples one).
    public let ttftSeconds: Double
    /// Content tokens, plus one if the phase ended on its end token (`eos`).
    public let outputTokens: Int
    /// Generated content tokens (excludes the end token, which is consumed, not returned).
    public let contentTokens: Int
    public let outputTPS: Double
    public let prefixTokens: Int
    /// 1 normally, 2 under CFG (two forward branches per step).
    public let cfgBranches: Int
}

/// One `generate()` call's result: the generated **global** vocabulary ids (never the phase's
/// end token, which is consumed to stop the loop, not returned), timing, and whether the loop
/// hit `maxTokens` before the end token (`truncated`, mirroring `not eos`).
public struct GenerationResult: Sendable {
    public let tokens: [Int]
    public let timing: GenerationTiming
    public let truncated: Bool
}

extension TokenGenerator {
    /// Chunked prefill that stops at the final hidden state (pre-`lm_head`), so the caller can
    /// project it through a `RestrictedHead` instead of the full (`vocabSize`-wide) `lm_head` —
    /// same chunking rationale as `prefill(...)` (T-2.4, piège n°8: `eval` per block).
    private static func prefillLastHidden(
        model: YuE2ForCausalLM, tokens: [Int], cache: KVCache, stepSize: Int = 2048
    ) -> MLXArray {
        var lastHidden = MLXArray.zeros([1, 1, 1])
        var index = 0
        while index < tokens.count {
            YuE2GPUGate.shared.wait()
            let end = min(index + stepSize, tokens.count)
            let block = MLXArray(tokens[index..<end].map { Int32($0) }).reshaped([1, end - index])
            let hidden = model.model.forwardAR(tokens: block, cache: cache)
            lastHidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
            eval(lastHidden)
            cache.evalAll()
            index = end
        }
        return lastHidden
    }

    /// `unconditional + cfgScale · (conditional - unconditional)`, computed **before** any fp32
    /// upcast (piège n°6: CFG combines in the model's native dtype, `LogitsProcessor.scores`
    /// does the upcast next). A no-op when CFG is off (`cfgScale == 1`, no `unconditional`).
    private static func combine(conditional: MLXArray, unconditional: MLXArray?, cfgScale: Double) -> MLXArray {
        guard let unconditional, cfgScale != 1 else { return conditional }
        return unconditional + MLXArray(Float(cfgScale)) * (conditional - unconditional)
    }

    /// Reproduces `sampling.py`'s `generate_tokens()`: prefill (one or two branches under CFG),
    /// then a token-by-token loop over `sampling.maxTokens`, stopping early on the phase's end
    /// token. `lm_head` is never evaluated over the full vocabulary — every logit computed here
    /// comes from `head.logits(hidden:)` (T-2.6's O1), so the sampling loop runs entirely in
    /// `head`'s local (restricted) index space; only `tokens` in the result are mapped back to
    /// global vocabulary ids via `head.global(_:)`.
    ///
    /// Pipelining (piège-adjacent, not a correctness requirement but load-bearing for
    /// throughput): each step samples `next` *without* reading it (`sampleArray`, no `.item()`),
    /// kicks off its evaluation (`asyncEval`), builds the *next* step's forward pass on top of
    /// the still-unevaluated `next` (legal — MLX ops accept arrays regardless of evaluation
    /// state), and only then reads `next.item()` — so the blocking host read overlaps with the
    /// next step's graph construction instead of serializing in front of it.
    public static func generate(
        model: YuE2ForCausalLM, head: RestrictedHead, prefix: [Int], sampling: Sampling, seed: UInt64,
        bounds: LogitsBounds, negative: [Int]? = nil, cfgScale: Double = 1, legacyOff: Bool = false,
        cache: KVCache? = nil, onToken: ((Int) -> Void)? = nil, cancel: (() -> Bool)? = nil
    ) throws -> GenerationResult {
        guard prefix.count + sampling.maxTokens <= YuE2Token.context else {
            throw YuE2Error.invalidRequest("prefix + max_tokens exceeds context (\(YuE2Token.context))")
        }
        if cfgScale != 1, negative == nil {
            throw YuE2Error.invalidRequest("CFG requires a negative prefix")
        }
        if let negative, negative.count + sampling.maxTokens > YuE2Token.context {
            throw YuE2Error.invalidRequest("negative prefix + max_tokens exceeds context")
        }
        if cancel?() == true {
            throw YuE2Error.cancelled
        }

        let n = head.globalIDs.count
        precondition(n == bounds.allowedRange.count + 1, "head/bounds row count mismatch")
        let endLocal = n - 1
        let processor = LogitsProcessor(sampling: sampling, bounds: .local(count: n), vocabSize: n, legacyOff: legacyOff)

        let start = Date()
        let profiler = MLXProfiler.shared
        // Tokenization already happened by the time `prefix` reaches this call (Prefixes,
        // T-1.2); this back-to-back start/end only records the prompt token count so
        // `getLLMMetrics()`'s prefill tok/s is populated, not a separately-timed phase.
        profiler.startTokenization()
        profiler.endTokenization(tokenCount: prefix.count)
        profiler.startPrefill()
        let positiveCache = cache ?? KVCache()
        let conditional = head.logits(hidden: prefillLastHidden(model: model, tokens: prefix, cache: positiveCache))
            .reshaped([1, n])

        var negativeCache: KVCache?
        var unconditional: MLXArray?
        if let negative {
            let negCache = KVCache()
            unconditional = head.logits(hidden: prefillLastHidden(model: model, tokens: negative, cache: negCache))
                .reshaped([1, n])
            negativeCache = negCache
        }
        profiler.endPrefill()
        let prefillSeconds = Date().timeIntervalSince(start)

        // `nextArray` (below) is a *local* index into `head.globalIDs` — the model's
        // `embed_tokens` needs the *global* vocabulary id. Translating through this gather
        // keeps the whole step lazy (no `.item()` needed just to feed the next forward pass).
        let globalIDsArray = MLXArray(head.globalIDs.map { Int32($0) })

        var currentLogits = combine(conditional: conditional, unconditional: unconditional, cfgScale: cfgScale)
        var key = MLXRandom.key(seed)
        var history: [Int] = []
        var eos = false
        var ttft: Double = 0

        profiler.startGeneration()
        for step in 0..<sampling.maxTokens {
            if cancel?() == true || !YuE2GPUGate.shared.wait(cancel: cancel) {
                throw YuE2Error.cancelled
            }
            let scores = processor.scores(logits: currentLogits, step: step)
            var subkey: MLXArray?
            if sampling.temperature != 0 {
                let (nextKey, stepKey) = MLXRandom.split(key: key)
                key = nextKey
                subkey = stepKey
            }
            let nextArray = processor.sampleArray(scores: scores, key: subkey)
            asyncEval(nextArray)

            let isLastStep = step + 1 >= sampling.maxTokens
            var nextConditional: MLXArray?
            var nextUnconditional: MLXArray?
            if !isLastStep {
                let tokenInput = MLX.take(globalIDsArray, nextArray, axis: 0).reshaped([1, 1])
                nextConditional = head.logits(hidden: model.model.forwardAR(tokens: tokenInput, cache: positiveCache))
                    .reshaped([1, n])
                if let negativeCache {
                    nextUnconditional = head.logits(
                        hidden: model.model.forwardAR(tokens: tokenInput, cache: negativeCache)
                    ).reshaped([1, n])
                }
            }

            let local = Int(nextArray.item(UInt32.self))
            if ttft == 0 { ttft = Date().timeIntervalSince(start) }
            let global = head.global(local)
            onToken?(global)
            if local == endLocal {
                eos = true
                break
            }
            history.append(global)
            processor.push(token: local)
            if isLastStep { break }
            currentLogits = combine(conditional: nextConditional!, unconditional: nextUnconditional, cfgScale: cfgScale)
        }

        profiler.endGeneration(tokenCount: history.count)

        let seconds = Date().timeIntervalSince(start)
        let count = history.count + (eos ? 1 : 0)
        let timing = GenerationTiming(
            seconds: seconds, prefillSeconds: prefillSeconds, ttftSeconds: ttft,
            outputTokens: count, contentTokens: history.count,
            outputTPS: seconds > 0 ? Double(count) / seconds : 0,
            prefixTokens: prefix.count, cfgBranches: cfgScale == 1 ? 1 : 2)
        profiler.setTTFT(ttft)
        return GenerationResult(tokens: history, timing: timing, truncated: !eos)
    }
}
