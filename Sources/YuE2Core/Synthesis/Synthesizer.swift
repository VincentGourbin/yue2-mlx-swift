// Synthesizer.swift - serial multi-chunk NAR synthesis (plan §2.4, nar.py synthesize)
// Copyright 2026 Vincent Gourbin

import MLX

/// Drives `CachedNAR` over every chunk of a song, serially, releasing each chunk's cache before
/// starting the next (pitfall #8: never keep more than one chunk's KV cache alive at once).
public struct Synthesizer {
    private let model: YuE2ForCausalLM
    private let config: GenerationConfig

    public init(model: YuE2ForCausalLM, config: GenerationConfig) {
        self.model = model
        self.config = config
    }

    /// Returns `[T, 64]` fp32 latents for the whole `codec` sequence, one `CachedNAR.solve` per
    /// chunk (`NARChunk.songChunks`), concatenated in order. When there is exactly one chunk and
    /// `guidance == 1`, `reuseCache` (the semantic AR generation's own `KVCache`, already holding
    /// `prefix + codec`) is topped up to `chunk.arTokens` and reused as the NAR's AR-prefix cache
    /// instead of prefilling it a second time (O2/T-4.2, plan §7). `onProgress` reports
    /// `chunkIndex · steps + completedStepsInChunk` out of `steps · chunkCount`, matching the
    /// reference's global step counter. `onBeforeChunk(needsARPath:)` fires once per chunk,
    /// after its AR-prefix cache is settled and before its solve starts: `needsARPath == false`
    /// means the AR branch's weights are never read again in this solve (single chunk, reused
    /// cache) — the point where stage-scoped residency drops them (`ModelSession.releaseWeights
    /// (of: .ar)`) and brings the NAR branch in; `true` means `CachedNAR` will prefill the prefix
    /// itself, so both branches must be resident.
    public func synthesize(
        prefix: [Int], codec: [Int], seed: UInt64,
        steps: Int? = nil, context: Int? = nil,
        reuseCache: KVCache? = nil, guidance: Double = 1, noise: MLXArray? = nil,
        onProgress: ((Int, Int) -> Void)? = nil,
        onBeforeChunk: ((_ needsARPath: Bool) -> Void)? = nil,
        cancel: (() -> Bool)? = nil
    ) throws -> MLXArray {
        YuE2MemoryManager.configure(for: .nar)
        let resolvedSteps = steps ?? config.odeSteps
        let resolvedContext = context ?? config.context
        let chunks = try NARChunk.songChunks(
            prefix: prefix, codec: codec, seed: seed, context: resolvedContext, noise: noise
        )
        let totalSteps = resolvedSteps * chunks.count

        var outputs: [MLXArray] = []
        for (chunkIndex, chunk) in chunks.enumerated() {
            if cancel?() == true {
                throw YuE2Error.cancelled
            }
            let cacheForChunk: KVCache? =
                (chunks.count == 1 && guidance == 1)
                ? reuseCache.map { Self.topUp(cache: $0, to: chunk.arTokens, model: model) }
                : nil
            // A chunk without a ready cache prefills inside `CachedNAR`: the AR weights must
            // stay resident for it. Only the reuse path above frees the caller from that.
            onBeforeChunk?(cacheForChunk == nil)
            let engine = CachedNAR(model: model, chunk: chunk, cache: cacheForChunk)
            let chunkProgress = onProgress.map { report in
                { (completed: Int, total: Int) in report(chunkIndex * total + completed, totalSteps) }
            }
            outputs.append(try engine.solve(steps: resolvedSteps, onProgress: chunkProgress, cancel: cancel))
            YuE2MemoryManager.releaseBetweenStages()
        }
        return concatenated(outputs, axis: 0)
    }

    /// Forwards whichever suffix of `arTokens` the semantic generation's cache doesn't already
    /// hold — zero tokens when it ended naturally well before `maxTokens` (the end token's own
    /// forward pass already ran, piggybacked on sampling it — see `TokenGenerator.generate`'s
    /// pipelined lookahead), one or two when generation was truncated or the end token landed on
    /// the very last allowed step (whose lookahead forward pass never ran). Reuses `cache.offset`
    /// as the RoPE start position, so the catch-up is indistinguishable from having generated
    /// those tokens in the original loop.
    static func topUp(cache: KVCache, to arTokens: [Int], model: YuE2ForCausalLM) -> KVCache {
        let deficit = arTokens.count - cache.offset
        precondition(deficit >= 0, "O2: semantic cache.offset (\(cache.offset)) overshoots arTokens.count (\(arTokens.count))")
        if deficit > 0 {
            let missing = MLXArray(arTokens.suffix(deficit).map { Int32($0) }).reshaped([1, deficit])
            _ = model.model.forwardAR(tokens: missing, cache: cache)
            cache.evalAll()
        }
        precondition(cache.offset == arTokens.count, "O2: cache.offset (\(cache.offset)) != arTokens.count (\(arTokens.count)) after top-up")
        return cache
    }
}
