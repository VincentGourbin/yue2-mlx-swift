// Synthesizer.swift - serial multi-chunk NAR synthesis (plan §2.4, nar.py synthesize)
// Copyright 2026 Vincent Gourbin

import MLX

/// An edit of an existing song's latents, applied by `Synthesizer.synthesize(edit:)` —
/// single-chunk songs only, the same conditioning (plan, semantic tokens) as the source unless
/// the caller changed it.
public enum NAREdit {
    /// SDEdit variation: start the solve from `t·noise + (1-t)·latents` at `t = strength`
    /// (0 = the source untouched, 1 = a fresh generation), with this synthesis's own noise.
    case variation(latents: MLXArray, strength: Double)
    /// Keep the first `frames` frames of `latents` exactly (re-imposed at every step) and solve
    /// the rest against them — "regenerate from here on".
    case keep(latents: MLXArray, frames: Int)
}

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
    /// reference's global step counter. `resume`/`onStep` (`NARCheckpoint`): `onStep` fires after
    /// every completed ODE step with the live state (192 KB at 1 500 frames — persist it, never
    /// accumulate it); `resume` restarts the same schedule from a saved step, bit-for-bit the
    /// trajectory an uninterrupted solve would have taken, so a lost step costs one step, not the
    /// stage. `onBeforeChunk(needsARPath:)` fires once per chunk,
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
        resume: NARCheckpoint? = nil,
        onStep: ((NARCheckpoint) -> Void)? = nil,
        edit: NAREdit? = nil,
        cancel: (() -> Bool)? = nil
    ) throws -> MLXArray {
        YuE2MemoryManager.configure(for: .nar)
        let resolvedSteps = steps ?? config.odeSteps
        let resolvedContext = context ?? config.context
        let chunks = try NARChunk.songChunks(
            prefix: prefix, codec: codec, seed: seed, context: resolvedContext, noise: noise
        )
        let totalSteps = resolvedSteps * chunks.count
        if let edit {
            guard chunks.count == 1 else {
                throw YuE2Error.invalidRequest("NAR edits are only supported for single-chunk songs (got \(chunks.count) chunks)")
            }
            guard resume == nil else { throw YuE2Error.invalidRequest("NAR edit and resume cannot be combined") }
            let latents: MLXArray
            switch edit {
            case .variation(let l, let strength):
                guard (0...1).contains(strength) else { throw YuE2Error.invalidRequest("variation strength must be in 0...1") }
                latents = l
            case .keep(let l, let frames):
                guard frames >= 0, frames <= l.dim(0) else { throw YuE2Error.invalidRequest("keep frames out of range") }
                latents = l
            }
            guard latents.shape == chunks[0].noise.shape else {
                throw YuE2Error.invalidRequest("edit latents \(latents.shape) do not match this song's latents \(chunks[0].noise.shape)")
            }
        }
        if let resume {
            // Single-chunk songs only (every iPhone-length song): a multi-chunk resume would also
            // need the earlier chunks' latents, which this API does not carry.
            guard chunks.count == 1, resume.chunkIndex == 0 else {
                throw YuE2Error.invalidRequest("NAR resume is only supported for single-chunk songs (got \(chunks.count) chunks, checkpoint chunk \(resume.chunkIndex))")
            }
            guard resume.steps == resolvedSteps, (0..<resolvedSteps).contains(resume.step) else {
                throw YuE2Error.invalidRequest("NAR checkpoint (step \(resume.step)/\(resume.steps)) does not match this solve (\(resolvedSteps) steps)")
            }
            guard resume.state.shape == chunks[0].noise.shape else {
                throw YuE2Error.invalidRequest("NAR checkpoint state \(resume.state.shape) does not match this song's latents \(chunks[0].noise.shape)")
            }
        }

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
            let stepCallback: ((Int, MLXArray) -> Void)? = onStep.map { report in
                { completed, state in
                    report(NARCheckpoint(chunkIndex: chunkIndex, step: completed, steps: resolvedSteps, state: state))
                }
            }
            var initialState = resume?.state
            var startStep = resume?.step ?? 0
            var keep: (frames: Int, latents: MLXArray)?
            switch edit {
            case .variation(let latents, let strength):
                // t = 1 − s/steps ⇒ the step whose time is closest to `strength`.
                startStep = min(resolvedSteps, max(0, Int(((1 - strength) * Double(resolvedSteps)).rounded())))
                let t = Float(1.0 - Double(startStep) / Double(resolvedSteps))
                let noise = chunk.noise.asType(.float32)
                initialState = noise * t + latents.asType(.float32) * (1 - t)
            case .keep(let latents, let frames):
                keep = (frames, latents)
            case nil:
                break
            }
            if startStep >= resolvedSteps, let initialState {
                // strength 0: nothing to solve, the source is the result.
                outputs.append(initialState.asType(.float32))
                continue
            }
            outputs.append(try engine.solve(
                steps: resolvedSteps,
                initialState: initialState, startStep: startStep,
                onProgress: chunkProgress, cancel: cancel, onStep: stepCallback, keep: keep))
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
