// YuE2Pipeline+Edit.swift - edits of an existing song: variation (SDEdit) and "regenerate from here"
// Copyright 2026 Vincent Gourbin
//
// Both keep the song's plan (score) and work on what the pipeline already produced
// (`SongResult.latents`, `.semantic.tokens`): no audio understanding is needed, the model's own
// artifacts are the input. Single-chunk songs (every PocketAnthem-length song).

import Foundation
import MLX

extension YuE2Pipeline {
    /// A variation of `song`: same score, same semantic tokens, the NAR solve restarted from
    /// `strength·noise + (1-strength)·latents` with a new noise draw (`seed`). `strength` 0 returns
    /// the source latents unchanged, 1 is a fresh synthesis; 0.3-0.5 keeps the arrangement and
    /// changes the texture, 0.6-0.8 drifts further. Decodes with this pipeline's VAE settings.
    public func vary(
        _ song: SongResult, strength: Double, seed: UInt64,
        onEvent: ((PipelineEvent) -> Void)? = nil, cancel: (() -> Bool)? = nil
    ) async throws -> SongResult {
        try await resynthesize(
            song, semantic: song.semantic, seed: seed,
            edit: .variation(latents: song.latents, strength: strength),
            onEvent: onEvent, cancel: cancel)
    }

    /// How long the regenerated song may be.
    public enum RegenerateLength: Sendable {
        /// The source's length (the AR is cut there if it has not ended the song): the default,
        /// what "regenerate from here" means to a listener.
        case keepSource
        /// The AR decides, within the sampling's `maxTokens` — a song re-conditioned on its first
        /// half readily grows to two or three times its length.
        case free
        /// At most the source's length plus `frames` (25 per second): the AR may stop earlier.
        /// For a score lengthened after its end (`score:`); single NAR chunk, so the song stays
        /// under the context (≈ 860 s with a 3 000-token prefix).
        case extended(frames: Int)
    }

    /// Keeps `song` up to `frame` (one latent frame = 40 ms) exactly and regenerates the rest:
    /// the semantic tokens after `frame` are re-sampled by the AR model with the kept tokens as
    /// its past (`seed`), then the NAR solves the free frames against the kept latents. The
    /// seam at `frame` is continuous by construction (same conditioning, inpainting mask).
    ///
    /// `score` replaces the song's ABC score for the regenerated part (notes written into a
    /// silent outro, bars added after the end). The kept tokens and latents stay exactly as they
    /// are, but the whole song is then conditioned on the new score: change only what comes after
    /// `frame`. The score's tokens are kept as they were up to the first changed line, so an
    /// unchanged beginning keeps the planner's own ids.
    public func regenerate(
        _ song: SongResult, fromFrame frame: Int, seed: UInt64, score: String? = nil,
        semanticSampling: Sampling? = nil, length: RegenerateLength = .keepSource,
        onEvent: ((PipelineEvent) -> Void)? = nil, cancel: (() -> Bool)? = nil
    ) async throws -> SongResult {
        let total = song.semantic.tokens.count
        guard frame >= 0, frame < total else {
            throw YuE2Error.invalidRequest("fromFrame must be in 0..<\(total) (the song has \(total) frames)")
        }
        var sampling = semanticSampling ?? config.semantic
        switch length {
        case .keepSource:
            sampling.maxTokens = min(sampling.maxTokens, total)
            sampling.minTokens = min(sampling.minTokens, total)
        case .extended(let frames):
            guard frames > 0 else { throw YuE2Error.invalidRequest("extended(frames:) needs frames > 0") }
            sampling.maxTokens = min(sampling.maxTokens, total + frames)
            sampling.minTokens = min(sampling.minTokens, total)
        case .free:
            break
        }
        try session.loadWeights(of: .ar)
        YuE2MemoryManager.configure(for: .ar)
        onEvent?(.stage("semantic"))
        let generator = SemanticGenerator(model: session.model, tokenizer: session.tokenizer, config: config)
        var request = song.semantic.plan.request
        request.seed = Int(truncatingIfNeeded: seed)
        var plan = SymbolicPlan(
            request: request, abc: song.semantic.plan.abc, abcIDs: song.semantic.plan.abcIDs,
            prefix: song.semantic.plan.prefix)
        if let score { plan = try plan.replacingScore(score, tokenizer: session.tokenizer) }
        // Edits solve one NAR chunk: refuse a length the context cannot hold before paying the AR.
        guard plan.prefix.count + sampling.maxTokens + 1 <= config.context else {
            throw YuE2Error.invalidRequest("prefix (\(plan.prefix.count)) + up to \(sampling.maxTokens) frames exceeds the \(config.context)-token context of a single-chunk edit")
        }
        var count = 0
        let semantic = try generator.generateSemantic(
            plan: plan, sampling: sampling, continuation: Array(song.semantic.tokens.prefix(frame)),
            onToken: { _ in count += 1; onEvent?(.semanticToken(count: count)) }, cancel: cancel)
        // The kept latents must line up with the new token sequence: pad or crop to its length.
        let kept = song.latents[0..<min(frame, song.latents.dim(0)), 0...]
        let padded: MLXArray
        if semantic.tokens.count > kept.dim(0) {
            padded = concatenated([kept, MLXArray.zeros([semantic.tokens.count - kept.dim(0), kept.dim(1)], dtype: kept.dtype)], axis: 0)
        } else {
            padded = kept[0..<semantic.tokens.count, 0...]
        }
        return try await resynthesize(
            song, semantic: semantic, seed: seed,
            edit: .keep(latents: padded, frames: min(frame, semantic.tokens.count)),
            onEvent: onEvent, cancel: cancel)
    }

    private func resynthesize(
        _ song: SongResult, semantic: SemanticResult, seed: UInt64, edit: NAREdit,
        onEvent: ((PipelineEvent) -> Void)?, cancel: (() -> Bool)?
    ) async throws -> SongResult {
        let start = Date()
        try session.loadWeights(of: .nar)
        YuE2MemoryManager.configure(for: .nar)
        onEvent?(.stage("nar"))
        let narStart = Date()
        let synthesizer = Synthesizer(model: session.model, config: config)
        let latents = try synthesizer.synthesize(
            prefix: semantic.plan.prefix, codec: semantic.tokens, seed: seed,
            reuseCache: semantic.cache, guidance: semantic.plan.request.guidance,
            onProgress: { completed, total in onEvent?(.narProgress(completed: completed, total: total)) },
            onBeforeChunk: releaseWeightsBetweenStages ? { [session] (needsARPath: Bool) in
                if !needsARPath { session.releaseWeights(of: .ar); YuE2MemoryManager.releaseBetweenStages() }
            } : nil,
            edit: edit, cancel: cancel)
        let narSeconds = Date().timeIntervalSince(narStart)
        if cancel?() == true { throw YuE2Error.cancelled }
        if releaseWeightsBetweenStages { session.releaseAllWeights() }
        YuE2MemoryManager.releaseBetweenStages()
        YuE2MemoryManager.configure(for: .vae)
        onEvent?(.stage("vae"))
        let vaeStart = Date()
        let audio = try await decodeTiled(using: vae, latents.expandedDimensions(axis: 0), coreFrames: vaeCoreFrames) { completed, total in
            onEvent?(.vaeProgress(completed: completed, total: total))
        }
        let vaeSeconds = Date().timeIntervalSince(vaeStart)
        let timing = SongTiming(
            abc: nil, semantic: semantic.timing, narSeconds: narSeconds, vaeSeconds: vaeSeconds,
            e2eSeconds: Date().timeIntervalSince(start))
        return SongResult(
            audio: audio, sampleRate: vae.config.sampleRate, semantic: semantic, latents: latents,
            config: config, timing: timing)
    }
}
