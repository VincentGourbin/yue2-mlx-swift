// YuE2Pipeline.swift - end-to-end song generation: plan -> semantic -> synthesize -> decode
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXProfiler

/// Progress notifications from `YuE2Pipeline.generate`, for CLI/GUI callers to report.
public enum PipelineEvent: Sendable {
    case stage(String)
    case abcToken(count: Int)
    case semanticToken(count: Int)
    case narProgress(completed: Int, total: Int)
    case vaeProgress(completed: Int, total: Int)
}

/// Drives a `SongRequest` through every stage, mirroring `pipeline.py`'s `__call__`: `plan` (ABC)
/// → `generateSemantic` (codec tokens) → `synthesize` (NAR flow matching) → `decodeTiled` (VAE).
/// Configures `YuE2MemoryManager`'s cache limit per stage and releases cached (not active) GPU
/// memory between the NAR and VAE stages (pitfall #8).
public final class YuE2Pipeline {
    private let session: ModelSession
    private let vae: YuE2VAE
    private let config: GenerationConfig

    public init(session: ModelSession, vae: YuE2VAE, config: GenerationConfig) {
        self.session = session
        self.vae = vae
        self.config = config
    }

    public func generate(
        request: SongRequest,
        abcSampling: Sampling? = nil,
        semanticSampling: Sampling? = nil,
        profiling profilingSession: ProfilingSession? = nil,
        onEvent: ((PipelineEvent) -> Void)? = nil,
        cancel: (() -> Bool)? = nil
    ) throws -> SongResult {
        let e2eStart = Date()
        let profiler = MLXProfiler.shared

        YuE2MemoryManager.configure(for: .ar)
        onEvent?(.stage("plan"))
        var abcCount = 0
        let planner = Planner(model: session.model, tokenizer: session.tokenizer, config: config)
        profilingSession?.beginPhase("2. Plan (ABC)", category: .generation)
        let plan = try planner.plan(
            request: request, abcSampling: abcSampling,
            onToken: { _ in
                abcCount += 1
                onEvent?(.abcToken(count: abcCount))
            }, cancel: cancel)
        profilingSession?.endPhase("2. Plan (ABC)", category: .generation)

        onEvent?(.stage("semantic"))
        var semanticCount = 0
        let semanticGenerator = SemanticGenerator(model: session.model, tokenizer: session.tokenizer, config: config)
        profiler.startSemanticGen()
        let semantic = try semanticGenerator.generateSemantic(
            plan: plan, sampling: semanticSampling,
            onToken: { _ in
                semanticCount += 1
                onEvent?(.semanticToken(count: semanticCount))
            }, cancel: cancel)
        profiler.endSemanticGen(frameCount: semantic.tokens.count)
        profiler.setTTFT(semantic.timing.ttftSeconds)

        YuE2MemoryManager.configure(for: .nar)
        onEvent?(.stage("nar"))
        let narStart = Date()
        let synthesizer = Synthesizer(model: session.model, config: config)
        profiler.startFlowMatching()
        let latents = try synthesizer.synthesize(
            prefix: plan.prefix, codec: semantic.tokens, seed: UInt64(request.seed),
            reuseCache: semantic.cache, guidance: request.guidance,
            onProgress: { completed, total in onEvent?(.narProgress(completed: completed, total: total)) },
            cancel: cancel)
        profiler.endFlowMatching()
        let narSeconds = Date().timeIntervalSince(narStart)

        if cancel?() == true {
            throw YuE2Error.cancelled
        }
        YuE2MemoryManager.releaseBetweenStages()
        YuE2MemoryManager.configure(for: .vae)
        onEvent?(.stage("vae"))
        let vaeStart = Date()
        profiler.startCodecDecode()
        let audio = try vae.decodeTiled(latents.expandedDimensions(axis: 0)) { completed, total in
            onEvent?(.vaeProgress(completed: completed, total: total))
        }
        profiler.endCodecDecode()
        let vaeSeconds = Date().timeIntervalSince(vaeStart)
        profiler.setAudioDuration(Double(audio.dim(1)) / Double(vae.config.sampleRate))

        let timing = SongTiming(
            abc: plan.timing, semantic: semantic.timing, narSeconds: narSeconds,
            vaeSeconds: vaeSeconds, e2eSeconds: Date().timeIntervalSince(e2eStart))
        return SongResult(
            audio: audio, sampleRate: vae.config.sampleRate, semantic: semantic,
            latents: latents, config: config, timing: timing)
    }
}
