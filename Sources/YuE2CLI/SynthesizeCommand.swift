// SynthesizeCommand.swift - `yue2 synthesize`: resume from a saved plan + semantic.npy (T-3.6)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

private struct SemanticMetadata: Codable {
    let timing: GenerationTiming
    let truncated: Bool
}

struct SynthesizeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "synthesize",
        abstract: "Resume from a saved plan and semantic.npy: NAR synthesis + VAE decode."
    )

    @OptionGroup var modelOptions: ModelOptions

    @Option(name: .long, help: "Directory containing the saved plan (yue2 plan's --out).")
    var plan: String

    @Option(name: .long, help: "Directory containing semantic.npy/semantic.json (yue2 semantic's --out).")
    var semantic: String

    @Option(name: .long, help: "Which VAE checkpoint to decode with.")
    var vae: DecodeCommand.VAEVariant = .standard

    @Option(name: .long, help: "Flow-matching ODE steps (overrides the checkpoint default).")
    var odeSteps: Int?

    @Option(name: .long, help: "Output directory for the song's artifacts.")
    var out: String

    @Option(name: .long, help: "Directory for the per-step NAR checkpoint (nar-checkpoint.json + nar-state.npy), written after every ODE step and cleared on success.")
    var checkpointDir: String?

    @Flag(name: .long, help: "Resume the NAR solve from the checkpoint in --checkpoint-dir (same plan, semantic tokens, seed and steps).")
    var resume: Bool = false

    @Option(name: .long, help: "WAV sample format.")
    var format: DecodeCommand.WAVFormat = .int16

    func run() async throws {
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let savedPlan = try SymbolicPlan.load(from: URL(fileURLWithPath: plan))
        let semanticDir = URL(fileURLWithPath: semantic)
        let semanticTokens = try NumpyIO.readInt32(semanticDir.appendingPathComponent("semantic.npy")).map(Int.init)
        let semanticMeta = try JSONDecoder().decode(
            SemanticMetadata.self, from: Data(contentsOf: semanticDir.appendingPathComponent("semantic.json")))

        YuE2MemoryManager.configure(for: .nar)
        let session = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead,
            precision: modelOptions.precision)
        let vaeModel = try YuE2VAE.load(directory: modelsDir.appendingPathComponent(vae.model.directoryName))
        let config = try resolveConfig(session.config, odeSteps: odeSteps)

        let synthesizer = Synthesizer(model: session.model, config: config)
        let checkpointURL = checkpointDir.map { URL(fileURLWithPath: $0) }
        var resumeFrom: NARCheckpoint?
        if resume {
            guard let checkpointURL else { throw YuE2Error.invalidRequest("--resume needs --checkpoint-dir") }
            resumeFrom = try NARCheckpoint.load(from: checkpointURL)
            guard let resumeFrom else { throw YuE2Error.missingFile("no NAR checkpoint in \(checkpointURL.path)") }
            FileHandle.standardError.write(Data("resuming NAR from step \(resumeFrom.step)/\(resumeFrom.steps)\n".utf8))
        }
        let narStart = Date()
        let latents = try synthesizer.synthesize(
            prefix: savedPlan.prefix, codec: semanticTokens, seed: UInt64(savedPlan.request.seed),
            resume: resumeFrom,
            onStep: checkpointURL.map { url in { checkpoint in try? checkpoint.save(to: url) } })
        let narSeconds = Date().timeIntervalSince(narStart)
        if let checkpointURL { NARCheckpoint.clear(in: checkpointURL) }

        YuE2MemoryManager.releaseBetweenStages()
        YuE2MemoryManager.configure(for: .vae)
        let vaeStart = Date()
        let audio = try vaeModel.decodeTiled(latents.expandedDimensions(axis: 0))
        let vaeSeconds = Date().timeIntervalSince(vaeStart)

        let semanticResult = SemanticResult(
            plan: savedPlan, tokens: semanticTokens, timing: semanticMeta.timing, truncated: semanticMeta.truncated)
        let timing = SongTiming(
            abc: savedPlan.timing, semantic: semanticMeta.timing, narSeconds: narSeconds,
            vaeSeconds: vaeSeconds, e2eSeconds: narSeconds + vaeSeconds)
        let result = SongResult(
            audio: audio, sampleRate: vaeModel.config.sampleRate, semantic: semanticResult,
            latents: latents, config: config, timing: timing)
        try result.saveArtifacts(to: URL(fileURLWithPath: out), format: format.sampleFormat)

        let seconds = Double(audio.dim(1)) / Double(vaeModel.config.sampleRate)
        print(
            "synthesized \(savedPlan.request.id): \(String(format: "%.1f", seconds)) s audio"
                + " (nar \(String(format: "%.1f", narSeconds)) s, vae \(String(format: "%.1f", vaeSeconds)) s)"
        )
    }
}
