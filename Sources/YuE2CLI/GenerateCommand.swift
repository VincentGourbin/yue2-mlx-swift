// GenerateCommand.swift - `yue2 generate`: end-to-end song generation (plan §2.4/§8.3, T-3.6)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLXProfiler
import YuE2Core

/// Applies `--ode-steps` on top of a loaded `GenerationConfig`, if given. Shared by
/// `GenerateCommand` and `SynthesizeCommand`.
func resolveConfig(_ base: GenerationConfig, odeSteps: Int?) throws -> GenerationConfig {
    guard let odeSteps else { return base }
    return try GenerationConfig(
        abc: base.abc, semantic: base.semantic, odeSteps: odeSteps,
        odeMethod: base.odeMethod, context: base.context, version: base.version)
}

struct GenerateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate a complete song end to end: plan, semantic tokens, NAR synthesis, VAE decode."
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var requestOptions: RequestOptions
    @OptionGroup var samplingOverrides: SamplingOverrides

    @Option(name: .long, help: "Which VAE checkpoint to decode with.")
    var vae: DecodeCommand.VAEVariant = .standard

    @Option(name: .long, help: "Flow-matching ODE steps (overrides the checkpoint default).")
    var odeSteps: Int?

    @Option(name: .long, help: "Output directory for the song's artifacts.")
    var out: String

    @Option(name: .long, help: "WAV sample format.")
    var format: DecodeCommand.WAVFormat = .int16

    @Flag(name: .long, help: "Profile the run: phase report, TTS metrics, and trace.json in --out.")
    var profile: Bool = false

    @Option(name: .long, help: "VAE weight/compute precision (E1): fp32 (default) or fp16.")
    var vaePrecision: DecodeCommand.Precision = .fp32

    @Option(name: .long, help: "VAE tile size in latent frames (default: 1024, or 256 under the mobile memory profile).")
    var vaeCoreFrames: Int?

    @Flag(name: .long, help: "Stage-scoped weight residency: drop the AR branch once the NAR prefix cache is ready, the whole LM before the VAE (default under the mobile memory profile).")
    var releaseWeightsBetweenStages: Bool = false

    @Option(name: .long, help: "VAE decode backend (T-6.3): mlx (default), coreai-gpu, coreai-ane — Core AI falls back to MLX, loudly, on any failure.")
    var vaeBackend: VAEBackendKind = .mlx

    func run() async throws {
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let request = try requestOptions.makeRequest()

        let profilingSession: ProfilingSession? = profile ? ProfilingSession(config: .singleRun) : nil
        if let profilingSession {
            profilingSession.title = "yue2 generate"
            profilingSession.metadata = ["model": "YuE2-3B", "cot": request.cot.rawValue, "seed": "\(request.seed)"]
            MLXProfiler.shared.enable()
            MLXProfiler.shared.activeSession = profilingSession
        }

        YuE2MemoryManager.configure(for: .ar)
        profilingSession?.beginPhase("1. Model Loading", category: .modelLoad)
        let releaseWeights = releaseWeightsBetweenStages || YuE2MemoryManager.profile == .mobile
        let session = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead,
            precision: modelOptions.precision, residency: releaseWeights ? [.ar] : .all)
        let vaeDirectory = modelsDir.appendingPathComponent(vae.model.directoryName)
        var vaeModel: any VAEDecoding
        do {
            vaeModel = try await loadVAEBackend(vaeBackend, directory: vaeDirectory, precision: vaePrecision.vaePrecision)
        } catch {
            guard vaeBackend != .mlx else { throw error }
            FileHandle.standardError.write(Data("yue2 generate: \(vaeBackend) unavailable (\(error)), falling back to mlx\n".utf8))
            vaeModel = try YuE2VAE.load(directory: vaeDirectory, precision: vaePrecision.vaePrecision)
        }
        profilingSession?.endPhase("1. Model Loading", category: .modelLoad)
        let config = try resolveConfig(session.config, odeSteps: odeSteps)
        let abcSampling = try samplingOverrides.abcSampling(default: config.abc)
        let semanticSampling = try samplingOverrides.semanticSampling(default: config.semantic)

        var lastSemanticCount = 0
        let pipeline = YuE2Pipeline(
            session: session, vae: vaeModel, config: config,
            vaeCoreFrames: vaeCoreFrames, releaseWeightsBetweenStages: releaseWeights)
        let result = try await pipeline.generate(
            request: request, abcSampling: abcSampling, semanticSampling: semanticSampling,
            profiling: profilingSession
        ) { event in
            switch event {
            case .stage(let name):
                FileHandle.standardError.write(Data("\(name)\u{2026}\n".utf8))
            case .abcToken:
                break
            case .semanticToken(let count):
                if count > 0, count - lastSemanticCount >= 100 {
                    lastSemanticCount = count
                    FileHandle.standardError.write(Data("semantic \(count) tokens\u{2026}\n".utf8))
                }
            case .narProgress(let completed, let total), .vaeProgress(let completed, let total):
                if completed == total {
                    FileHandle.standardError.write(Data("\u{2026} \(completed)/\(total)\n".utf8))
                }
            }
        }

        try result.saveArtifacts(to: URL(fileURLWithPath: out), format: format.sampleFormat)

        if let profilingSession {
            profilingSession.finish()
            print(profilingSession.generateReport())
            print(MLXProfiler.shared.getTTSMetrics().summary)
            try ChromeTraceExporter.export(session: profilingSession)
                .write(to: URL(fileURLWithPath: out).appendingPathComponent("trace.json"))
        }

        let seconds = Double(result.audio.dim(1)) / Double(result.sampleRate)
        print(
            "generated \(request.id): \(String(format: "%.1f", seconds)) s audio"
                + " in \(String(format: "%.1f", result.timing.e2eSeconds)) s"
                + " (abc \(String(format: "%.1f", result.timing.abc?.seconds ?? 0)) s,"
                + " semantic \(String(format: "%.1f", result.timing.semantic.seconds)) s,"
                + " nar \(String(format: "%.1f", result.timing.narSeconds)) s,"
                + " vae \(String(format: "%.1f", result.timing.vaeSeconds)) s)"
        )
    }
}
