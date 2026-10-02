// EditCommands.swift - `yue2 melody`, `yue2 vary`, `yue2 regenerate`: audio-in and song edits
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import YuE2Core

/// `yue2 melody`: a hummed / sung recording -> ABC score (`melody.abc`) the model takes as its
/// plan (`yue2 generate --abc-file melody.abc …`).
struct MelodyCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "melody",
        abstract: "Transcribe a monophonic recording (hummed or sung melody) into an ABC score the model can use as its plan."
    )

    @Option(name: .long, help: "Path to the recording (WAV, M4A, MP3…), monophonic.")
    var audio: String

    @Option(name: .long, help: "Tempo of the recording in BPM (the grid the notes are quantized to).")
    var bpm: Double = 100

    @Option(name: .long, help: "Section label written in the score (verse, chorus…).")
    var section: String = "verse"

    @Option(name: .long, help: "Lowest pitch searched, Hz (70 for a voice).")
    var minHz: Double = 70

    @Option(name: .long, help: "Highest pitch searched, Hz (1000 for a voice; a whistle needs 2500).")
    var maxHz: Double = 1000

    @Option(name: .long, help: "Octaves added to every note, negative = down (default: automatic, the melody's median brought into the singing range).")
    var octaveShift: Int?

    @Option(name: .long, help: "Output directory for melody.abc and melody.json.")
    var out: String

    func run() async throws {
        let samples = try AudioImporter.loadAudio(url: URL(fileURLWithPath: audio))
        var options = MelodyTranscriber.Options()
        options.bpm = bpm
        options.section = section
        options.minHz = minHz
        options.maxHz = maxHz
        options.octaveShift = octaveShift
        let transcription = MelodyTranscriber.transcribe(audio: samples, sampleRate: 48_000, options: options)
        let dir = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try transcription.abc.write(to: dir.appendingPathComponent("melody.abc"), atomically: true, encoding: .utf8)
        let summary: [String: Any] = [
            "bpm": transcription.bpm, "key": transcription.key, "notes": transcription.notes.count,
            "voiced_ratio": transcription.voicedRatio,
            "midi": transcription.notes.map(\.midi),
            "starts": transcription.notes.map(\.startSixteenth), "lengths": transcription.notes.map(\.lengthSixteenths),
        ]
        try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            .write(to: dir.appendingPathComponent("melody.json"))
        print(transcription.abc)
        print("MELODY key=\(transcription.key) bpm=\(Int(transcription.bpm)) notes=\(transcription.notes.count) voiced=\(String(format: "%.2f", transcription.voicedRatio)) -> \(dir.appendingPathComponent("melody.abc").path)")
        if transcription.voicedRatio < 0.3 {
            FileHandle.standardError.write(Data("warning: only \(Int(transcription.voicedRatio * 100))% of the recording carried a pitch — check the input\n".utf8))
        }
    }
}

/// Shared loading for the two edit commands.
private func loadEditPipeline(_ modelOptions: ModelOptions, reference: String?, vae: DecodeCommand.VAEVariant, vaePrecision: DecodeCommand.Precision) async throws -> (YuE2Pipeline, URL) {
    let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
    var quant = modelOptions.quant, quantHead = modelOptions.quantHead, precision = modelOptions.precision
    var vaePrec = vaePrecision.vaePrecision
    var tile: Int? = nil
    var release = false
    if let reference {
        guard let profile = YuE2ReferenceProfile.named(reference) else {
            throw YuE2Error.invalidRequest("unknown reference profile \(reference)")
        }
        profile.applyGlobalPolicy()
        quant = profile.quant; quantHead = profile.quantizeHead; precision = profile.precision
        vaePrec = profile.vaePrecision; tile = profile.vaeCoreFrames; release = profile.releaseWeightsBetweenStages
    }
    let session = try await ModelSession.load(
        modelsDir: modelsDir, quant: quant, quantizeHead: quantHead, precision: precision, residency: release ? [.ar] : .all)
    let vaeModel = try YuE2VAE.load(directory: modelsDir.appendingPathComponent(vae.model.directoryName), precision: vaePrec)
    let pipeline = YuE2Pipeline(session: session, vae: vaeModel, config: session.config, vaeCoreFrames: tile, releaseWeightsBetweenStages: release)
    return (pipeline, modelsDir)
}

private func reportEvents(_ event: PipelineEvent) {
    switch event {
    case .stage(let name): FileHandle.standardError.write(Data("\(name)\u{2026}\n".utf8))
    case .narProgress(let c, let t), .vaeProgress(let c, let t): if c == t { FileHandle.standardError.write(Data("\u{2026} \(c)/\(t)\n".utf8)) }
    default: break
    }
}

/// `yue2 vary`: a variation of a saved song (same score and tokens, SDEdit on its latents).
struct VaryCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vary",
        abstract: "Generate a variation of a saved song: same score and semantic tokens, the acoustic solve restarted from its latents (SDEdit)."
    )

    @OptionGroup var modelOptions: ModelOptions
    @Option(name: .long, help: "Directory of the song to vary (a `generate` --out).") var song: String
    @Option(name: .long, help: "0 = unchanged, 1 = fresh synthesis; 0.3-0.5 keeps the arrangement, 0.6-0.8 drifts further.") var strength: Double = 0.4
    @Option(name: .long, help: "Noise seed for the variation.") var seed: UInt64 = 1
    @Option(name: .long, help: "Reference configuration (yue2 references).") var reference: String?
    @Option(name: .long) var vae: DecodeCommand.VAEVariant = .standard
    @Option(name: .long) var vaePrecision: DecodeCommand.Precision = .fp16
    @Option(name: .long, help: "Output directory.") var out: String
    @Option(name: .long) var format: DecodeCommand.WAVFormat = .int16

    func run() async throws {
        let source = try SongResult.load(from: URL(fileURLWithPath: song))
        let (pipeline, _) = try await loadEditPipeline(modelOptions, reference: reference, vae: vae, vaePrecision: vaePrecision)
        let result = try await pipeline.vary(source, strength: strength, seed: seed, onEvent: reportEvents)
        try result.saveArtifacts(to: URL(fileURLWithPath: out), format: format.sampleFormat)
        print("varied \(source.semantic.plan.request.id): strength \(strength), seed \(seed) -> \(out) (nar \(String(format: "%.1f", result.timing.narSeconds)) s, vae \(String(format: "%.1f", result.timing.vaeSeconds)) s)")
    }
}

/// `yue2 regenerate`: keep a saved song up to a point, regenerate the rest.
struct RegenerateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "regenerate",
        abstract: "Keep a saved song up to a point in time and regenerate everything after it (new semantic tokens, acoustic solve against the kept part)."
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var samplingOverrides: SamplingOverrides
    @Option(name: .long, help: "Directory of the song (a `generate` --out).") var song: String
    @Option(name: .long, help: "Keep the song up to this many seconds, regenerate after.") var fromSeconds: Double
    @Option(name: .long, help: "Sampling seed for the regenerated part.") var seed: UInt64 = 1
    @Flag(name: .long, help: "Let the AR choose the new length (default: the source's length, cut there if the song has not ended).") var freeLength: Bool = false
    @Option(name: .long, help: "Reference configuration (yue2 references).") var reference: String?
    @Option(name: .long) var vae: DecodeCommand.VAEVariant = .standard
    @Option(name: .long) var vaePrecision: DecodeCommand.Precision = .fp16
    @Option(name: .long, help: "Output directory.") var out: String
    @Option(name: .long) var format: DecodeCommand.WAVFormat = .int16

    func run() async throws {
        let source = try SongResult.load(from: URL(fileURLWithPath: song))
        let frame = Int((fromSeconds * 25).rounded())  // 25 latent frames per second
        let (pipeline, _) = try await loadEditPipeline(modelOptions, reference: reference, vae: vae, vaePrecision: vaePrecision)
        let sampling = try samplingOverrides.semanticSampling(default: source.config.semantic)
        var count = 0
        let result = try await pipeline.regenerate(source, fromFrame: frame, seed: seed, semanticSampling: sampling, length: freeLength ? .free : .keepSource) { event in
            if case .semanticToken(let c) = event { count = c; if c % 100 == 0 { FileHandle.standardError.write(Data("semantic \(c) tokens\u{2026}\n".utf8)) } }
            reportEvents(event)
        }
        try result.saveArtifacts(to: URL(fileURLWithPath: out), format: format.sampleFormat)
        print("regenerated \(source.semantic.plan.request.id) from \(fromSeconds) s (frame \(frame)): \(result.semantic.tokens.count) frames, seed \(seed) -> \(out) (nar \(String(format: "%.1f", result.timing.narSeconds)) s, vae \(String(format: "%.1f", result.timing.vaeSeconds)) s)")
        _ = count
    }
}
