// EncodeCommand.swift - encode an audio file into VAE latents, with an optional round-trip (T-5.2)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import YuE2Core

struct EncodeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "encode",
        abstract: "Encode an audio file (MP3, WAV, ...) into VAE latents."
    )

    @Option(name: .long, help: "Path to the input audio file.")
    var audio: String

    @Option(name: .long, help: "Output directory for latent.npy (and roundtrip.wav with --roundtrip).")
    var out: String

    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Which VAE checkpoint to use.")
    var vae: DecodeCommand.VAEVariant = .standard

    @Flag(name: .long, help: "Also decode the resulting latent back to audio, for a quality round-trip check.")
    var roundtrip: Bool = false

    @Option(name: .long, help: "WAV sample format for --roundtrip.")
    var format: DecodeCommand.WAVFormat = .int16

    func run() async throws {
        let modelsDirectory = try resolveModelsDir(modelsDir)
        let audioURL = URL(fileURLWithPath: audio)
        let vaeDirectory = modelsDirectory.appendingPathComponent(vae.model.directoryName)
        let outDirectory = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)

        let waveform = try AudioImporter.loadAudio(url: audioURL)
        let sourceSamples = waveform.dim(1)
        let sourceSeconds = Double(sourceSamples) / 48_000.0

        let encoder = try YuE2VAEEncoder.load(directory: vaeDirectory)
        let start = Date()
        let latent = try encoder.encode(waveform).squeezed(axis: 0) // [1,T,64] -> [T,64]
        eval(latent)
        let elapsed = Date().timeIntervalSince(start)

        try NumpyIO.writeFloat32(
            latent.asArray(Float.self), shape: latent.shape, url: outDirectory.appendingPathComponent("latent.npy"))

        print(
            "encoded \(audioURL.lastPathComponent): \(sourceSamples) samples"
                + " (\(String(format: "%.1f", sourceSeconds)) s) \u{2192} \(latent.dim(0)) frames latent"
                + " in \(String(format: "%.2f", elapsed)) s"
        )

        guard roundtrip else { return }

        let decoder = try YuE2VAE.load(directory: vaeDirectory)
        let decodeStart = Date()
        let decoded = try decoder.decodeTiled(latent.expandedDimensions(axis: 0)) // [T,64] -> [1,T,64]
        eval(decoded)
        let decodeElapsed = Date().timeIntervalSince(decodeStart)

        let roundtripURL = outDirectory.appendingPathComponent("roundtrip.wav")
        try AudioExporter.exportToWAV(
            audio: decoded.transposed(0, 2, 1), // NLC [1,S,2] -> [1,2,S]
            url: roundtripURL, sampleRate: decoder.config.sampleRate, format: format.sampleFormat
        )
        let outSeconds = Double(decoded.dim(1)) / Double(decoder.config.sampleRate)
        print("roundtrip: \(roundtripURL.path) (\(String(format: "%.1f", outSeconds)) s) in \(String(format: "%.2f", decodeElapsed)) s")
    }
}
