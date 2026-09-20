// DecodeCommand.swift - decode a [T, 64] latent .npy into a stereo WAV file (plan §1.11)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import YuE2Core

struct DecodeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decode",
        abstract: "Decode a [T, 64] latent .npy into a stereo WAV file."
    )

    enum VAEVariant: String, ExpressibleByArgument {
        case standard, legacy
        var model: YuE2Model { self == .standard ? .vae : .vaeLegacy }
    }

    enum WAVFormat: String, ExpressibleByArgument {
        case int16, float32
        var sampleFormat: AudioExporter.SampleFormat { self == .int16 ? .int16 : .float32 }
    }

    @Option(name: .long, help: "Path to the input latent .npy ([T, 64], float32).")
    var latents: String

    @Option(name: .long, help: "Path to the output WAV file.")
    var out: String

    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Which VAE checkpoint to decode with.")
    var vae: VAEVariant = .standard

    @Option(name: .long, help: "Tile size, in latent frames, for the tiled decoder.")
    var coreFrames: Int = 1024

    @Option(name: .long, help: "WAV sample format.")
    var format: WAVFormat = .int16

    func run() async throws {
        let modelsDirectory = try resolveModelsDir(modelsDir)
        let (shape, data) = try NumpyIO.readFloat32(URL(fileURLWithPath: latents))
        guard shape.count == 2 else {
            throw YuE2Error.invalidRequest("expected a [T, 64] latent .npy, got shape \(shape)")
        }
        let frames = shape[0]
        let latentDim = shape[1]
        let z = MLXArray(data).reshaped([1, frames, latentDim])

        YuE2MemoryManager.configure(for: .vae)
        let model = try YuE2VAE.load(directory: modelsDirectory.appendingPathComponent(vae.model.directoryName))

        let start = Date()
        let audio = try model.decodeTiled(z, coreFrames: coreFrames)
        eval(audio)
        let elapsed = Date().timeIntervalSince(start)

        let audioNCL = audio.transposed(0, 2, 1) // NLC [1, samples, 2] -> [1, 2, samples]
        try AudioExporter.exportToWAV(
            audio: audioNCL,
            url: URL(fileURLWithPath: out),
            sampleRate: model.config.sampleRate,
            format: format.sampleFormat
        )

        let samples = audio.dim(1)
        let seconds = Double(samples) / Double(model.config.sampleRate)
        let peak = YuE2MemoryManager.snapshot().peakMB
        print(
            "decoded \(frames) frames \u{2192} \(samples) samples (\(String(format: "%.1f", seconds)) s)"
                + " in \(String(format: "%.2f", elapsed)) s, peak \(peak) MB"
        )
    }
}
