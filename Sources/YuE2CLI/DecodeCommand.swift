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

    enum Precision: String, ExpressibleByArgument {
        case fp32, fp16
        var vaePrecision: VAEPrecision { self == .fp32 ? .fp32 : .fp16 }
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

    @Option(name: .long, help: "VAE weight/compute precision (E1, plan/13-ios-backend.md).")
    var precision: Precision = .fp32

    @Option(name: .long, help: "VAE decode backend (T-6.3, E3): mlx (default), coreai-gpu, coreai-ane.")
    var vaeBackend: VAEBackendKind = .mlx

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
        let vaeDirectory = modelsDirectory.appendingPathComponent(vae.model.directoryName)

        // Never a silent Core AI -> MLX fallback for `parity vae --backend coreai-*` (strict by
        // design), but `decode` is the user-facing path: a Core AI problem here falls back to
        // MLX, loudly logged, rather than failing the whole command (plan/13-ios-backend.md).
        let footprintBeforeLoadMB = Int(FootprintSampler.current() / (1024 * 1024))
        var backend: any VAEDecoding
        do {
            backend = try await loadVAEBackend(vaeBackend, directory: vaeDirectory, precision: precision.vaePrecision)
        } catch {
            guard vaeBackend != .mlx else { throw error }
            FileHandle.standardError.write(Data("yue2 decode: \(vaeBackend) unavailable (\(error)), falling back to mlx\n".utf8))
            backend = try YuE2VAE.load(directory: vaeDirectory, precision: precision.vaePrecision)
        }
        let config = backend.config
        let footprintAfterLoadMB = Int(FootprintSampler.current() / (1024 * 1024))
        let footprint = FootprintSampler()
        footprint.start() // decode-only peaks: the load (weight-norm merge temporaries) is excluded

        let start = Date()
        let audio: MLXArray
        do {
            audio = try await decodeTiled(using: backend, z, coreFrames: coreFrames)
        } catch {
            guard vaeBackend != .mlx else { throw error }
            FileHandle.standardError.write(Data("yue2 decode: \(vaeBackend) run failed (\(error)), falling back to mlx\n".utf8))
            let mlxBackend = try YuE2VAE.load(directory: vaeDirectory, precision: precision.vaePrecision)
            audio = try await decodeTiled(using: mlxBackend, z, coreFrames: coreFrames)
        }
        eval(audio)
        let elapsed = Date().timeIntervalSince(start)
        let footprintPeakMB = Int(footprint.stop() / (1024 * 1024))
        let plan = planVAETiles(
            frames: frames, coreFrames: coreFrames, haloFrames: 16, enumerated: backend.enumeratedFrameCounts)
        let paddedTiles = plan.filter { $0.paddedFrames > 0 }.count

        let audioNCL = audio.transposed(0, 2, 1) // NLC [1, samples, 2] -> [1, 2, samples]
        try AudioExporter.exportToWAV(
            audio: audioNCL,
            url: URL(fileURLWithPath: out),
            sampleRate: config.sampleRate,
            format: format.sampleFormat
        )

        let samples = audio.dim(1)
        let seconds = Double(samples) / Double(config.sampleRate)
        let peak = YuE2MemoryManager.snapshot().peakMB
        print(
            "decoded \(frames) frames \u{2192} \(samples) samples (\(String(format: "%.1f", seconds)) s)"
                + " in \(String(format: "%.2f", elapsed)) s, peak \(peak) MB"
        )
        // MLX's peak never sees a Core AI backend's buffers; the process footprint does
        // (`phys_footprint`, what iOS jetsam judges). Reported for every backend so the two
        // numbers stay comparable across `--vae-backend` values.
        print(
            "DECODE backend=\(vaeBackend.rawValue) precision=\(precision.rawValue) core_frames=\(coreFrames)"
                + " tiles=\(plan.count) padded_tiles=\(paddedTiles) seconds=\(String(format: "%.2f", elapsed))"
                + " footprint_before_load_mb=\(footprintBeforeLoadMB) footprint_after_load_mb=\(footprintAfterLoadMB)"
                + " footprint_decode_peak_mb=\(footprintPeakMB) mlx_decode_peak_mb=\(footprint.mlxActivePeakMB)"
                + " mlx_process_peak_mb=\(peak)"
        )
    }
}
