// GenerationSmokeTests.swift - end-to-end smoke checks against real checkpoints (tier 2, T-1.11)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXRandom
import Testing
@testable import YuE2Core

private func realVAEReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("YuE2-Vae/model.safetensors").path)
}

private func songPipelineReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("YuE2-3B/model.safetensors").path)
}

@Suite("GenerationSmoke", .enabled(if: realVAEReady()))
struct GenerationSmokeTests {
    /// `reference/yue/examples/song.json`, inlined rather than read from disk: that directory is
    /// a gitignored Python-tooling clone (T-1.1), not guaranteed present wherever `$YUE2_MODELS_DIR`
    /// is (mirrors why `RealLMParityTests` stores prefixes in the fixture instead of re-deriving
    /// them from `reference/yue/examples/score.abc` at Swift test time).
    private static let songRequest = try! SongRequest(
        style: "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light "
            + "drums, lyrical memorable melody, unhurried phrasing, 88 BPM",
        lyrics: "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\nFold the night and leave "
            + "it here\nMorning has a sky to clear\n\n[Chorus]\nLet the day come into view\nEvery road begins "
            + "with you\nHold a little room for light\nWe will sing beyond the night",
        cot: .full, seed: 831_001, id: "city_lights")

    @Test(.enabled(if: songPipelineReady()))
    func generatesAShortSongEndToEnd() async throws {
        let dir = try #require(modelsDir())
        let session = try await ModelSession.load(modelsDir: dir)
        let vae = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"))

        // Same short-run knobs as the fiche's manual validation (`--abc-max-tokens 64
        // --semantic-max-tokens 100 --semantic-min-tokens 0 --ode-steps 4`), keeping every other
        // sampling field at the checkpoint's own tuned defaults.
        let abcSampling = try Sampling(
            temperature: session.config.abc.temperature, topP: session.config.abc.topP,
            topK: session.config.abc.topK, repetitionPenalty: session.config.abc.repetitionPenalty,
            penaltyWindow: session.config.abc.penaltyWindow, minTokens: 0, maxTokens: 64)
        let semanticSampling = try Sampling(
            temperature: session.config.semantic.temperature, topP: session.config.semantic.topP,
            topK: session.config.semantic.topK, repetitionPenalty: session.config.semantic.repetitionPenalty,
            penaltyWindow: session.config.semantic.penaltyWindow, minTokens: 0, maxTokens: 100)
        let config = try GenerationConfig(odeSteps: 4)

        let pipeline = YuE2Pipeline(session: session, vae: vae, config: config)
        let result = try await pipeline.generate(
            request: Self.songRequest, abcSampling: abcSampling, semanticSampling: semanticSampling)

        let outDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: outDir) }
        try result.saveArtifacts(to: outDir)

        let expectedSamples = vae.naturalOutputLength(frames: result.latents.dim(0))
        #expect(result.audio.dim(1) == expectedSamples)
        #expect(MLX.all(MLX.isFinite(result.audio)).item(Bool.self))
        #expect(FileManager.default.fileExists(atPath: outDir.appendingPathComponent("audio.wav").path))

        let samples = result.audio.asArray(Float.self)
        let rms = (samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count)).squareRoot()
        #expect(rms >= 1e-3 && rms <= 0.7)

        // 100 tokens is far short of a natural [Chorus]-length ending for this lyric.
        #expect(result.truncated.semantic)

        let resultData = try Data(contentsOf: outDir.appendingPathComponent("result.json"))
        let manifest = try #require(try JSONSerialization.jsonObject(with: resultData) as? [String: Any])
        #expect(manifest["status"] as? String == "complete")
        #expect(manifest["sample_rate"] as? Int == vae.config.sampleRate)
    }

    @Test func decodesASeededLatentToAFiniteClampedWAV() throws {
        let dir = try #require(modelsDir())
        let vae = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"))

        let frames = 50
        let key = MLXRandom.key(20_260_917)
        let z = MLXRandom.normal([1, frames, 64], key: key)
        let audio = try vae.decodeTiled(z, coreFrames: 1024)

        let expectedSamples = vae.naturalOutputLength(frames: frames)
        #expect(audio.dim(1) == expectedSamples)
        #expect(MLX.all(MLX.isFinite(audio)).item(Bool.self))

        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: outURL) }
        try AudioExporter.exportToWAV(
            audio: audio.transposed(0, 2, 1),
            url: outURL,
            sampleRate: vae.config.sampleRate,
            format: .float32
        )

        let (writtenShape, writtenSamples) = try readFloat32WAV(outURL)
        #expect(writtenShape == [2, expectedSamples])
        #expect(writtenSamples.allSatisfy { $0.isFinite && abs($0) <= 1 })
    }
}

/// Minimal float32 WAV reader (this test's own round-trip check, not a general-purpose parser).
private func readFloat32WAV(_ url: URL) throws -> (shape: [Int], samples: [Float]) {
    let data = try Data(contentsOf: url)
    let channels = Int(data[22]) | (Int(data[23]) << 8)
    let dataStart = 44
    let sampleCount = (data.count - dataStart) / 4
    let samples: [Float] = data.subdata(in: dataStart..<data.count).withUnsafeBytes { raw in
        raw.bindMemory(to: UInt32.self).map { Float(bitPattern: UInt32(littleEndian: $0)) }
    }
    return ([channels, sampleCount / channels], samples)
}
