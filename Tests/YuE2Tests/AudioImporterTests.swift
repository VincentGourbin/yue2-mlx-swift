// AudioImporterTests.swift - synthetic round-trip + resampling + error paths (T-5.2)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

/// A short seeded sine-ish stereo signal, `[2, frameCount]` NCL (what `AudioExporter` expects).
private func syntheticWave(frameCount: Int, sampleRate: Double) -> MLXArray {
    var left = [Float](repeating: 0, count: frameCount)
    var right = [Float](repeating: 0, count: frameCount)
    for i in 0..<frameCount {
        let t = Float(i) / Float(sampleRate)
        left[i] = sin(2 * .pi * 440 * t) * 0.5
        right[i] = sin(2 * .pi * 660 * t) * 0.5
    }
    return MLXArray(left + right, [2, frameCount])
}

private func tempWAV() -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
}

@Suite("AudioImporter")
struct AudioImporterTests {
    @Test func roundTripAtSameSampleRatePreservesFrameCountAndValues() throws {
        let url = tempWAV()
        defer { try? FileManager.default.removeItem(at: url) }
        let frameCount = 4800 // 0.1 s @ 48 kHz
        let wave = syntheticWave(frameCount: frameCount, sampleRate: 48_000)
        try AudioExporter.exportToWAV(audio: wave, url: url, sampleRate: 48_000, format: .float32)

        let loaded = try AudioImporter.loadAudio(url: url, sampleRate: 48_000)
        #expect(loaded.shape == [1, frameCount, 2])

        let expected = wave.transposed(1, 0).reshaped([1, frameCount, 2]) // NCL -> NLC
        let diff = MLX.max(MLX.abs(loaded - expected)).item(Float.self)
        #expect(diff < 1e-4, "max abs diff \(diff)")
    }

    @Test func resamplesToRequestedRate() throws {
        let url = tempWAV()
        defer { try? FileManager.default.removeItem(at: url) }
        let sourceRate = 44_100.0
        let frameCount = 4410 // 0.1 s @ 44.1 kHz
        let wave = syntheticWave(frameCount: frameCount, sampleRate: sourceRate)
        try AudioExporter.exportToWAV(audio: wave, url: url, sampleRate: Int(sourceRate), format: .float32)

        let loaded = try AudioImporter.loadAudio(url: url, sampleRate: 48_000)
        let expectedFrames = Int((Double(frameCount) * 48_000 / sourceRate).rounded())
        #expect(loaded.dim(0) == 1 && loaded.dim(2) == 2)
        // Resampler frame counts aren't exact to the sample; allow a small window.
        #expect(abs(loaded.dim(1) - expectedFrames) <= 8, "got \(loaded.dim(1)), expected ~\(expectedFrames)")
    }

    @Test func upmixesMonoToStereo() throws {
        let url = tempWAV()
        defer { try? FileManager.default.removeItem(at: url) }
        let frameCount = 4800
        var mono = [Float](repeating: 0, count: frameCount)
        for i in 0..<frameCount { mono[i] = sin(2 * .pi * 440 * Float(i) / 48_000) * 0.5 }
        let monoArray = MLXArray(mono, [1, frameCount]) // [1, T] NCL, mono
        try AudioExporter.exportToWAV(audio: monoArray, url: url, sampleRate: 48_000, format: .float32)

        let loaded = try AudioImporter.loadAudio(url: url, sampleRate: 48_000)
        #expect(loaded.shape == [1, frameCount, 2])
    }

    @Test func throwsOnMissingFile() throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        #expect(throws: YuE2Error.self) {
            try AudioImporter.loadAudio(url: missing)
        }
    }
}
