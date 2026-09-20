// VAEEncoderTests.swift - tiny parity (encode, per-layer bisection, input rejection) for T-5.1
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import Testing
@testable import YuE2Core

private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let tinyVAEFixture = repoRoot.appendingPathComponent("parity/tiny_vae.safetensors")

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a - b)).item(Float.self)
}

/// A temp directory holding `model.safetensors` (copy of the tiny fixture, which carries both
/// `encoder.*` and `decoder.*`) plus a matching `config.json` with both `encoder_config` and
/// `decoder_config`, so `YuE2VAEEncoder.load(directory:)` runs its whole real pipeline.
private func makeTinyModelDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: tinyVAEFixture, to: dir.appendingPathComponent("model.safetensors"))
    let configJSON = """
    {
      "encoder_config": {
        "in_channels": 2, "channels": 1, "c_mults": [1,1,1,1,1,1], "strides": [2,2,4,4,5,6],
        "latent_dim": 4, "use_snake": true
      },
      "decoder_config": {
        "channels": 1, "c_mults": [1,1,1,1,1,1], "strides": [2,2,4,4,5,6],
        "latent_dim": 2, "out_channels": 2, "use_snake": true, "snake_type": "vanilla",
        "final_tanh": false
      },
      "latent_dim": 2, "downsampling_ratio": 1920, "sample_rate": 48000,
      "decode_core_frames": 1024, "decode_halo_frames": 16, "release_variant": "tiny"
    }
    """
    try configJSON.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    return dir
}

@Suite("VAEEncoder")
struct VAEEncoderTests {
    @Test func loadsTinyWithoutCoverageMismatch() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try YuE2VAEEncoder.load(directory: dir) // throws on any missing/unexpected parameter
    }

    @Test func encodeMatchesReferenceForEachLength() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let encoder = try YuE2VAEEncoder.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)

        for length in [1, 9] {
            let audioNCL = try #require(raw["audio_T\(length)"]) // torch [1, 2, S]
            let expectedNCL = try #require(raw["expected_encoded_T\(length)"]) // torch [1, 2, T]
            let audioNLC = audioNCL.transposed(0, 2, 1).asType(.float32) // MLX NLC [1, S, 2]
            let outputNLC = try encoder.encode(audioNLC)
            let outputNCL = outputNLC.transposed(0, 2, 1)
            #expect(outputNCL.shape == expectedNCL.shape, "T=\(length): shape \(outputNCL.shape) != \(expectedNCL.shape)")
            let diff = maxAbsDiff(outputNCL, expectedNCL)
            #expect(diff < 1e-4, "T=\(length): max abs diff \(diff)")
        }
    }

    @Test func perLayerBisectionMatchesReferenceFromLayerTwo() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let encoder = try YuE2VAEEncoder.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)

        let audioNCL = try #require(raw["audio_T9"])
        let audioNLC = audioNCL.transposed(0, 2, 1).asType(.float32)

        var taps: [Int: MLXArray] = [:]
        _ = try encoder.encode(audioNLC) { index, output in taps[index] = output }

        // Fixture only stores taps from layer 2 onward (parity/ size budget, T-1.3/T-5.1 note).
        for index in 2...8 {
            let expected = try #require(raw["tap_encode_T9.after_layer_\(index)"])
            let ours = try #require(taps[index]).transposed(0, 2, 1) // NLC -> torch NCL
            let diff = maxAbsDiff(ours, expected)
            #expect(diff < 1e-4, "layer \(index): max abs diff \(diff)")
        }
    }

    @Test func rejectsAudioShorterThanOneFrame() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let encoder = try YuE2VAEEncoder.load(directory: dir)
        let tooShort = MLXArray.zeros([1, 100, 2]).asType(.float32) // < downsamplingRatio (1920)
        #expect(throws: YuE2Error.self) {
            try encoder.encode(tooShort)
        }
    }

    @Test func rejectsWrongChannelCount() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let encoder = try YuE2VAEEncoder.load(directory: dir)
        let monoAudio = MLXArray.zeros([1, 1920, 1]).asType(.float32)
        #expect(throws: YuE2Error.self) {
            try encoder.encode(monoAudio)
        }
    }

    @Test func rejectsNonFiniteAudio() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let encoder = try YuE2VAEEncoder.load(directory: dir)
        var values = [Float](repeating: 0, count: 1920 * 2)
        values[0] = .nan
        let audio = MLXArray(values, [1, 1920, 2])
        #expect(throws: YuE2Error.self) {
            try encoder.encode(audio)
        }
    }
}
