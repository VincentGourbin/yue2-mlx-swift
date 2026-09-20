// VAEDecoderTests.swift - tiny parity (full decode, length, per-layer bisection) for T-1.8
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

/// numpy-`allclose`-style check: `|a - b| <= atol + rtol * |b|`, elementwise.
private func allClose(_ a: MLXArray, _ b: MLXArray, rtol: Float, atol: Float) -> Bool {
    let diff = MLX.abs(a - b)
    let bound = atol + rtol * MLX.abs(b)
    return MLX.all(diff .<= bound).item(Bool.self)
}

/// A temp directory holding `model.safetensors` (copy of the tiny fixture) + a matching
/// `config.json`, so `YuE2VAE.load(directory:)` runs its whole real pipeline against it.
private func makeTinyModelDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: tinyVAEFixture, to: dir.appendingPathComponent("model.safetensors"))
    let configJSON = """
    {
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

@Suite("VAEDecoder")
struct VAEDecoderTests {
    @Test func loadsTinyWithoutCoverageMismatch() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try YuE2VAE.load(directory: dir) // throws on any missing/unexpected parameter
    }

    @Test func decodeMatchesFullReferenceForEachLength() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)

        for length in [1, 15, 17, 33] {
            let inputNCL = try #require(raw["input_T\(length)"]) // torch [1, 2, T]
            let expectedNCL = try #require(raw["expected_full_T\(length)"]) // torch [1, 2, T_out]
            let z = inputNCL.transposed(0, 2, 1).asType(.float32) // MLX NLC [1, T, 2]
            let outputNLC = vae.decode(z)
            let outputNCL = outputNLC.transposed(0, 2, 1)
            let diff = maxAbsDiff(outputNCL, expectedNCL)
            #expect(diff < 1e-4, "T=\(length): max abs diff \(diff)")
        }
    }

    @Test func naturalOutputLengthMatchesReference() throws {
        let raw = try loadArrays(url: tinyVAEFixture)
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)

        for length in [1, 15, 17, 33] {
            let expected = try #require(raw["expected_full_T\(length)"])
            #expect(vae.naturalOutputLength(frames: length) == expected.dim(2))
            #expect(expected.dim(2) == 1920 * length - 64)
        }
    }

    @Test func perLayerBisectionMatchesTapsInOrder() throws {
        let raw = try loadArrays(url: tinyVAEFixture)
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        // `vae.decoder`'s parameter keys are prefixed "decoder." (nested under YuE2VAE), matching
        // VAEWeightLoader's own key convention; a standalone OobleckDecoder would not.
        let vae = try YuE2VAE.load(directory: dir)

        let inputNCL = try #require(raw["input_T33"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)

        var firstMismatch: (Int, Float)?
        _ = vae.decoder(z) { index, output in
            guard firstMismatch == nil, let expected = raw["tap_T33.after_layer_\(index)"] else { return }
            let diff = maxAbsDiff(output.transposed(0, 2, 1), expected)
            print("PARITY after_layer_\(index): max \(diff)")
            if diff >= 1e-4 { firstMismatch = (index, diff) }
        }

        if let (index, diff) = firstMismatch {
            Issue.record("first divergence at after_layer_\(index): max abs diff \(diff)")
        }
    }

    @Test(arguments: [(frames: 1, core: 1), (frames: 15, core: 7), (frames: 17, core: 8), (frames: 33, core: 16)])
    func decodeTiledMatchesFullDecodeEverywhereIncludingSeams(_ pair: (frames: Int, core: Int)) throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)
        let ratio = vae.config.downsamplingRatio
        let (frames, core) = pair

        let inputNCL = try #require(raw["input_T\(frames)"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)
        let full = vae.decode(z)
        let tiled = try vae.decodeTiled(z, coreFrames: core, haloFrames: 16)
        #expect(tiled.shape == full.shape, "frames=\(frames) core=\(core)")
        #expect(allClose(tiled, full, rtol: 3e-5, atol: 3e-6), "frames=\(frames) core=\(core): tiled != full globally")

        var seam = core
        while seam < frames {
            let center = seam * ratio
            let lo = max(0, center - 128)
            let hi = min(full.dim(1), center + 128)
            let fullWindow = full[0..., lo..<hi, 0...]
            let tiledWindow = tiled[0..., lo..<hi, 0...]
            #expect(
                allClose(tiledWindow, fullWindow, rtol: 3e-5, atol: 3e-6),
                "seam at frame \(seam) (sample \(center)): tiled != full"
            )
            seam += core
        }
    }

    @Test(arguments: [(frames: 17, core: 8), (frames: 33, core: 16)])
    func decodeTiledMatchesFixtureExpectedTiled(_ pair: (frames: Int, core: Int)) throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)
        let (frames, core) = pair

        let inputNCL = try #require(raw["input_T\(frames)"])
        let expectedNCL = try #require(raw["expected_tiled_T\(frames)_core\(core)"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)
        let tiled = try vae.decodeTiled(z, coreFrames: core, haloFrames: 16)
        let diff = maxAbsDiff(tiled.transposed(0, 2, 1), expectedNCL)
        #expect(diff < 1e-4, "frames=\(frames) core=\(core): max abs diff \(diff)")
    }

    @Test func decodeTiledRejectsHaloBelowMinimum() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)
        let inputNCL = try #require(raw["input_T33"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)
        #expect(throws: YuE2Error.self) {
            try vae.decodeTiled(z, coreFrames: 16, haloFrames: 15)
        }
    }

    @Test func decodeTiledRejectsNonPositiveCoreFrames() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)
        let inputNCL = try #require(raw["input_T33"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)
        #expect(throws: YuE2Error.self) {
            try vae.decodeTiled(z, coreFrames: 0, haloFrames: 16)
        }
    }

    @Test func decodeTiledWithOversizedCoreFramesIsOneTileEqualToFull() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)
        let inputNCL = try #require(raw["input_T33"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)
        let full = vae.decode(z)

        var progress: [(Int, Int)] = []
        let tiled = try vae.decodeTiled(z, coreFrames: 1000, haloFrames: 16) { completed, total in
            progress.append((completed, total))
        }
        #expect(progress.map(\.0) == [1] && progress.map(\.1) == [1])
        #expect(tiled.shape == full.shape)
        #expect(allClose(tiled, full, rtol: 3e-5, atol: 3e-6))
    }

    @Test func decodeTiledReportsProgressInOrder() throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)
        let inputNCL = try #require(raw["input_T33"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)

        var progress: [(Int, Int)] = []
        _ = try vae.decodeTiled(z, coreFrames: 16, haloFrames: 16) { completed, total in
            progress.append((completed, total))
        }
        let total = try #require(progress.first?.1)
        #expect(progress.map(\.0) == Array(1...total))
        #expect(progress.allSatisfy { $0.1 == total })
    }
}
