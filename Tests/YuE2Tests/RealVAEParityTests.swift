// RealVAEParityTests.swift - real-checkpoint VAE parity (tier 2, T-1.10)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXRandom
import Testing
@testable import YuE2Core

private func vaeFixtureReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("parity/vae.safetensors").path)
}

@Suite("RealVAEParity", .enabled(if: vaeFixtureReady()))
struct RealVAEParityTests {
    private static let toleranceAbs: Float = 2e-3
    private static let toleranceRel: Float = 1e-3

    private func require(_ fixture: [String: MLXArray], _ key: String) throws -> MLXArray {
        try #require(fixture[key])
    }

    @Test func decode40FramesMatchesTorch() throws {
        let fixture = try loadFixture(name: "vae")
        let dir = try #require(modelsDir())
        let vae = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"))

        let latent = try require(fixture, "latent_40").transposed(0, 2, 1).asType(.float32)
        let reference = try require(fixture, "audio_40")
        let ours = vae.decode(latent).transposed(0, 2, 1)

        let m = maxAbs(ours, reference)
        let r = relativeError(ours, reference)
        #expect(m <= Self.toleranceAbs, "max|Δ| \(m)")
        #expect(r <= Self.toleranceRel, "rel error \(r)")
    }

    @Test func tiledDecode1100FramesMatchesFullAndTorch() throws {
        let fixture = try loadFixture(name: "vae")
        let dir = try #require(modelsDir())
        let vae = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"))

        let latent = try require(fixture, "latent_1100").transposed(0, 2, 1).asType(.float32)
        let full = vae.decode(latent)
        let tiled = try vae.decodeTiled(latent, coreFrames: 1024)
        let referenceTiled = try require(fixture, "audio_1100_tiled")

        let vsFullMax = maxAbs(tiled, full)
        let vsFullRel = relativeError(tiled, full)
        #expect(vsFullMax <= Self.toleranceAbs, "tiled vs full max|Δ| \(vsFullMax)")
        #expect(vsFullRel <= Self.toleranceRel, "tiled vs full rel error \(vsFullRel)")

        let tiledNCL = tiled.transposed(0, 2, 1)
        let vsTorchMax = maxAbs(tiledNCL, referenceTiled)
        let vsTorchRel = relativeError(tiledNCL, referenceTiled)
        #expect(vsTorchMax <= Self.toleranceAbs, "tiled vs torch max|Δ| \(vsTorchMax)")
        #expect(vsTorchRel <= Self.toleranceRel, "tiled vs torch rel error \(vsTorchRel)")
    }

    /// E1 (plan/13-ios-backend.md §13.5): fp16 weights/compute, measured against this same
    /// model's own fp32 output (not the PyTorch fixture) — the question is fp16-vs-fp32.
    @Test func fp16DecodeStaysAboveMinimumSNR() throws {
        let dir = try #require(modelsDir())
        let fp32 = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"))
        let fp16 = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"), precision: .fp16)

        let latent = MLXRandom.normal([1, 40, 64], key: MLXRandom.key(11))
        let reference = fp32.decode(latent)
        let ours = fp16.decode(latent)

        #expect(MLX.all(MLX.isFinite(ours)).item(Bool.self), "fp16 decode produced non-finite values")
        let snr = snrDB(reference: reference, test: ours)
        #expect(snr > 40, "fp16 SNR \(snr) dB, expected > 40 dB")
    }

    @Test func encodeMatchesTorch() throws {
        let fixture = try loadFixture(name: "vae")
        try #require(fixture["audio_encode"] != nil) // older fixtures predate T-5.1
        let dir = try #require(modelsDir())
        let encoder = try YuE2VAEEncoder.load(directory: dir.appendingPathComponent("YuE2-Vae"))

        let audio = try require(fixture, "audio_encode").transposed(0, 2, 1).asType(.float32)
        let reference = try require(fixture, "encoded_expected")
        let ours = try encoder.encode(audio).transposed(0, 2, 1)

        let m = maxAbs(ours, reference)
        let r = relativeError(ours, reference)
        #expect(m <= Self.toleranceAbs, "max|Δ| \(m)")
        #expect(r <= Self.toleranceRel, "rel error \(r)")
    }
}
