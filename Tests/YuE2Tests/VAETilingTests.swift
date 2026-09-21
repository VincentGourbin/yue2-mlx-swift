// VAETilingTests.swift - planVAETiles / decodeTiled(using:) windowing for enumerated-shape backends
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let tinyVAEFixture = repoRoot.appendingPathComponent("parity/tiny_vae.safetensors")

private func allClose(_ a: MLXArray, _ b: MLXArray, rtol: Float, atol: Float) -> Bool {
    let diff = MLX.abs(a - b)
    let bound = atol + rtol * MLX.abs(b)
    return MLX.all(diff .<= bound).item(Bool.self)
}

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

/// A stand-in for `CoreAIVAEDecoder`: the MLX tiny decoder, but declaring enumerated shapes and
/// refusing any other frame count (Core AI's `run` would reject a wrong shape the same way).
private struct EnumeratedBackend: VAEDecoding {
    let vae: YuE2VAE
    let shapes: [Int]
    var config: VAEConfig { vae.config }
    var enumeratedFrameCounts: [Int]? { shapes }

    func decode(_ z: MLXArray) async throws -> MLXArray {
        guard shapes.contains(z.dim(1)) else {
            throw YuE2Error.invalidRequest("EnumeratedBackend: \(z.dim(1)) frames is not one of \(shapes)")
        }
        return vae.decode(z)
    }
}

@Suite("VAETiling")
struct VAETilingTests {
    /// Every window covers its core plus at least the halo on each side (except at the signal
    /// edges), cores tile `[0, frames)` contiguously, and nothing is padded when the signal is
    /// at least as long as the smallest enumerated shape.
    private func checkInvariants(_ plan: [VAETileWindow], frames: Int, core: Int, halo: Int, enumerated: [Int]?) {
        var expectedStart = 0
        for w in plan {
            #expect(w.coreStart == expectedStart)
            #expect(w.coreEnd == min(frames, w.coreStart + core))
            #expect(w.windowStart >= 0 && w.windowEnd <= frames)
            #expect(w.windowStart <= max(0, w.coreStart - halo))
            #expect(w.windowEnd >= min(frames, w.coreEnd + halo))
            if let enumerated {
                #expect(enumerated.contains(w.windowFrames + w.paddedFrames))
                if frames >= enumerated.min()! { #expect(w.paddedFrames == 0) }
            } else {
                #expect(w.paddedFrames == 0)
            }
            expectedStart = w.coreEnd
        }
        #expect(expectedStart == frames)
    }

    @Test func withoutEnumeratedShapesReproducesTheReferencePlan() {
        let plan = planVAETiles(frames: 1100, coreFrames: 1024, haloFrames: 16, enumerated: nil)
        #expect(plan == [
            VAETileWindow(windowStart: 0, windowEnd: 1040, coreStart: 0, coreEnd: 1024, paddedFrames: 0),
            VAETileWindow(windowStart: 1008, windowEnd: 1100, coreStart: 1024, coreEnd: 1100, paddedFrames: 0),
        ])
        checkInvariants(plan, frames: 1100, core: 1024, halo: 16, enumerated: nil)
    }

    /// The T-6.3 production case: the short final tile (92 real frames) used to be zero-padded to
    /// 288 and lost 34 dB; it is now anchored to the signal's end as an exact 288-frame window.
    @Test func enumeratedShapesWidenWindowsInsteadOfPadding() {
        let shapes = [288, 544, 1056]
        let plan = planVAETiles(frames: 1100, coreFrames: 1024, haloFrames: 16, enumerated: shapes)
        #expect(plan == [
            VAETileWindow(windowStart: 0, windowEnd: 1056, coreStart: 0, coreEnd: 1024, paddedFrames: 0),
            VAETileWindow(windowStart: 812, windowEnd: 1100, coreStart: 1024, coreEnd: 1100, paddedFrames: 0),
        ])
        checkInvariants(plan, frames: 1100, core: 1024, halo: 16, enumerated: shapes)

        let phone = planVAETiles(frames: 1599, coreFrames: 256, haloFrames: 16, enumerated: shapes)
        #expect(phone.count == 7)
        #expect(phone.allSatisfy { $0.windowFrames == 288 && $0.paddedFrames == 0 })
        checkInvariants(phone, frames: 1599, core: 256, halo: 16, enumerated: shapes)
    }

    @Test func signalShorterThanSmallestShapeIsPaddedByTheBackend() {
        let plan = planVAETiles(frames: 100, coreFrames: 1024, haloFrames: 16, enumerated: [288, 544, 1056])
        #expect(plan == [VAETileWindow(windowStart: 0, windowEnd: 100, coreStart: 0, coreEnd: 100, paddedFrames: 188)])
    }

    @Test(arguments: [(frames: 33, core: 7, shapes: [16, 24, 33]), (frames: 33, core: 16, shapes: [33]), (frames: 17, core: 8, shapes: [17])])
    func decodeTiledThroughEnumeratedBackendMatchesFullDecode(_ c: (frames: Int, core: Int, shapes: [Int])) async throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let raw = try loadArrays(url: tinyVAEFixture)
        let inputNCL = try #require(raw["input_T\(c.frames)"])
        let z = inputNCL.transposed(0, 2, 1).asType(.float32)
        let full = vae.decode(z)

        let backend = EnumeratedBackend(vae: vae, shapes: c.shapes)
        let tiled = try await decodeTiled(using: backend, z, coreFrames: c.core, haloFrames: 16)
        #expect(tiled.shape == full.shape)
        #expect(allClose(tiled, full, rtol: 3e-5, atol: 3e-6), "frames=\(c.frames) core=\(c.core) shapes=\(c.shapes)")

        // And the MLX backend (no enumerated shapes) still goes through the original plan, bit
        // for bit the same as `YuE2VAE.decodeTiled`.
        let viaProtocol = try await decodeTiled(using: vae, z, coreFrames: c.core, haloFrames: 16)
        let direct = try vae.decodeTiled(z, coreFrames: c.core, haloFrames: 16)
        #expect(MLX.all(viaProtocol .== direct).item(Bool.self))
    }
}
