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

/// Issue #4 stand-in for MLX's shape-dependent fp16 `convTransposed1d`: decode windows above
/// `badWindowFrames` return a NaN in the middle of the output (so it survives the halo crop),
/// smaller windows return clean silence. Records every window it is asked to decode.
private final class ShapeSensitiveBackend: VAEDecoding {
    let config: VAEConfig
    let badWindowFrames: Int
    private(set) var seenWindows: [Int] = []

    init(config: VAEConfig, badWindowFrames: Int) {
        self.config = config
        self.badWindowFrames = badWindowFrames
    }

    func decode(_ z: MLXArray) async throws -> MLXArray {
        let t = z.dim(1)
        seenWindows.append(t)
        let length = naturalOutputLength(frames: t)
        let midpoint = length / 2
        let bad = t > badWindowFrames
        let head = MLXArray.zeros([1, midpoint, 1], dtype: .float32)
        let spike = MLXArray([bad ? Float.nan : Float(0)], [1, 1, 1])
        let tail = MLXArray.zeros([1, length - midpoint - 1, 1], dtype: .float32)
        return concatenated([head, spike, tail], axis: 1)
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

    /// Issue #4: the retry policy halves an offending tile and stops at 64 core frames. A clean
    /// pass, or one at the minimum tile, is final (data-side NaNs are left to the exporter).
    @Test func nonFiniteTileRetryPolicyHalvesAndStopsAtMinimum() {
        #expect(nextVAECoreFrames(current: 1024, foundNonFinite: true) == 512)
        #expect(nextVAECoreFrames(current: 512, foundNonFinite: true) == 256)
        #expect(nextVAECoreFrames(current: 128, foundNonFinite: true) == 64)
        #expect(nextVAECoreFrames(current: 64, foundNonFinite: true) == nil)
        #expect(nextVAECoreFrames(current: 1024, foundNonFinite: false) == nil)
        #expect(nextVAECoreFrames(current: 1, foundNonFinite: true) == nil)
    }

    /// Issue #4 end to end through the package-level entry point: a backend that mimics the
    /// shape-dependent fp16 NaN (bad above 700 frames) must be re-decoded at half the tile, and
    /// the returned audio must contain no non-finite sample — including callers that explicitly
    /// request `--vae-core-frames 1024` / `--core-frames 1024`.
    @Test func decodeTiledRetriesNonFiniteWindowsAtHalfTheTile() async throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try VAEConfig.load(from: dir.appendingPathComponent("config.json"))
        let backend = ShapeSensitiveBackend(config: config, badWindowFrames: 700)
        let z = MLXArray.zeros([1, 1400, config.latentDim], dtype: .float32)

        let audio = try await decodeTiled(using: backend, z, coreFrames: 1024, haloFrames: 16)
        let expectedLength = config.downsamplingRatio * 1400 - 64
        #expect(audio.shape == [1, expectedLength, 1])
        #expect(vaeCropIsFinite(audio))
        // Pass 1 asks for one 1040-frame window (bad) and aborts; pass 2 uses 512-frame cores
        // whose windows (528 / 544 / 392 frames) are all below the bad band.
        #expect(backend.seenWindows.first == 1040)
        #expect(backend.seenWindows.count == 4)
        #expect(backend.seenWindows.dropFirst().allSatisfy { $0 <= 700 })
    }

    /// The same shape with a clean backend must take exactly one pass (no extra decode, no
    /// extra memory): the finiteness reduction is the only added work.
    @Test func decodeTiledDoesNotRetryCleanWindows() async throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try VAEConfig.load(from: dir.appendingPathComponent("config.json"))
        let backend = ShapeSensitiveBackend(config: config, badWindowFrames: Int.max)
        let z = MLXArray.zeros([1, 1400, config.latentDim], dtype: .float32)

        let audio = try await decodeTiled(using: backend, z, coreFrames: 1024, haloFrames: 16)
        #expect(backend.seenWindows == [1040, 392])
        #expect(vaeCropIsFinite(audio))
    }

    /// Both decode entry points (`decodeTiled(using:)` and `YuE2VAE.decodeTiled`) keep their
    /// bit-for-bit independence of `coreFrames` with the retry guard in place. The second one
    /// is the path used by `synthesize` / `remix` / `encode` / `parity`.
    @Test func bothDecodeEntryPointsStillAgreeAcrossTileSizes() async throws {
        let dir = try makeTinyModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let vae = try YuE2VAE.load(directory: dir)
        let z = MLXArray.zeros([1, 1300, vae.config.latentDim], dtype: .float32)

        let viaProtocol = try await decodeTiled(using: vae, z, coreFrames: 1024)
        let direct = try vae.decodeTiled(z, coreFrames: 1024)
        let smaller = try vae.decodeTiled(z, coreFrames: 640)
        #expect(MLX.all(viaProtocol .== direct).item(Bool.self))
        #expect(MLX.all(smaller .== direct).item(Bool.self))
        #expect(vaeCropIsFinite(direct))
    }

    /// Issue #4: fp16 `convTransposed1d` emits NaNs for decode windows in roughly the
    /// 704...1280-frame band. `decodeTiled` output is bit-for-bit independent of
    /// `coreFrames`, so keeping fp16 profiles at <= 640 costs no accuracy, time or memory.
    @Test func fp16ProfilesStayBelowUnsafeVAETileBand() {
        let fp16Profiles = YuE2ReferenceProfile.all.filter { $0.vaePrecision == .fp16 }
        #expect(fp16Profiles.count == 7, "expected all seven reference profiles to use the fp16 VAE")
        for profile in fp16Profiles {
            #expect(
                profile.vaeCoreFrames <= 640,
                "\(profile.id) uses \(profile.vaeCoreFrames)-frame fp16 VAE tiles; see issue #4"
            )
        }
    }
}
