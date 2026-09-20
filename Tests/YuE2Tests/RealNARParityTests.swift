// RealNARParityTests.swift - real-checkpoint NAR parity (tier 2, T-3.5)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func narFixtureReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("parity/nar.safetensors").path)
}

/// Mirrors `yue2 parity nar`: `Scripts/reference/real_fixtures.py nar`'s real (bf16, MPS)
/// velocity evaluations, bisection taps, and 4-step solve, run through this build's `CachedNAR`.
@Suite("RealNARParity", .enabled(if: narFixtureReady()))
struct RealNARParityTests {
    private static let toleranceVelocity: Float = 3e-2
    private static let toleranceSolve: Float = 5e-2
    private static let watchedLayers = [0, 7, 14, 27]

    private func makeCachedNAR(_ fixture: [String: MLXArray]) throws -> (nar: CachedNAR, noise: MLXArray) {
        let dir = try #require(modelsDir()).appendingPathComponent("YuE2-3B")
        let config = try YuE2Config.load(from: dir.appendingPathComponent("config.json"))
        let model = try LMWeightLoader.load(directory: dir, config: config)
        let arTokens = try #require(fixture["ar_tokens"]).asType(.int32).asArray(Int32.self).map(Int.init)
        let noise = try #require(fixture["noise"])
        let chunk = NARChunk(arTokens: arTokens, noise: noise)
        return (CachedNAR(model: model, chunk: chunk), noise)
    }

    @Test(arguments: [20.0, 0.0, -2.3])
    func velocityMatchesFixture(_ raw: Double) throws {
        let fixture = try loadFixture(name: "nar")
        let (nar, noise) = try makeCachedNAR(fixture)
        let velocity = nar.velocity(state: noise, rawT: raw)
        let reference = try #require(fixture["velocity_raw\(raw)"])
        #expect(relativeError(velocity, reference) <= Self.toleranceVelocity)
    }

    @Test func bisectionTapsMatchFixtureAtRaw20() throws {
        let fixture = try loadFixture(name: "nar")
        let (nar, noise) = try makeCachedNAR(fixture)
        var taps: [String: MLXArray] = [:]
        _ = nar.velocity(state: noise, rawT: 20.0) { name, value in taps[name] = value }

        let pos = try #require(taps["after_pos"])
        #expect(relativeError(pos, try #require(fixture["nar_after_pos"])) <= Self.toleranceVelocity)
        for i in Self.watchedLayers {
            let got = try #require(taps["after_layer_\(i)"])
            let reference = try #require(fixture["nar_after_layer_\(i)"])
            #expect(relativeError(got, reference) <= Self.toleranceVelocity)
        }
    }

    @Test func solveFourStepsMatchesFixture() throws {
        let fixture = try loadFixture(name: "nar")
        let (nar, _) = try makeCachedNAR(fixture)
        let result = try nar.solve(steps: 4)
        let reference = try #require(fixture["solve4_expected"])
        #expect(relativeError(result, reference) <= Self.toleranceSolve)
    }
}
