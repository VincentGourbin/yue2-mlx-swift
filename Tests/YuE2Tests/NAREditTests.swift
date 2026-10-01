// NAREditTests.swift - NAREdit.variation / .keep through Synthesizer on the tiny model
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float { MLX.max(MLX.abs(a - b)).item(Float.self) }

@Suite("NAREdit", .serialized)
struct NAREditTests {
    @Test func strengthZeroReturnsTheSourceAndOneIsAFreshSolve() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let codec = (0..<noise.dim(0)).map { $0 % 4 }
        let synthesizer = Synthesizer(model: model, config: try GenerationConfig(odeSteps: 4))
        let fresh = try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise)
        let source = MLXRandom.normal(fresh.shape, key: MLXRandom.key(99))

        let unchanged = try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise, edit: .variation(latents: source, strength: 0))
        #expect(maxAbsDiff(unchanged, source) == 0)
        let full = try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise, edit: .variation(latents: source, strength: 1))
        #expect(maxAbsDiff(full, fresh) == 0)
        let mid = try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise, edit: .variation(latents: source, strength: 0.5))
        #expect(maxAbsDiff(mid, fresh) > 0 && maxAbsDiff(mid, source) > 0)
    }

    @Test func keepFramesAreReturnedExactlyAndTheRestIsSolved() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let codec = (0..<noise.dim(0)).map { $0 % 4 }
        let synthesizer = Synthesizer(model: model, config: try GenerationConfig(odeSteps: 4))
        let source = MLXRandom.normal(noise.shape, key: MLXRandom.key(5))
        let keptFrames = max(1, noise.dim(0) - 1)
        let out = try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise, edit: .keep(latents: source, frames: keptFrames))
        #expect(maxAbsDiff(out[0..<keptFrames, 0...], source[0..<keptFrames, 0...]) == 0)
        #expect(MLX.all(MLX.isFinite(out)).item(Bool.self))
        #expect(throws: YuE2Error.self) {
            try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise, edit: .keep(latents: source, frames: noise.dim(0) + 1))
        }
    }
}
