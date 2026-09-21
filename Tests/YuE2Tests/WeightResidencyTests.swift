// WeightResidencyTests.swift - stage-scoped weight residency (YuE2ForCausalLM.releaseWeights)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a - b)).item(Float.self)
}

@Suite("WeightResidency")
struct WeightResidencyTests {
    @Test func everyCheckpointKeyIsClassified() throws {
        let model = try loadedTinyModel()
        var counts: [String: Int] = [:]
        for (key, _) in model.parameters().flattened() {
            switch YuE2WeightPath.of(key: key) {
            case .ar: counts["ar", default: 0] += 1
            case .nar: counts["nar", default: 0] += 1
            case nil:
                #expect(key.contains("norm"), "unclassified key that is not a norm: \(key)")
                counts["shared", default: 0] += 1
            }
        }
        // Per layer: q/k/v/o + q_norm/k_norm + gate/up/down = 9 leaves on each branch; the NAR
        // branch also owns its two `nar_*` layer norms. AR adds embed_tokens + lm_head; NAR adds
        // vae2llm/llm2vae (weight+bias), time_embedder (4), pe. Shared: the AR-side input/
        // post-attention norms (a few KB, deliberately kept) and `model.norm`, read by both.
        #expect(counts["ar"] == 2 * 9 + 2)
        #expect(counts["nar"] == 2 * 11 + 4 + 4 + 1)
        #expect(counts["shared"] == 2 * 2 + 1)
        #expect(YuE2WeightPath.of(key: "model.layers.3.nar_mlp.down_proj.weight") == .nar)
        #expect(YuE2WeightPath.of(key: "model.layers.3.mlp.down_proj.scales") == .ar)
        #expect(YuE2WeightPath.of(key: "model.embed_tokens.weight") == .ar)
        #expect(YuE2WeightPath.of(key: "latent_pos_embed.pe") == .nar)
        #expect(YuE2WeightPath.of(key: "model.norm.weight") == nil)
    }

    /// Dropping the AR branch after the NAR prefix cache exists leaves `velocity` bit-identical
    /// (nothing in the solve reads those weights), keeps the module tree intact, and releases
    /// exactly the AR bytes.
    @Test func releasingARPathLeavesNARVelocityIdentical() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let chunk = NARChunk(arTokens: [2, 3, 4, 5], noise: noise)
        let nar = CachedNAR(model: model, chunk: chunk)
        let before = nar.velocity(state: noise, rawT: 0.0)
        eval(before)

        let keysBefore = model.parameters().flattened().map(\.0).sorted()
        let expectedBytes = model.parameters().flattened()
            .filter { YuE2WeightPath.of(key: $0.0) == .ar }
            .reduce(0) { $0 + $1.1.nbytes }
        let released = model.releaseWeights(of: .ar)
        #expect(released == expectedBytes)
        #expect(model.parameters().flattened().map(\.0).sorted() == keysBefore)

        let after = nar.velocity(state: noise, rawT: 0.0)
        #expect(maxAbsDiff(before, after) == 0)
        let expected = try #require(raw["velocity_raw0.0"])
        #expect(maxAbsDiff(after, expected) <= 1e-4)

        // Releasing again is a no-op on the AR bytes (already zeros of the same shape), and the
        // whole-LM release still reports everything that is left.
        #expect(model.releaseWeights(of: .ar) == expectedBytes)
        #expect(model.releaseAllWeights() > expectedBytes)
    }

    /// `LMWeightLoader.load(residency: [.ar])` leaves the NAR branch unmaterialized (zeros), and
    /// `load(path: .nar, into:)` brings it in afterwards — the tiny bf16 checkpoint through the
    /// same code path the prequantized int4-mixed pack takes.
    @Test func loadingOneBranchThenTheOtherMatchesAFullLoad() throws {
        let dir = try makeTinyLMDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let raw = try loadArrays(url: tinyLMFixture)
        let full = try loadedTinyModel()
        let partial = try LMWeightLoader.load(directory: dir, config: tinyLMConfig(), residency: [.ar])

        let fullParams = Dictionary(uniqueKeysWithValues: full.parameters().flattened())
        for (key, value) in partial.parameters().flattened() {
            let same = MLX.all(value .== fullParams[key]!).item(Bool.self)
            switch YuE2WeightPath.of(key: key) {
            case .nar where key.contains("norm"):
                break // RMSNorm inits to ones and the fixture's norms are ones too: no signal either way
            case .nar:
                #expect(!same, "NAR key loaded although not resident: \(key)")
            default:
                #expect(same, "AR/shared key differs from the full load: \(key)")
            }
        }

        try LMWeightLoader.load(path: .nar, into: partial, directory: dir)
        for (key, value) in partial.parameters().flattened() {
            #expect(MLX.all(value .== fullParams[key]!).item(Bool.self), "\(key) differs after loading the NAR branch")
        }
        let noise = try #require(raw["nar_noise"])
        let nar = CachedNAR(model: partial, chunk: NARChunk(arTokens: [2, 3, 4, 5], noise: noise))
        let expected = try #require(raw["velocity_raw0.0"])
        #expect(maxAbsDiff(nar.velocity(state: noise, rawT: 0.0), expected) <= 1e-4)
    }
}
