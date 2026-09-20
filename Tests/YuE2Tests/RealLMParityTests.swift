// RealLMParityTests.swift - real-checkpoint LM parity (tier 2, T-2.9, GATE G-2)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func lmFixtureReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("parity/lm.safetensors").path)
}

/// Mirrors `yue2 parity lm`: `Scripts/reference/real_fixtures.py lm`'s real (bf16, MPS)
/// prefixes/logits/hidden taps/greedy decodes, run through this build's AR backbone. Full
/// (unrestricted) `lm_head` for logits/bisection; the real O1 `RestrictedHead` path for the
/// two 8-token greedy decodes.
@Suite("RealLMParity", .enabled(if: lmFixtureReady()))
struct RealLMParityTests {
    private static let toleranceRel: Float = 2e-2
    private static let greedySampling = try! Sampling(
        temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1,
        penaltyWindow: 1, minTokens: 0, maxTokens: 8)

    private func loadRealModel() throws -> (model: YuE2ForCausalLM, config: YuE2Config) {
        let dir = try #require(modelsDir()).appendingPathComponent("YuE2-3B")
        let config = try YuE2Config.load(from: dir.appendingPathComponent("config.json"))
        return (try LMWeightLoader.load(directory: dir, config: config), config)
    }

    private func idArray(_ fixture: [String: MLXArray], _ key: String) throws -> [Int] {
        try #require(fixture[key]).asType(.int32).asArray(Int32.self).map(Int.init)
    }

    private func argmaxEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        MLX.argMax(a, axis: -1).item(UInt32.self) == MLX.argMax(b, axis: -1).item(UInt32.self)
    }

    @Test func lastTokenLogitsMatchFixtureABC() throws {
        let fixture = try loadFixture(name: "lm")
        let (model, config) = try loadRealModel()
        let prefix = try idArray(fixture, "prefix_abc")
        let ids = MLXArray(prefix.map { Int32($0) }).reshaped([1, prefix.count])
        let ours = model.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])
        let reference = try #require(fixture["logits_last_abc"]).reshaped([1, config.vocabSize])
        #expect(relativeError(ours, reference) <= Self.toleranceRel)
        #expect(argmaxEqual(ours, reference))
    }

    @Test func lastTokenLogitsMatchFixtureSemantic() throws {
        let fixture = try loadFixture(name: "lm")
        let (model, config) = try loadRealModel()
        let prefix = try idArray(fixture, "prefix_semantic")
        let ids = MLXArray(prefix.map { Int32($0) }).reshaped([1, prefix.count])
        let ours = model.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])
        let reference = try #require(fixture["logits_last_semantic"]).reshaped([1, config.vocabSize])
        #expect(relativeError(ours, reference) <= Self.toleranceRel)
        #expect(argmaxEqual(ours, reference))
    }

    /// Per-layer bisection on the ABC prefix's forward pass (`plan/08-tests.md`'s recipe:
    /// locate the first divergence by comparing taps in order).
    @Test(arguments: [0, 7, 14, 27])
    func hiddenLayerBisectionMatchesFixture(_ layer: Int) throws {
        let fixture = try loadFixture(name: "lm")
        let (model, _) = try loadRealModel()
        let prefix = try idArray(fixture, "prefix_abc")
        let ids = MLXArray(prefix.map { Int32($0) }).reshaped([1, prefix.count])
        var taps: [Int: MLXArray] = [:]
        _ = model.forwardAR(tokens: ids, cache: nil, keepLast: true) { i, h in taps[i] = h }
        let ours = try #require(taps[layer])
        let reference = try #require(fixture["hidden_after_layer_\(layer)"])
        #expect(relativeError(ours, reference) <= Self.toleranceRel)
    }

    @Test func greedyABCMatchesFixture() throws {
        let fixture = try loadFixture(name: "lm")
        let (model, _) = try loadRealModel()
        let prefix = try idArray(fixture, "prefix_abc")
        let head = model.makeRestrictedHead(bounds: .abc)
        let result = try TokenGenerator.generate(
            model: model, head: head, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .abc)
        #expect(result.tokens == (try idArray(fixture, "greedy_abc")))
    }

    @Test func greedySemanticMatchesFixture() throws {
        let fixture = try loadFixture(name: "lm")
        let (model, _) = try loadRealModel()
        let prefix = try idArray(fixture, "prefix_semantic")
        let head = model.makeRestrictedHead(bounds: .semantic)
        let result = try TokenGenerator.generate(
            model: model, head: head, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .semantic)
        #expect(result.tokens == (try idArray(fixture, "greedy_semantic")))
    }
}
