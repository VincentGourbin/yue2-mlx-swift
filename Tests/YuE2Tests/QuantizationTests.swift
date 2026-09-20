// QuantizationTests.swift - real-checkpoint AR quantization parity (tier 2, T-4.5, O5)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import Testing
@testable import YuE2Core

private func lmFixtureReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("parity/lm.safetensors").path)
}

/// Compares the qint8 AR path (O5, `--quant qint8`) against the checkpoint-native bf16 path on
/// the same real prefixes `RealLMParityTests` uses — never against the Python reference here,
/// since the fiche's own criterion is quantized-vs-bf16, not quantized-vs-torch. Both models are
/// loaded fresh (no prequantized export reused) so this suite always exercises the on-the-fly
/// quantization path, the one every first run of `--quant qint8` actually takes.
@Suite("Quantization", .enabled(if: lmFixtureReady()))
struct QuantizationTests {
    private static let toleranceRel: Float = 5e-2
    private static let greedySampling = try! Sampling(
        temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1,
        penaltyWindow: 1, minTokens: 0, maxTokens: 8)

    private func lmDir() throws -> URL {
        try #require(modelsDir()).appendingPathComponent("YuE2-3B")
    }

    private func idArray(_ fixture: [String: MLXArray], _ key: String) throws -> [Int] {
        try #require(fixture[key]).asType(.int32).asArray(Int32.self).map(Int.init)
    }

    private func argmaxEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        MLX.argMax(a, axis: -1).item(UInt32.self) == MLX.argMax(b, axis: -1).item(UInt32.self)
    }

    /// Every projection this preset is allowed to touch became a `QuantizedLinear`; everything
    /// else (embeddings, norms, NAR branch, VAE bridge) is untouched — the structural half of
    /// the fiche's exclusion list (plan/07-jalon4.md O5), checked once so a future refactor of
    /// `YuE2QuantizationFilter` can't silently widen or narrow it.
    @Test func qint8QuantizesOnlyEligibleProjections() throws {
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let model = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .qint8)

        var quantizedPaths = Set<String>()
        var plainLinearPaths = Set<String>()
        for (path, module) in model.leafModules().flattened() {
            if module is QuantizedLinear { quantizedPaths.insert(path) }
            if module is Linear && !(module is QuantizedLinear) { plainLinearPaths.insert(path) }
        }

        #expect(!quantizedPaths.isEmpty)
        #expect(quantizedPaths.allSatisfy { $0.contains("self_attn") || $0.contains("mlp") })
        #expect(!quantizedPaths.contains(where: { $0.contains("nar_") }))
        #expect(!quantizedPaths.contains("lm_head"))  // --quant-head not requested
        // Untouched by construction (not Linear at all): embed_tokens, every *norm*,
        // latent_pos_embed. Untouched by the filter (Linear, but not self_attn/mlp):
        // lm_head (no --quant-head), vae2llm, llm2vae, time_embedder.fc1/fc2.
        for excluded in ["lm_head", "vae2llm", "llm2vae"] {
            #expect(plainLinearPaths.contains(excluded))
        }
    }

    @Test func qint8LogitsCloseToBF16OnABCPrefix() throws {
        let fixture = try loadFixture(name: "lm")
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let bf16 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .none)
        let qint8 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .qint8)

        let prefix = try idArray(fixture, "prefix_abc")
        let ids = MLXArray(prefix.map { Int32($0) }).reshaped([1, prefix.count])
        let refLogits = bf16.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])
        let quantLogits = qint8.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])

        let rel = relativeError(quantLogits, refLogits)
        print("quant abc rel=\(rel)")
        #expect(rel <= Self.toleranceRel)
        #expect(argmaxEqual(quantLogits, refLogits))
    }

    @Test func qint8LogitsCloseToBF16OnSemanticPrefix() throws {
        let fixture = try loadFixture(name: "lm")
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let bf16 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .none)
        let qint8 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .qint8)

        let prefix = try idArray(fixture, "prefix_semantic")
        let ids = MLXArray(prefix.map { Int32($0) }).reshaped([1, prefix.count])
        let refLogits = bf16.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])
        let quantLogits = qint8.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])

        let rel = relativeError(quantLogits, refLogits)
        print("quant semantic rel=\(rel)")
        #expect(rel <= Self.toleranceRel)
    }

    /// ≥ 7/8 tokens identical (fiche's own bar — quantization is exact-ish, not bit-exact, so an
    /// occasional last-token flip near a decision boundary is tolerated).
    @Test func qint8GreedyMatchesBF16OnABCPrefix() throws {
        let fixture = try loadFixture(name: "lm")
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let bf16 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .none)
        let qint8 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .qint8)
        let prefix = try idArray(fixture, "prefix_abc")

        let refHead = bf16.makeRestrictedHead(bounds: .abc)
        let refResult = try TokenGenerator.generate(
            model: bf16, head: refHead, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .abc)
        let quantHead = qint8.makeRestrictedHead(bounds: .abc)
        let quantResult = try TokenGenerator.generate(
            model: qint8, head: quantHead, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .abc)

        let matches = zip(refResult.tokens, quantResult.tokens).filter { $0 == $1 }.count
        print("quant greedy_abc matches=\(matches)/8 ref=\(refResult.tokens) quant=\(quantResult.tokens)")
        #expect(matches >= 7)
    }

    @Test func qint8GreedyMatchesBF16OnSemanticPrefix() throws {
        let fixture = try loadFixture(name: "lm")
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let bf16 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .none)
        let qint8 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .qint8)
        let prefix = try idArray(fixture, "prefix_semantic")

        let refHead = bf16.makeRestrictedHead(bounds: .semantic)
        let refResult = try TokenGenerator.generate(
            model: bf16, head: refHead, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .semantic)
        let quantHead = qint8.makeRestrictedHead(bounds: .semantic)
        let quantResult = try TokenGenerator.generate(
            model: qint8, head: quantHead, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .semantic)

        let matches = zip(refResult.tokens, quantResult.tokens).filter { $0 == $1 }.count
        print("quant greedy_semantic matches=\(matches)/8 ref=\(refResult.tokens) quant=\(quantResult.tokens)")
        #expect(matches >= 7)
    }

    /// `--quant-head` (O5's optional extra): `RestrictedHead` must dequantize the gathered rows
    /// of a `QuantizedLinear` `lm_head`, not just of a plain `Linear` one (§9 n°10).
    @Test func quantHeadRestrictedHeadClosesToBF16() throws {
        let fixture = try loadFixture(name: "lm")
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let bf16 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .none)
        let qint8Head = try LMWeightLoader.load(
            directory: try lmDir(), config: config, quantization: .qint8, quantizeHead: true)
        #expect(qint8Head.makeRestrictedHead(bounds: .abc).globalIDs.count > 0)

        let prefix = try idArray(fixture, "prefix_abc")
        let refHead = bf16.makeRestrictedHead(bounds: .abc)
        let refResult = try TokenGenerator.generate(
            model: bf16, head: refHead, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .abc)
        let quantHead = qint8Head.makeRestrictedHead(bounds: .abc)
        let quantResult = try TokenGenerator.generate(
            model: qint8Head, head: quantHead, prefix: prefix, sampling: Self.greedySampling, seed: 0, bounds: .abc)

        let matches = zip(refResult.tokens, quantResult.tokens).filter { $0 == $1 }.count
        print("quant quant_head matches=\(matches)/8 ref=\(refResult.tokens) quant=\(quantResult.tokens)")
        #expect(matches >= 7)
    }
}
