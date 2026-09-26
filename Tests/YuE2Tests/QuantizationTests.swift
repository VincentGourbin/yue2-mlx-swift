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

    // MARK: - "-all" presets (T-6.2, E2): both MoT branches + embed_tokens, 4-bit.

    /// Structural check mirroring `qint8QuantizesOnlyEligibleProjections`: an "-all" preset must
    /// reach `nar_self_attn`/`nar_mlp` and `embed_tokens` (via `QuantizedEmbedding`), which the
    /// AR-only presets never touch.
    @Test func int4AllQuantizesNARAndEmbeddingsToo() throws {
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let model = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .int4All)

        var quantizedPaths = Set<String>()
        for (path, module) in model.leafModules().flattened() where module is Quantized {
            quantizedPaths.insert(path)
        }
        #expect(quantizedPaths.contains(where: { $0.contains("nar_self_attn") }))
        #expect(quantizedPaths.contains(where: { $0.contains("nar_mlp") }))
        #expect(quantizedPaths.contains(where: { $0.contains("embed_tokens") }))
        #expect(!quantizedPaths.contains("lm_head")) // --quant-head not requested here either
    }

    /// int4-all vs the checkpoint-native bf16 path, on the same real ABC prefix `RealLMParityTests`
    /// uses — logits closeness and greedy agreement, same bar as the AR-only preset's tests.
    @Test func int4AllLogitsAndGreedyCloseToBF16() throws {
        let fixture = try loadFixture(name: "lm")
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let bf16 = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .none)
        let int4All = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .int4All)
        let prefix = try idArray(fixture, "prefix_abc")
        let ids = MLXArray(prefix.map { Int32($0) }).reshaped([1, prefix.count])

        let refLogits = bf16.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])
        let quantLogits = int4All.forwardAR(tokens: ids, cache: nil, keepLast: true).reshaped([1, config.vocabSize])
        let rel = relativeError(quantLogits, refLogits)
        print("int4-all abc logits rel=\(rel)")
        #expect(rel <= Self.toleranceRel)

        let refResult = try TokenGenerator.generate(
            model: bf16, head: bf16.makeRestrictedHead(bounds: .abc), prefix: prefix,
            sampling: Self.greedySampling, seed: 0, bounds: .abc)
        let quantResult = try TokenGenerator.generate(
            model: int4All, head: int4All.makeRestrictedHead(bounds: .abc), prefix: prefix,
            sampling: Self.greedySampling, seed: 0, bounds: .abc)
        let matches = zip(refResult.tokens, quantResult.tokens).filter { $0 == $1 }.count
        print("int4-all greedy_abc matches=\(matches)/8 ref=\(refResult.tokens) quant=\(quantResult.tokens)")
        #expect(matches >= 7)
    }

    // NAR-side int4-all parity (velocity/solve4 vs bf16-native) was measured, not committed as a
    // test: velocity rel=0.085, solve4 rel=0.161 — both well past the fiche's 5e-2/8e-2 budget,
    // unlike the AR side (logits rel=0.0075, greedy 8/8) or fp16 (E1, inaudible). The NAR solver
    // chains 64 velocity evaluations per chunk, so 4-bit quantization error compounds along the
    // ODE trajectory in a way a single AR forward pass never sees — a real accuracy limit, not a
    // bug (structural quantization coverage is verified above). See `tasks/ASK.md` (T-6.2): this
    // blocks the pack export and needs Vincent's call before proceeding.

    /// `dequantizeWeights(of: .nar)` swaps the 8-bit NAR projections for bf16 `Linear`s holding
    /// the same weights: the velocity field must match the packed path up to accumulation order,
    /// and the AR branch must stay quantized.
    @Test func dequantizedNARVelocityMatchesPackedPath() throws {
        let fixture = try loadFixture(name: "nar")
        let config = try YuE2Config.load(from: try lmDir().appendingPathComponent("config.json"))
        let model = try LMWeightLoader.load(directory: try lmDir(), config: config, quantization: .int4Mixed)
        let noise = try #require(fixture["noise"])
        let arTokens = try idArray(fixture, "ar_tokens")
        let nar = CachedNAR(model: model, chunk: NARChunk(arTokens: arTokens, noise: noise))
        let packed = nar.velocity(state: noise, rawT: 0.0)
        eval(packed)

        let added = model.dequantizeWeights(of: .nar)
        #expect(added > 1_000_000_000, "expected the 8-bit NAR branch to add > 1 GB once dequantized, got \(added)")
        for (path, module) in model.leafModules().flattened() {
            guard module is Linear else { continue } // q_norm/k_norm live under the same paths
            if path.contains("nar_self_attn") || path.contains("nar_mlp") {
                #expect(!(module is QuantizedLinear), "\(path) still quantized")
            } else if path.contains("self_attn") || path.contains(".mlp.") {
                #expect(module is QuantizedLinear, "\(path) lost its quantization")
            }
        }
        let dequantized = nar.velocity(state: noise, rawT: 0.0)
        let rel = MLX.mean(MLX.abs(dequantized - packed)).item(Float.self) / MLX.mean(MLX.abs(packed)).item(Float.self)
        // The dequantized weights are rounded to bf16 (scale · q + bias no longer fits 8 bits of
        // mantissa), which `quantizedMatmul` avoids by dequantizing in fp32 inside the kernel:
        // measured rel ≈ 1-3e-2 on the real fixture, inside E2's 5e-2 velocity budget.
        print("DEQUANT_NAR rel=\(rel)")
        #expect(rel < 5e-2, "rel \(rel)")
    }
}

