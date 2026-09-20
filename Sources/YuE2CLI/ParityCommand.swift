// ParityCommand.swift - compare Swift outputs against real_fixtures.py dumps (plan §8.2)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import YuE2Core

struct ParityCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "parity",
        abstract: "Compare Swift outputs against real_fixtures.py dumps.",
        subcommands: [VAEParityCommand.self, LMParityCommand.self, NARParityCommand.self]
    )
}

private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
}

private func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
    let num = MLX.mean(MLX.abs(a.asType(.float32) - b.asType(.float32))).item(Float.self)
    let den = MLX.mean(MLX.abs(b.asType(.float32))).item(Float.self)
    return num / den
}

struct VAEParityCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vae",
        abstract: "Compare YuE2VAE.decode/decodeTiled against real_fixtures.py's vae.safetensors."
    )

    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Max tolerated max|Δ|.")
    var toleranceAbs: Float = 2e-3

    @Option(name: .long, help: "Max tolerated mean|Δ| / mean|reference|.")
    var toleranceRel: Float = 1e-3

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let fixture = try loadArrays(url: dir.appendingPathComponent("parity/vae.safetensors"))
        let vae = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"))

        func require(_ key: String) throws -> MLXArray {
            guard let value = fixture[key] else {
                throw YuE2Error.missingFile("parity/vae.safetensors is missing key \(key)")
            }
            return value
        }

        var worstMax: Float = 0
        var worstRel: Float = 0
        func check(_ name: String, ours: MLXArray, reference: MLXArray) {
            let oursNCL = ours.transposed(0, 2, 1) // NLC -> torch's NCL layout
            let m = maxAbs(oursNCL, reference)
            let r = relativeError(oursNCL, reference)
            worstMax = max(worstMax, m)
            worstRel = max(worstRel, r)
            print("PARITY \(name): max=\(m) rel=\(r)")
        }

        let latent40 = try require("latent_40").transposed(0, 2, 1).asType(.float32) // torch NCL -> MLX NLC
        check("vae_40", ours: vae.decode(latent40), reference: try require("audio_40"))

        let latent1100 = try require("latent_1100").transposed(0, 2, 1).asType(.float32)
        let full1100 = vae.decode(latent1100)
        let tiled1100 = try vae.decodeTiled(latent1100, coreFrames: 1024)
        check("vae_1100_tiled_vs_full", ours: tiled1100, reference: full1100.transposed(0, 2, 1))
        check("vae_1100_tiled_vs_torch", ours: tiled1100, reference: try require("audio_1100_tiled"))

        // T-5.1: encoder parity, only run when the fixture has it (older `vae.safetensors`
        // generated before T-5.1 with `decoder_only=True` simply lack these two keys).
        if fixture["audio_encode"] != nil {
            let encoder = try YuE2VAEEncoder.load(directory: dir.appendingPathComponent("YuE2-Vae"))
            let audioEncode = try require("audio_encode").transposed(0, 2, 1).asType(.float32) // NCL -> NLC
            let encoded = try encoder.encode(audioEncode)
            check("vae_encode", ours: encoded, reference: try require("encoded_expected"))
        }

        guard worstMax <= toleranceAbs, worstRel <= toleranceRel else {
            print("PARITY FAILED vae max=\(worstMax) rel=\(worstRel)")
            throw ExitCode.failure
        }
        print("PARITY OK vae max=\(worstMax) rel=\(worstRel)")
    }
}

/// GATE G-2: `Scripts/reference/real_fixtures.py lm`'s prefixes run through this build's AR
/// backbone (full, unrestricted `lm_head` for the logits/bisection taps; the real O1
/// `RestrictedHead` path for the two 8-token greedy decodes — the actual production path).
struct LMParityCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "lm",
        abstract: "Compare the AR backbone (logits, layer bisection, greedy decode) against real_fixtures.py's lm.safetensors."
    )

    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Max tolerated mean|Δ| / mean|reference| on last-token logits and hidden taps.")
    var toleranceRel: Float = 2e-2

    private static let watchedLayers = [0, 7, 14, 27]

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let fixture = try loadArrays(url: dir.appendingPathComponent("parity/lm.safetensors"))
        let config = try YuE2Config.load(from: dir.appendingPathComponent("YuE2-3B/config.json"))
        let model = try LMWeightLoader.load(directory: dir.appendingPathComponent("YuE2-3B"), config: config)

        func require(_ key: String) throws -> MLXArray {
            guard let value = fixture[key] else {
                throw YuE2Error.missingFile("parity/lm.safetensors is missing key \(key)")
            }
            return value
        }
        func idArray(_ key: String) throws -> [Int] {
            try require(key).asType(.int32).asArray(Int32.self).map(Int.init)
        }

        var worstRel: Float = 0
        var allArgmaxEqual = true

        /// Full (unrestricted) `lm_head` logits for the last prefix token, plus — for `"abc"`
        /// only, since both prefixes share the same backbone — hidden-state bisection taps.
        func checkLogits(_ name: String, prefix: [Int], captureTaps: Bool) throws {
            let ids = MLXArray(prefix.map { Int32($0) }).reshaped([1, prefix.count])
            var taps: [Int: MLXArray] = [:]
            let logits = model.forwardAR(tokens: ids, cache: nil, keepLast: true) { i, h in
                if captureTaps { taps[i] = h }
            }
            let ours = logits.reshaped([1, config.vocabSize])
            let reference = try require("logits_last_\(name)").reshaped([1, config.vocabSize])
            let rel = relativeError(ours, reference)
            let argmaxEqual =
                MLX.argMax(ours, axis: -1).item(UInt32.self) == MLX.argMax(reference, axis: -1).item(UInt32.self)
            worstRel = max(worstRel, rel)
            allArgmaxEqual = allArgmaxEqual && argmaxEqual
            print("PARITY logits_\(name): rel=\(rel) argmax_equal=\(argmaxEqual)")

            for i in Self.watchedLayers {
                guard captureTaps, let got = taps[i] else { continue }
                let reference = try require("hidden_after_layer_\(i)")
                let r = relativeError(got, reference)
                worstRel = max(worstRel, r)
                print("PARITY hidden_layer_\(i): rel=\(r)")
            }
        }

        let prefixABC = try idArray("prefix_abc")
        let prefixSemantic = try idArray("prefix_semantic")
        try checkLogits("abc", prefix: prefixABC, captureTaps: true)
        try checkLogits("semantic", prefix: prefixSemantic, captureTaps: false)

        // Exactly the config that produced the fixture's greedy_* (T-2.9): argmax-only,
        // repetition penalty off, 8 tokens.
        let sampling = try Sampling(
            temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1,
            penaltyWindow: 1, minTokens: 0, maxTokens: 8)

        func checkGreedy(_ name: String, prefix: [Int], bounds: LogitsBounds) throws -> Int {
            let head = model.makeRestrictedHead(bounds: bounds)
            let result = try TokenGenerator.generate(
                model: model, head: head, prefix: prefix, sampling: sampling, seed: 0, bounds: bounds)
            let expected = try idArray("greedy_\(name)")
            let matches = zip(result.tokens, expected).filter { $0 == $1 }.count
            print("PARITY greedy_\(name): \(matches)/8 identical ours=\(result.tokens) reference=\(expected)")
            return matches
        }

        let greedyABC = try checkGreedy("abc", prefix: prefixABC, bounds: .abc)
        let greedySemantic = try checkGreedy("semantic", prefix: prefixSemantic, bounds: .semantic)

        let verdict =
            "lm rel=\(worstRel) argmax_equal=\(allArgmaxEqual) greedy_abc=\(greedyABC)/8 greedy_semantic=\(greedySemantic)/8"
        guard worstRel <= toleranceRel, allArgmaxEqual, greedyABC == 8, greedySemantic == 8 else {
            print("PARITY FAILED \(verdict)")
            throw ExitCode.failure
        }
        print("PARITY OK \(verdict)")
    }
}

/// Compares `CachedNAR.velocity`/`.solve` against `Scripts/reference/real_fixtures.py nar`'s
/// dump: three `velocity` evaluations, four bisection taps (`after_pos`, `after_layer_{0,7,14,27}`
/// — one raw value, 20.0), and a 4-step `solve`.
struct NARParityCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "nar",
        abstract: "Compare CachedNAR.velocity/.solve against real_fixtures.py's nar.safetensors."
    )

    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Max tolerated mean|Δ| / mean|reference| on velocity and bisection taps.")
    var toleranceVelocity: Float = 3e-2

    @Option(name: .long, help: "Max tolerated mean|Δ| / mean|reference| on the 4-step solve.")
    var toleranceSolve: Float = 5e-2

    private static let watchedLayers = [0, 7, 14, 27]

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let fixture = try loadArrays(url: dir.appendingPathComponent("parity/nar.safetensors"))
        let config = try YuE2Config.load(from: dir.appendingPathComponent("YuE2-3B/config.json"))
        let model = try LMWeightLoader.load(directory: dir.appendingPathComponent("YuE2-3B"), config: config)

        func require(_ key: String) throws -> MLXArray {
            guard let value = fixture[key] else {
                throw YuE2Error.missingFile("parity/nar.safetensors is missing key \(key)")
            }
            return value
        }

        let arTokens = try require("ar_tokens").asType(.int32).asArray(Int32.self).map(Int.init)
        let noise = try require("noise")
        let chunk = NARChunk(arTokens: arTokens, noise: noise)
        let nar = CachedNAR(model: model, chunk: chunk)

        var worstVelocityRel: Float = 0
        for raw in [20.0, 0.0, -2.3] {
            let velocity = nar.velocity(state: noise, rawT: raw)
            let reference = try require("velocity_raw\(raw)")
            let rel = relativeError(velocity, reference)
            worstVelocityRel = max(worstVelocityRel, rel)
            print("PARITY velocity_raw\(raw): rel=\(rel)")
        }

        var taps: [String: MLXArray] = [:]
        _ = nar.velocity(state: noise, rawT: 20.0) { name, value in taps[name] = value }
        func requireTap(_ name: String) throws -> MLXArray {
            guard let value = taps[name] else {
                throw YuE2Error.missingFile("velocity(rawT: 20.0) did not report tap \(name)")
            }
            return value
        }

        let posRel = relativeError(try requireTap("after_pos"), try require("nar_after_pos"))
        worstVelocityRel = max(worstVelocityRel, posRel)
        print("PARITY nar_after_pos: rel=\(posRel)")
        for i in Self.watchedLayers {
            let got = try requireTap("after_layer_\(i)")
            let reference = try require("nar_after_layer_\(i)")
            let rel = relativeError(got, reference)
            worstVelocityRel = max(worstVelocityRel, rel)
            print("PARITY nar_after_layer_\(i): rel=\(rel)")
        }

        let solved = try nar.solve(steps: 4)
        let solveRel = try relativeError(solved, require("solve4_expected"))
        print("PARITY solve4: rel=\(solveRel)")

        let verdict = "nar velocity_rel=\(worstVelocityRel) solve4_rel=\(solveRel)"
        guard worstVelocityRel <= toleranceVelocity, solveRel <= toleranceSolve else {
            print("PARITY FAILED \(verdict)")
            throw ExitCode.failure
        }
        print("PARITY OK \(verdict)")
    }
}
