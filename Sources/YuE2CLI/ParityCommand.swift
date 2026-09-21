// ParityCommand.swift - compare Swift outputs against real_fixtures.py dumps (plan §8.2)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import MLXRandom
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

/// `10*log10(Σref² / Σ(ref-test)²)`, in dB — higher is better, `+inf` for a bit-exact match.
private func snrDB(reference: MLXArray, test: MLXArray) -> Float {
    let ref32 = reference.asType(.float32)
    let signal = MLX.sum(ref32 * ref32).item(Float.self)
    let noise = MLX.sum(MLX.square(ref32 - test.asType(.float32))).item(Float.self)
    return 10 * log10(signal / noise)
}

/// Cosine similarity between two flattened arrays — same metric `Scripts/coreai/check_asset.py`
/// reports on the Python side, kept here so `yue2 parity vae --backend coreai-*`'s Swift-side
/// number is directly comparable.
private func cosineSimilarity(_ a: MLXArray, _ b: MLXArray) -> Float {
    let a32 = a.asType(.float32).reshaped([-1])
    let b32 = b.asType(.float32).reshaped([-1])
    let dot = MLX.sum(a32 * b32).item(Float.self)
    let normA = sqrt(MLX.sum(a32 * a32).item(Float.self))
    let normB = sqrt(MLX.sum(b32 * b32).item(Float.self))
    return dot / (normA * normB)
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

    @Option(name: .long, help: "Additionally measure fp16-vs-fp32 SNR (E1, plan/13-ios-backend.md).")
    var precision: DecodeCommand.Precision = .fp32

    @Option(name: .long, help: "Additionally measure a Core AI backend against this same MLX fp32 output (T-6.3, E3): mlx (skip, default), coreai-gpu, coreai-ane.")
    var backend: VAEBackendKind = .mlx

    @Option(name: .long, help: "Tile size (latent frames) for the Core AI tiled check (default 1024, the production tile; 256 is the iPhone tile).")
    var coreFrames: Int = 1024

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
        let full40 = vae.decode(latent40)
        check("vae_40", ours: full40, reference: try require("audio_40"))

        let latent1100 = try require("latent_1100").transposed(0, 2, 1).asType(.float32)
        let full1100 = vae.decode(latent1100)
        let tiled1100 = try vae.decodeTiled(latent1100, coreFrames: 1024)
        check("vae_1100_tiled_vs_full", ours: tiled1100, reference: full1100.transposed(0, 2, 1))
        check("vae_1100_tiled_vs_torch", ours: tiled1100, reference: try require("audio_1100_tiled"))

        // T-6.3 (E3): the same 1100-frame tiled decode, through Core AI instead of MLX, compared
        // against this run's own `tiled1100` (MLX fp32) — not the PyTorch fixture, same reasoning
        // as the fp16 check below. `--backend mlx` (default) skips this entirely.
        if backend != .mlx {
            let coreaiBackend = try await loadVAEBackend(backend, directory: dir.appendingPathComponent("YuE2-Vae"))

            // T-6.3 follow-up (Vincent's bounded plan, step 1): is the degradation seen on the
            // *tiled* 1100-frame decode (33.75 dB GPU) already present on the exact enumerated
            // shapes themselves (288/544/1056, no padding, no tiling — `frames == targetFrames`
            // in `CoreAIVAEDecoder.decode`, ruling out the zero-pad path entirely), or is it
            // specific to tiling? Each shape gets its own seeded random latent, decoded once by
            // MLX fp32 and once by the Core AI backend, compared directly (no PyTorch fixture
            // needed — this is backend-vs-backend, like the tiled check above).
            // Mirrors `CoreAIVAEDecoder.enumeratedFrames` (declared `#if canImport(CoreAI)`,
            // not referenceable unconditionally from this file).
            for frames in [288, 544, 1056] {
                let key = MLXRandom.key(UInt64(900_000 + frames))
                let exactLatent = MLXRandom.normal([1, frames, vae.config.decoderConfig.latentDim], key: key)
                let mlxExact = vae.decode(exactLatent)
                let coreaiExact = try await coreaiBackend.decode(exactLatent)
                let snrExact = snrDB(reference: mlxExact, test: coreaiExact)
                print("PARITY vae_coreai_exact_\(frames): snr=\(snrExact)")
            }

            let plan = planVAETiles(
                frames: 1100, coreFrames: coreFrames, haloFrames: 16,
                enumerated: coreaiBackend.enumeratedFrameCounts)
            for (i, w) in plan.enumerated() {
                print(
                    "PARITY vae_coreai_window_\(i): window=[\(w.windowStart),\(w.windowEnd)) frames=\(w.windowFrames)"
                        + " core=[\(w.coreStart),\(w.coreEnd)) padded=\(w.paddedFrames)")
            }
            let tileCount = plan.count
            let start = Date()
            let coreaiTiled = try await decodeTiled(using: coreaiBackend, latent1100, coreFrames: coreFrames)
            let elapsedMS = Date().timeIntervalSince(start) * 1000
            let snr = snrDB(reference: tiled1100, test: coreaiTiled)
            let cos = cosineSimilarity(tiled1100, coreaiTiled)
            let msPerTile = elapsedMS / Double(tileCount)
            let units = backend == .coreaiGPU ? "gpu" : "ane"
            print("PARITY vae_coreai: snr=\(snr) cos=\(cos) ms_per_tile=\(msPerTile) units=\(units)")

            // T-6.3 follow-up, step 2 (only informative if step 1 shows the exact shapes are
            // fine — i.e. the degradation is specific to tiling): per-tile-core SNR (are errors
            // uniform across a tile, or concentrated near the crop boundary?) and a ±128-sample
            // window straddling the one seam this 1100-frame/1024-core tiling has (at frame
            // 1024 -> sample 1024*1920). `naturalOutputLength`/`downsamplingRatio` mirror
            // `decodeTiled`'s own math so these slices land exactly on its tile boundaries.
            let ratio = vae.config.downsamplingRatio
            let total = vae.naturalOutputLength(frames: 1100)
            var worstTile: Float = .infinity
            for (i, w) in plan.enumerated() {
                let tileStart = w.coreStart * ratio
                let tileEnd = min(w.coreEnd * ratio, total)
                guard tileEnd > tileStart else { continue }
                let refSlice = tiled1100[0..., tileStart..<tileEnd, 0...]
                let testSlice = coreaiTiled[0..., tileStart..<tileEnd, 0...]
                let tileSNR = snrDB(reference: refSlice, test: testSlice)
                worstTile = min(worstTile, tileSNR)
                print("PARITY vae_coreai_tile_\(i): snr=\(tileSNR)")
            }
            var worstSeam: Float = .infinity
            for w in plan.dropFirst() {
                let seamSample = w.coreStart * ratio
                let seamStart = max(0, seamSample - 128)
                let seamEnd = min(total, seamSample + 128)
                guard seamEnd > seamStart else { continue }
                let refSeam = tiled1100[0..., seamStart..<seamEnd, 0...]
                let testSeam = coreaiTiled[0..., seamStart..<seamEnd, 0...]
                worstSeam = min(worstSeam, snrDB(reference: refSeam, test: testSeam))
            }
            print("PARITY vae_coreai_worst_tile: snr=\(worstTile) worst_seam_snr=\(worstSeam)")

            guard snr > 40 else {
                print("PARITY FAILED vae_coreai snr=\(snr) (threshold 40 dB)")
                throw ExitCode.failure
            }
        }

        // E1 (plan/13-ios-backend.md §13.5): is fp16 acceptable? A future Neural Engine backend
        // needs fp16 weights — measured here against this same fp32 model's own output, not the
        // PyTorch fixture (the question is fp16-vs-fp32, not fp16-vs-reference).
        if precision.vaePrecision == .fp16 {
            let vaeFP16 = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"), precision: .fp16)
            for (name, latent, reference) in [("40", latent40, full40), ("1100", latent1100, full1100)] {
                let fp16Out = vaeFP16.decode(latent)
                let snr = snrDB(reference: reference, test: fp16Out)
                let maxDiff = maxAbs(reference, fp16Out)
                print("PARITY vae_fp16_vs_fp32_\(name): snr=\(snr) max=\(maxDiff)")
            }
        }

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

    @Option(name: .long, help: "Quantization preset (T-4.5 AR-only, or T-6.2's -all presets).")
    var quant: ModelSession.Quantization = .none

    @Option(name: .long, help: "Compute precision for unquantized weights/activations (T-6.2, E2).")
    var precision: YuE2ComputePrecision = .bf16

    private static let watchedLayers = [0, 7, 14, 27]

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let fixture = try loadArrays(url: dir.appendingPathComponent("parity/lm.safetensors"))
        let config = try YuE2Config.load(from: dir.appendingPathComponent("YuE2-3B/config.json"))
        let model = try LMWeightLoader.load(
            directory: dir.appendingPathComponent("YuE2-3B"), config: config, quantization: quant)
        model.applyPrecision(precision)

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

    @Option(name: .long, help: "Quantization preset (T-4.5 AR-only, or T-6.2's -all presets).")
    var quant: ModelSession.Quantization = .none

    @Option(name: .long, help: "Compute precision for unquantized weights/activations (T-6.2, E2).")
    var precision: YuE2ComputePrecision = .bf16

    @Option(name: .long, help: "T-6.2 (E2) sensitivity sweep: quantize only this NAR projection family (rest of NAR stays int4 g64); omit family with --sweep-mode for an all-NAR run. Overrides --quant.")
    var sweepFamily: YuE2NARSweep.Family?

    @Option(name: .long, help: "Sweep-only: quantization mode for the targeted slice (affine or mxfp8). Requires --sweep-mode or --sweep-family to activate the sweep.")
    var sweepMode: String?

    @Option(name: .long, help: "Sweep-only: bits for the targeted slice.")
    var sweepBits: Int = 8

    @Option(name: .long, help: "Sweep-only: group size for the targeted slice.")
    var sweepGroupSize: Int = 64

    @Option(name: .long, help: "Additionally measure a Core AI NAR backend against this same real fixture (T-6.4, E4): mlx (skip, default), coreai-gpu, coreai-ane.")
    var backend: NARBackendKind = .mlx

    private static let watchedLayers = [0, 7, 14, 27]

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let fixture = try loadArrays(url: dir.appendingPathComponent("parity/nar.safetensors"))
        let config = try YuE2Config.load(from: dir.appendingPathComponent("YuE2-3B/config.json"))
        let model: YuE2ForCausalLM
        if sweepFamily != nil || sweepMode != nil {
            guard let modeString = sweepMode, let mode = QuantizationMode(rawValue: modeString) else {
                throw YuE2Error.invalidRequest("--sweep-mode required (affine|mxfp4|mxfp8|nvfp4) when sweeping")
            }
            model = try LMWeightLoader.load(directory: dir.appendingPathComponent("YuE2-3B"), config: config, quantization: .none)
            YuE2NARSweep.apply(to: model, family: sweepFamily, bits: sweepBits, mode: mode, groupSize: sweepGroupSize)
        } else {
            model = try LMWeightLoader.load(
                directory: dir.appendingPathComponent("YuE2-3B"), config: config, quantization: quant)
        }
        model.applyPrecision(precision)

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

        // T-6.4 (E4): the same three `velocity` evaluations and 4-step solve, through Core AI
        // instead of MLX, against this same real fixture -- unlike `parity vae --backend`
        // (informational only), this fiche's own validation runs this command once per backend
        // and expects each run to print its own `PARITY OK/FAILED`, so when `--backend` selects
        // Core AI, *that* backend's numbers gate the verdict below, not the MLX ones above.
        var gatedVelocityRel = worstVelocityRel
        var gatedSolveRel = solveRel
        if backend != .mlx {
            guard let coreaiBackend = try await loadNARBackend(
                backend, modelsDir: dir, cache: nar.arPrefixCache,
                arLength: nar.chunkARLength, narLength: nar.chunkNARLength)
            else {
                throw YuE2Error.backendUnavailable("loadNARBackend returned nil for a non-mlx backend")
            }
            var worstCoreaiRel: Float = 0
            for raw in [20.0, 0.0, -2.3] {
                let velocity = try await coreaiBackend.velocity(state: noise, rawT: raw)
                let reference = try require("velocity_raw\(raw)")
                let rel = relativeError(velocity, reference)
                worstCoreaiRel = max(worstCoreaiRel, rel)
                print("PARITY velocity_raw\(raw)_coreai: rel=\(rel)")
            }
            let solvedCoreai = try await nar.solve(steps: 4, backend: coreaiBackend)
            let solveRelCoreai = try relativeError(solvedCoreai, require("solve4_expected"))
            print("PARITY solve4_coreai: rel=\(solveRelCoreai)")
            gatedVelocityRel = worstCoreaiRel
            gatedSolveRel = solveRelCoreai
        }

        // T-6.2 (E2 sensitivity sweep): the production step count is 32, and a quantized NAR's
        // error compounds over all 64 velocity() calls (32 steps x 2 midpoint evals) — solve4
        // alone under- or over-states a quantization preset's real-world impact. Informational
        // only (never gates PARITY OK/FAILED below): older fixtures lack this key.
        if let solve32Ref = fixture["solve32_expected"] {
            let solved32 = try nar.solve(steps: 32)
            let rel32 = relativeError(solved32, solve32Ref)
            let snr32 = snrDB(reference: solve32Ref, test: solved32)
            print("PARITY nar_solve32: rel=\(rel32) snr=\(snr32)")

            let vae = try YuE2VAE.load(directory: dir.appendingPathComponent("YuE2-Vae"))
            let refAudio = vae.decode(solve32Ref.expandedDimensions(axis: 0))
            let testAudio = vae.decode(solved32.expandedDimensions(axis: 0))
            let audioSNR = snrDB(reference: refAudio, test: testAudio)
            print("PARITY nar_solve32_audio: snr=\(audioSNR)")
        }

        let verdict = "nar velocity_rel=\(gatedVelocityRel) solve4_rel=\(gatedSolveRel)"
        guard gatedVelocityRel <= toleranceVelocity, gatedSolveRel <= toleranceSolve else {
            print("PARITY FAILED \(verdict)")
            throw ExitCode.failure
        }
        print("PARITY OK \(verdict)")
    }
}
