// Options.swift - shared ArgumentParser option groups for plan/semantic (T-2.10)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

/// Where to find the checkpoints. Deliberately narrower than the fiche's sketch (which also
/// listed a `--vae` field): neither `plan` nor `semantic` touches the VAE, and `decode` already
/// has its own `--vae` option — a shared, unused flag on text-generation commands would be
/// confusing, not future-proofing.
struct ModelOptions: ParsableArguments {
    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .customLong("quant"), help: "AR quantization preset (O5, opt-in): none (default), qint8, int4. Wins over --reference.")
    var quantArgument: ModelSession.Quantization?

    @Flag(name: .long, help: "Also quantize lm_head under --quant (RestrictedHead dequantizes rows on demand). Wins over --reference.")
    var quantHead: Bool = false

    @Option(name: .customLong("precision"), help: "Compute precision for unquantized weights/activations: bf16 (default) or fp16 (E2). Wins over --reference.")
    var precisionArgument: YuE2ComputePrecision?

    var quant: ModelSession.Quantization { quantArgument ?? .none }
    var precision: YuE2ComputePrecision { precisionArgument ?? .bf16 }
}

/// What a command loads: a `--reference` profile's settings, then every flag the user wrote on top
/// of it (issue #5: flags were silently discarded under `--reference`). Each override is printed
/// with the profile's summary, so a run that is no longer the reference says so.
struct ModelSetup {
    var quant: ModelSession.Quantization
    var quantHead: Bool
    var precision: YuE2ComputePrecision
    var vaePrecision: VAEPrecision
    var vaeCoreFrames: Int?
    var releaseWeights: Bool
    var profile: YuE2ReferenceProfile?

    /// `vaePrecision` / `vaeCoreFrames` nil and `releaseWeights` false mean "not written".
    static func resolve(
        reference: String?, model: ModelOptions, vaePrecision: DecodeCommand.Precision?, defaultVAEPrecision: VAEPrecision,
        vaeCoreFrames: Int?, releaseWeights: Bool
    ) throws -> ModelSetup {
        var setup = ModelSetup(
            quant: model.quant, quantHead: model.quantHead, precision: model.precision,
            vaePrecision: vaePrecision?.vaePrecision ?? defaultVAEPrecision, vaeCoreFrames: vaeCoreFrames,
            releaseWeights: releaseWeights, profile: nil)
        guard let reference else { return setup }
        guard let profile = YuE2ReferenceProfile.named(reference) else {
            throw YuE2Error.invalidRequest("unknown reference profile \(reference); see `yue2 references`")
        }
        profile.applyGlobalPolicy()
        setup.profile = profile
        var overrides: [String] = []
        setup.quant = model.quantArgument ?? profile.quant
        if let q = model.quantArgument, q != profile.quant { overrides.append("--quant \(q.rawValue)") }
        setup.quantHead = model.quantHead || profile.quantizeHead
        if model.quantHead, !profile.quantizeHead { overrides.append("--quant-head") }
        setup.precision = model.precisionArgument ?? profile.precision
        if let p = model.precisionArgument, p != profile.precision { overrides.append("--precision \(p.rawValue)") }
        setup.vaePrecision = vaePrecision?.vaePrecision ?? profile.vaePrecision
        if let v = vaePrecision, v.vaePrecision != profile.vaePrecision { overrides.append("--vae-precision \(v.rawValue)") }
        setup.vaeCoreFrames = vaeCoreFrames ?? profile.vaeCoreFrames
        if let t = vaeCoreFrames, t != profile.vaeCoreFrames { overrides.append("--vae-core-frames \(t)") }
        setup.releaseWeights = releaseWeights || profile.releaseWeightsBetweenStages
        if releaseWeights, !profile.releaseWeightsBetweenStages { overrides.append("--release-weights-between-stages") }
        var line = "reference \(profile.id): \(profile.summary)"
        if !overrides.isEmpty { line += "\n  overridden by flags (not the reference any more): " + overrides.joined(separator: ", ") }
        FileHandle.standardError.write(Data((line + "\n").utf8))
        return setup
    }
}

/// `ModelSession.Quantization` (a `YuE2Quantization` alias) lives in `YuE2Core`, which has no
/// `ArgumentParser` dependency; the default `RawRepresentable where RawValue: ExpressibleByArgument`
/// conformance (`RawValue == String`) is picked up here instead, in the one target that imports both.
extension ModelSession.Quantization: ExpressibleByArgument {}
extension YuE2ComputePrecision: ExpressibleByArgument {}
extension YuE2ExecutionPolicy.NARCompute: ExpressibleByArgument {}
/// T-6.2 (E2) sensitivity sweep only, `yue2 parity nar --sweep-family`.
extension YuE2NARSweep.Family: ExpressibleByArgument {}
/// T-6.3 (E3): `--vae-backend`/`--backend` on `decode`/`parity vae`.
extension VAEBackendKind: ExpressibleByArgument {}
/// T-6.4 (E4): `--backend` on `parity nar`/`bench-coreai-nar`.
extension NARBackendKind: ExpressibleByArgument {}

/// Builds a `SongRequest` from `--request <json>` and/or individual overrides, mirroring
/// `cli.py`'s `generate()`: a request file is the base, then any flag present overrides that
/// one field.
struct RequestOptions: ParsableArguments {
    @Option(name: .long, help: "Path to a request JSON (style, lyrics, cot, seed, abc, cfg_scale, id).")
    var request: String?

    @Option(name: .long, help: "Style tags (overrides --request's style).")
    var style: String?

    @Option(name: .long, help: "Lyrics text (overrides --request's lyrics).")
    var lyrics: String?

    @Option(name: .long, help: "Chain-of-thought mode: off, melody, or full (overrides --request's cot).")
    var cot: String?

    @Option(name: .long, help: "Sampling seed (overrides --request's seed).")
    var seed: Int?

    @Option(name: .long, help: "Path to an external ABC score (overrides --request's abc).")
    var abcFile: String?

    @Option(name: .long, help: "Path to a beginning of ABC score the planner continues from — header with key and tempo, first sections (overrides --request's abc_prefix).")
    var abcPrefixFile: String?

    @Flag(name: .long, help: "Force the score header to the key and tempo the style states (\"key of D major\", \"120 BPM\"): the planner otherwise treats them as hints (ASK Q12). Ignored when a score or a prefix is imposed.")
    var followStyleKey = false

    @Option(name: .long, help: "Classifier-free guidance scale, in [0, 20] (overrides --request's cfg_scale).")
    var cfgScale: Double?

    @Option(name: .long, help: "EXPERIMENTAL: style tags for the negative CFG branch, e.g. \"vocals, singing\" for an instrumental; needs --cfg-scale > 1 (overrides --request's negative_style).")
    var negativeStyle: String?

    @Option(name: .long, help: "Filename-safe run id (overrides --request's id).")
    var id: String?

    func makeRequest() throws -> SongRequest {
        var style = self.style
        var lyrics = self.lyrics
        var cotRaw: String?
        var seed = self.seed
        var abc: String?
        var abcPrefix: String?
        var cfgScale = self.cfgScale
        var id = self.id
        var negativeStyle = self.negativeStyle

        if let request {
            let data = try Data(contentsOf: URL(fileURLWithPath: request))
            let base = try JSONDecoder().decode(SongRequest.self, from: data)
            style = style ?? base.style
            lyrics = lyrics ?? base.lyrics
            cotRaw = base.cot.rawValue
            seed = seed ?? base.seed
            abc = base.abc
            abcPrefix = base.abcPrefix
            cfgScale = cfgScale ?? base.cfgScale
            id = id ?? base.id
            negativeStyle = negativeStyle ?? base.negativeStyle
        }
        if let cot { cotRaw = cot }
        if let abcFile {
            abc = try String(contentsOf: URL(fileURLWithPath: abcFile), encoding: .utf8)
        }
        if let abcPrefixFile {
            abcPrefix = try String(contentsOf: URL(fileURLWithPath: abcPrefixFile), encoding: .utf8)
        }
        guard let style, let lyrics else {
            throw YuE2Error.invalidRequest("pass --request or both --style and --lyrics")
        }
        guard let cotMode = CoTMode(rawValue: cotRaw ?? CoTMode.full.rawValue) else {
            throw YuE2Error.invalidRequest("cot must be off, melody or full")
        }
        var request = try SongRequest(
            style: style, lyrics: lyrics, cot: cotMode, seed: seed ?? 831_001,
            abc: abc, abcPrefix: abcPrefix, cfgScale: cfgScale, id: id ?? "song")
        request.negativeStyle = negativeStyle
        return followStyleKey ? request.followingStyleKey() : request
    }
}

/// Per-phase sampling overrides. A field stays `nil` (falling back to the checkpoint-native
/// `GenerationConfig` default) unless the matching flag is passed.
struct SamplingOverrides: ParsableArguments {
    @Option(name: .long, help: "Max tokens for the ABC phase (yue2 plan).")
    var abcMaxTokens: Int?

    @Option(name: .long, help: "Min tokens before the ABC phase may stop (yue2 plan).")
    var abcMinTokens: Int?

    @Option(name: .long, help: "Max tokens for the semantic phase (yue2 semantic).")
    var semanticMaxTokens: Int?

    @Option(name: .long, help: "Min tokens before the semantic phase may stop (yue2 semantic).")
    var semanticMinTokens: Int?

    @Option(name: .long, help: "Sampling temperature (both phases).")
    var temperature: Double?

    @Option(name: .long, help: "Nucleus sampling threshold (both phases).")
    var topP: Double?

    @Option(name: .long, help: "Top-k cutoff (both phases).")
    var topK: Int?

    /// `nil` when nothing was overridden, so `resolveSampling` falls back to `base` unchanged.
    func abcSampling(default base: Sampling) throws -> Sampling? {
        guard abcMaxTokens != nil || abcMinTokens != nil || temperature != nil || topP != nil || topK != nil else {
            return nil
        }
        return try Sampling(
            temperature: temperature ?? base.temperature, topP: topP ?? base.topP, topK: topK ?? base.topK,
            repetitionPenalty: base.repetitionPenalty, penaltyWindow: base.penaltyWindow,
            minTokens: abcMinTokens ?? base.minTokens, maxTokens: abcMaxTokens ?? base.maxTokens)
    }

    func semanticSampling(default base: Sampling) throws -> Sampling? {
        guard semanticMaxTokens != nil || semanticMinTokens != nil || temperature != nil || topP != nil || topK != nil
        else {
            return nil
        }
        return try Sampling(
            temperature: temperature ?? base.temperature, topP: topP ?? base.topP, topK: topK ?? base.topK,
            repetitionPenalty: base.repetitionPenalty, penaltyWindow: base.penaltyWindow,
            minTokens: semanticMinTokens ?? base.minTokens, maxTokens: semanticMaxTokens ?? base.maxTokens)
    }
}
