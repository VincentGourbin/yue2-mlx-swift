// Sampling.swift - sampling hyperparameters and generation defaults (plan §2.1, protocol.py)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Sampling hyperparameters for one generation phase (`abc` or `semantic`).
///
/// Mirrors `Sampling` in `protocol.py`: same fields, same defaults, same bounds
/// checked in `init` (there is no way to build an out-of-range value).
public struct Sampling: Codable, Equatable, Sendable {
    public var temperature: Double
    public var topP: Double
    public var topK: Int
    public var repetitionPenalty: Double
    public var penaltyWindow: Int
    public var minTokens: Int
    public var maxTokens: Int

    public init(
        temperature: Double = 1.0,
        topP: Double = 0.95,
        topK: Int = 100,
        repetitionPenalty: Double = 1.2,
        penaltyWindow: Int = 50,
        minTokens: Int = 200,
        maxTokens: Int = 9000
    ) throws {
        guard temperature.isFinite, topP.isFinite, repetitionPenalty.isFinite else {
            throw YuE2Error.invalidRequest("Sampling numbers must be finite")
        }
        guard temperature >= 0, temperature <= 5, topP > 0, topP <= 1, topK >= 1 else {
            throw YuE2Error.invalidRequest("Invalid sampling temperature/top_p/top_k")
        }
        guard repetitionPenalty > 0, (1...100).contains(penaltyWindow) else {
            throw YuE2Error.invalidRequest("Invalid repetition penalty/window")
        }
        guard minTokens >= 0, minTokens <= maxTokens, maxTokens >= 1 else {
            throw YuE2Error.invalidRequest("Require 0 <= min_tokens <= max_tokens")
        }
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.penaltyWindow = penaltyWindow
        self.minTokens = minTokens
        self.maxTokens = maxTokens
    }

    enum CodingKeys: String, CodingKey {
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case penaltyWindow = "penalty_window"
        case minTokens = "min_tokens"
        case maxTokens = "max_tokens"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            temperature: container.decodeIfPresent(Double.self, forKey: .temperature) ?? 1.0,
            topP: container.decodeIfPresent(Double.self, forKey: .topP) ?? 0.95,
            topK: container.decodeIfPresent(Int.self, forKey: .topK) ?? 100,
            repetitionPenalty: container.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? 1.2,
            penaltyWindow: container.decodeIfPresent(Int.self, forKey: .penaltyWindow) ?? 50,
            minTokens: container.decodeIfPresent(Int.self, forKey: .minTokens) ?? 200,
            maxTokens: container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 9000
        )
    }

    /// Merges a partial JSON override (only the keys present) onto `base`, then re-validates.
    init(merging container: KeyedDecodingContainer<CodingKeys>, into base: Sampling) throws {
        try self.init(
            temperature: container.decodeIfPresent(Double.self, forKey: .temperature) ?? base.temperature,
            topP: container.decodeIfPresent(Double.self, forKey: .topP) ?? base.topP,
            topK: container.decodeIfPresent(Int.self, forKey: .topK) ?? base.topK,
            repetitionPenalty: container.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? base.repetitionPenalty,
            penaltyWindow: container.decodeIfPresent(Int.self, forKey: .penaltyWindow) ?? base.penaltyWindow,
            minTokens: container.decodeIfPresent(Int.self, forKey: .minTokens) ?? base.minTokens,
            maxTokens: container.decodeIfPresent(Int.self, forKey: .maxTokens) ?? base.maxTokens
        )
    }
}

/// The NAR flow-matching integrator. `midpoint` is the reference (`nar.py`). `ab2` is an
/// extension of this port (2026-11-21, after Riff/audio.cpp and YuE2-Studio): Adams–Bashforth 2,
/// a midpoint first step and then one velocity evaluation per step extrapolated with the previous
/// one, `x ← x − dt·(3/2·v_n − 1/2·v_{n−1})` — `steps + 1` evaluations instead of `2·steps`.
public enum ODEMethod: String, CaseIterable, Sendable {
    case midpoint
    case ab2

    /// Velocity evaluations one solve of `steps` steps costs from `startStep`, from a cold start:
    /// an AB2 resume that carries its checkpoint's velocity skips the bootstrap (one fewer).
    public func evaluations(steps: Int, startStep: Int = 0) -> Int {
        let remaining = Swift.max(0, steps - startStep)
        switch self {
        case .midpoint: return 2 * remaining
        case .ab2: return remaining == 0 ? 0 : remaining + 1
        }
    }
}

/// Versioned generation defaults, checkpoint-native (`yue2_generation_config.json`).
///
/// Mirrors `GenerationConfig` in `protocol.py`.
public struct GenerationConfig: Codable, Equatable, Sendable {
    public var abc: Sampling
    public var semantic: Sampling
    public var odeSteps: Int
    /// `"midpoint"` (the reference, 2 velocity evaluations per step) or `"ab2"` (this port's
    /// extension, `ODEMethod.ab2`: steps + 1 evaluations).
    public var odeMethod: String
    public var context: Int

    /// `odeMethod` as the typed integrator (validated at init).
    public var method: ODEMethod { ODEMethod(rawValue: odeMethod) ?? .midpoint }
    public var version: String

    /// `Sampling(.7, .9, 30, 1.005, 100, 32, 4096)` in the reference.
    public static let defaultAbc = try! Sampling(
        temperature: 0.7, topP: 0.9, topK: 30, repetitionPenalty: 1.005,
        penaltyWindow: 100, minTokens: 32, maxTokens: 4096
    )
    public static let defaultSemantic = try! Sampling()

    public init(
        abc: Sampling = GenerationConfig.defaultAbc,
        semantic: Sampling = GenerationConfig.defaultSemantic,
        odeSteps: Int = 32,
        odeMethod: String = "midpoint",
        context: Int = YuE2Token.context,
        version: String = YuE2Token.protocolVersion
    ) throws {
        guard context == YuE2Token.context, ODEMethod(rawValue: odeMethod) != nil, odeSteps >= 1 else {
            throw YuE2Error.invalidRequest("Require context=24576, ode_method midpoint or ab2, and positive integer steps")
        }
        self.abc = abc
        self.semantic = semantic
        self.odeSteps = odeSteps
        self.odeMethod = odeMethod
        self.context = context
        self.version = version
    }

    enum CodingKeys: String, CodingKey {
        case abc, semantic
        case odeSteps = "ode_steps"
        case odeMethod = "ode_method"
        case context, version
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let abc: Sampling
        if container.contains(.abc) {
            let nested = try container.nestedContainer(keyedBy: Sampling.CodingKeys.self, forKey: .abc)
            abc = try Sampling(merging: nested, into: GenerationConfig.defaultAbc)
        } else {
            abc = GenerationConfig.defaultAbc
        }
        let semantic: Sampling
        if container.contains(.semantic) {
            let nested = try container.nestedContainer(keyedBy: Sampling.CodingKeys.self, forKey: .semantic)
            semantic = try Sampling(merging: nested, into: GenerationConfig.defaultSemantic)
        } else {
            semantic = GenerationConfig.defaultSemantic
        }
        try self.init(
            abc: abc,
            semantic: semantic,
            odeSteps: container.decodeIfPresent(Int.self, forKey: .odeSteps) ?? 32,
            odeMethod: container.decodeIfPresent(String.self, forKey: .odeMethod) ?? "midpoint",
            context: container.decodeIfPresent(Int.self, forKey: .context) ?? YuE2Token.context,
            version: container.decodeIfPresent(String.self, forKey: .version) ?? YuE2Token.protocolVersion
        )
    }

    /// Decodes `yue2_generation_config.json` (or any checkpoint-native generation config).
    public static func load(from url: URL) throws -> GenerationConfig {
        try JSONDecoder().decode(GenerationConfig.self, from: Data(contentsOf: url))
    }
}

/// Resolves an optional caller-provided `Sampling` against a phase default, mirroring
/// `resolve_sampling` in `protocol.py` (the `None` and `Sampling` branches; the reference's
/// third branch — a partial dictionary of overrides — is handled by `GenerationConfig`'s
/// own JSON decoding, the only place partial overrides originate in this port).
public func resolveSampling(_ overrides: Sampling?, default fallback: Sampling) -> Sampling {
    overrides ?? fallback
}
