// Constants.swift - protocol constants and native prompt instructions (plan §2.1)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Reserved token identifiers, copied verbatim from `protocol.py`.
///
/// These are the only magic numbers allowed outside `Constants.swift`; every
/// other file must reference them through this namespace (plan §2.1, trap n°6).
public enum YuE2Token {
    /// `<|endoftext|>` — first token of every prefix; exclusive upper bound of text tokens.
    public static let eod = 151643
    /// `<abc>` / `</abc>`.
    public static let abcStart = 151847
    public static let abcEnd = 151848
    /// Start / end of the semantic (codec) token span.
    public static let musicStart = 151851
    public static let musicEnd = 151852
    /// Semantic token `c` ∈ [0, 32768) ↔ id `codecOffset + c`; 1 token = 1 latent frame = 40 ms (25 Hz).
    public static let codecOffset = 151853
    public static let codecSize = 32768
    /// Reserved ids (NAR positions) — never sampled.
    public static let latentStart = 184621
    public static let latentEnd = 184622
    public static let latentPad = 184623
    /// Size of `embed_tokens` and `lm_head`.
    public static let vocabSize = 184704
    /// `max_position_embeddings` — never exceeded, never silently truncated.
    public static let context = 24576

    /// Protocol version string, matching `PROTOCOL_VERSION` in the reference.
    public static let protocolVersion = "yue2-native-v1"
}

/// Chain-of-thought mode selecting the native prompt instruction.
public enum CoTMode: String, Codable, CaseIterable, Sendable {
    case off
    case melody
    case full

    /// Exact native instruction for this mode, in the reference's key order.
    public var instruction: String {
        switch self {
        case .off:
            return "Generate music with codec tokens from the given conditions."
        case .melody:
            return "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
        case .full:
            return "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
        }
    }
}
