// Prefixes.swift - token prefix construction (plan §2.1, protocol.py)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Minimal text tokenizer surface needed to build prefixes. The real tokenizer
/// (T-1.5) will conform to this.
public protocol TextEncoding {
    func encode(_ text: String) -> [Int]
}

/// `TextEncoding` plus `decode`, needed wherever generated ids must become text again
/// (`Planner`'s ABC-generation branch, T-2.8).
public protocol TextDecoding: TextEncoding {
    func decode(_ ids: [Int]) -> String
}

/// Prefix construction for the AR phases, mirroring `protocol.py`'s free functions.
public enum Prefixes {
    /// The positive (conditional) prefix: ABC phase if `abcIDs` and `request.abc` are both
    /// absent, else the semantic phase (ABC ids embedded between `<abc>`/`</abc>`).
    public static func tokenPrefixes(
        request: SongRequest,
        tokenizer: TextEncoding,
        abcIDs: [Int]? = nil
    ) throws -> [Int] {
        let base = [YuE2Token.eod] + tokenizer.encode(request.text())
        if request.cot == .off {
            return base + [YuE2Token.abcStart, YuE2Token.abcEnd, YuE2Token.musicStart]
        }
        let ids: [Int]
        if let abcIDs {
            ids = abcIDs
        } else if let abc = request.abc {
            ids = tokenizer.encode(abc)
        } else {
            return base + [YuE2Token.abcStart]
        }
        try requireOrdinaryTextIDs(ids)
        return base + [YuE2Token.abcStart] + ids + [YuE2Token.abcEnd, YuE2Token.musicStart]
    }

    /// The negative (unconditional) prefix for classifier-free guidance.
    public static func negativePrefix(
        request: SongRequest,
        tokenizer: TextEncoding,
        abcIDs: [Int]? = nil
    ) throws -> [Int] {
        let base = [YuE2Token.eod] + tokenizer.encode(request.cot.instruction)
        if request.cot == .off {
            return base + [YuE2Token.musicStart]
        }
        guard let abcIDs else {
            throw YuE2Error.invalidRequest("Symbolic CFG must retain the exact positive-branch ABC IDs")
        }
        try requireOrdinaryTextIDs(abcIDs)
        return base + [YuE2Token.abcStart] + abcIDs + [YuE2Token.abcEnd, YuE2Token.musicStart]
    }

    private static func requireOrdinaryTextIDs(_ ids: [Int]) throws {
        guard ids.allSatisfy({ $0 >= 0 && $0 < YuE2Token.eod }) else {
            throw YuE2Error.invalidRequest("ABC IDs must remain inside the ordinary text vocabulary")
        }
    }

    /// Splits `frames` codec frames into acoustic-context-sized chunks that leave room for
    /// `prefixTokens` text tokens inside `context` positions.
    public static func chunkRanges(frames: Int, prefixTokens: Int, context: Int = YuE2Token.context) throws -> [(Int, Int)] {
        let size = min((context - prefixTokens - 3) / 2, YuE2Token.context)
        guard frames >= 1, size >= 1 else {
            throw YuE2Error.invalidRequest("Empty codec or prefix leaves no acoustic context")
        }
        var ranges: [(Int, Int)] = []
        var start = 0
        while start < frames {
            ranges.append((start, min(start + size, frames)))
            start += size
        }
        return ranges
    }
}
