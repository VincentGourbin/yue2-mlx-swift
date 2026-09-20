// SongRequest.swift - a single song generation request (plan §2.1, protocol.py)
// Copyright 2026 Vincent Gourbin

import Foundation

/// A song generation request: style tags, lyrics, and generation knobs.
///
/// Mirrors `SongRequest` in `protocol.py`, including the `__post_init__` validation.
public struct SongRequest: Codable, Equatable, Sendable {
    public var style: String
    public var lyrics: String
    public var cot: CoTMode
    public var seed: Int
    public var abc: String?
    public var cfgScale: Double?
    public var id: String

    public init(
        style: String,
        lyrics: String,
        cot: CoTMode = .full,
        seed: Int = 831_001,
        abc: String? = nil,
        cfgScale: Double? = nil,
        id: String = "song"
    ) throws {
        guard seed >= 0 else {
            throw YuE2Error.invalidRequest("seed must be an integer in [0, 2**63)")
        }
        guard SongRequest.isFilenameSafeID(id) else {
            throw YuE2Error.invalidRequest("id must be a filename-safe identifier")
        }
        if let abc {
            guard cot != .off, !abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw YuE2Error.invalidRequest("External ABC requires nonempty text and cot=melody/full")
            }
        }
        if let cfgScale {
            guard cfgScale.isFinite, cfgScale >= 0, cfgScale <= 20 else {
                throw YuE2Error.invalidRequest("cfg_scale must be finite and in [0,20]")
            }
        }
        self.style = style
        self.lyrics = lyrics
        self.cot = cot
        self.seed = seed
        self.abc = abc
        self.cfgScale = cfgScale
        self.id = id
    }

    /// `re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", id)` and `id not in {".", ".."}`, ASCII-only.
    private static func isFilenameSafeID(_ id: String) -> Bool {
        guard id != ".", id != ".." else { return false }
        let scalars = Array(id.unicodeScalars)
        guard (1...180).contains(scalars.count) else { return false }
        func isAlnum(_ s: Unicode.Scalar) -> Bool {
            (s.value >= 48 && s.value <= 57) || (s.value >= 65 && s.value <= 90) || (s.value >= 97 && s.value <= 122)
        }
        func isAllowedRest(_ s: Unicode.Scalar) -> Bool {
            isAlnum(s) || s == "_" || s == "." || s == "-"
        }
        guard isAlnum(scalars[0]) else { return false }
        return scalars.dropFirst().allSatisfy(isAllowedRest)
    }

    /// Effective CFG scale: the provided value, else 1.01 for `off` and 1.0 otherwise.
    public var guidance: Double {
        if let cfgScale { return cfgScale }
        return cot == .off ? 1.01 : 1.0
    }

    /// `"{instruction}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"`.
    public func text() -> String {
        "\(cot.instruction)\n[Tags]\n\(style)\n[Lyrics]\n\(lyrics)\n"
    }

    enum CodingKeys: String, CodingKey {
        case style, lyrics, cot, seed, abc
        case cfgScale = "cfg_scale"
        case id
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let cotRaw = try container.decodeIfPresent(String.self, forKey: .cot) ?? CoTMode.full.rawValue
        guard let cot = CoTMode(rawValue: cotRaw) else {
            throw YuE2Error.invalidRequest("cot must be off, melody or full")
        }
        try self.init(
            style: container.decode(String.self, forKey: .style),
            lyrics: container.decode(String.self, forKey: .lyrics),
            cot: cot,
            seed: container.decodeIfPresent(Int.self, forKey: .seed) ?? 831_001,
            abc: container.decodeIfPresent(String.self, forKey: .abc),
            cfgScale: container.decodeIfPresent(Double.self, forKey: .cfgScale),
            id: container.decodeIfPresent(String.self, forKey: .id) ?? "song"
        )
    }
}
