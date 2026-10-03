// SheetSage2Tokenizer.swift - typed prompt/event vocabulary, schema v1 (plan/14-sheetsage2.md, S-4)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Port of `SheetSage2Tokenizer` (schema `v1`): contiguous token blocks whose bounds depend only
/// on the window length and the time resolution, plus the strict/lenient sequence decoder.
public struct SheetSage2Tokenizer: Sendable {
    public enum TokenType: String, Sendable {
        case pad, sos, eos, out, prompt
        case subbeatShift = "subbeat_shift"
        case time, meter
        case eighthPosition = "eighth_position"
        case structure, key
        case chordMajmin = "chord_majmin"
        case chordFull = "chord_full"
        case pitch, duration
    }

    public static let chromaticSharps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    public static let structureLabels = [
        "silence", "intro", "outro", "verse", "chorus", "bridge", "pre-chorus", "post-chorus",
        "interlude", "fade-out", "loop", "rap", "preshot", "irregular", "instrumental",
        "intro and verse", "pre-chorus and chorus", "verse and pre-chorus", "solo", "theme",
        "development", "variation", "pre-outro",
    ]
    public static let durationTemplates = [
        1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
        3072, 4096,
    ]
    /// Schema v1 tasks in canonical order: (name, output field).
    public static let tasks: [(name: String, field: String)] = [
        ("timestamp", "timestamp"), ("downbeat_meter", "rhythm"), ("structure", "structure"),
        ("key", "key"), ("chord_majmin", "chord"), ("chord_full", "chord"),
        ("melody_vocal", "melody"), ("melody_full", "melody"),
    ]
    public static let eventFieldOrder = ["timestamp", "rhythm", "structure", "key", "chord", "melody"]
    /// The upstream default `FULL_TASK_PROMPTS`.
    public static let fullTaskPrompts = ["timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_full"]

    static let fullChordQualities = [
        "maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "sus2", "sus4",
        "sus4(b7)", "maj6", "min6",
    ]
    static let fullChordInversions: [String: [String]] = [
        "maj": ["/2", "/3", "/5"], "min": ["/2", "/b3", "/5"], "maj7": ["/3", "/5", "/7"],
        "min7": ["/b3", "/5", "/b7"], "7": ["/3", "/5", "/b7"],
    ]
    public static let fullChordLabels: [String] = {
        var labels = ["N"]
        for quality in fullChordQualities {
            let inversions = fullChordInversions[quality] ?? []
            for root in chromaticSharps {
                for inversion in inversions + [""] { labels.append("\(root):\(quality)\(inversion)") }
            }
        }
        return labels
    }()
    public static let majminChordLabels = ["N"] + chromaticSharps.map { "\($0):maj" } + chromaticSharps.map { "\($0):min" }
    public static let meterPairs: [(Int, Int)] = (1...32).flatMap { n in [1, 2, 4, 8, 16, 32].map { (n, $0) } }

    public let timeHz: Int
    public let nTimeTokens: Int

    public let padToken = 0, sosToken = 1, eosToken = 2, outToken = 3
    public let promptTokenStart = 4
    public let promptTokenEnd: Int
    public let subbeatShiftTokenStart: Int, subbeatShiftTokenEnd: Int
    public let timeTokenStart: Int, timeTokenEnd: Int
    public let meterTokenStart: Int, meterTokenEnd: Int
    public let eighthPositionTokenStart: Int, eighthPositionTokenEnd: Int
    public let structureTokenStart: Int, structureTokenEnd: Int
    public let keyTokenStart: Int, keyTokenEnd: Int
    public let majminChordTokenStart: Int, majminChordTokenEnd: Int
    public let fullChordTokenStart: Int, fullChordTokenEnd: Int
    public let pitchTokenStart: Int, pitchTokenEnd: Int
    public let durationTokenStart: Int, durationTokenEnd: Int
    public let nTokens: Int

    public init(audioLengthSeconds: Double = 300, timeHz: Int = 100) {
        self.timeHz = timeHz
        nTimeTokens = Int((audioLengthSeconds * Double(timeHz)).rounded())
        promptTokenEnd = promptTokenStart + 256
        subbeatShiftTokenStart = promptTokenEnd
        subbeatShiftTokenEnd = subbeatShiftTokenStart + 257
        timeTokenStart = subbeatShiftTokenEnd
        timeTokenEnd = timeTokenStart + nTimeTokens
        meterTokenStart = timeTokenEnd
        meterTokenEnd = meterTokenStart + Self.meterPairs.count
        eighthPositionTokenStart = meterTokenEnd
        eighthPositionTokenEnd = eighthPositionTokenStart + 256
        structureTokenStart = eighthPositionTokenEnd
        structureTokenEnd = structureTokenStart + Self.structureLabels.count
        keyTokenStart = structureTokenEnd
        keyTokenEnd = keyTokenStart + 24
        majminChordTokenStart = keyTokenEnd
        majminChordTokenEnd = majminChordTokenStart + Self.majminChordLabels.count
        fullChordTokenStart = majminChordTokenEnd
        fullChordTokenEnd = fullChordTokenStart + Self.fullChordLabels.count
        pitchTokenStart = fullChordTokenEnd
        pitchTokenEnd = pitchTokenStart + 256
        durationTokenStart = pitchTokenEnd
        durationTokenEnd = durationTokenStart + Self.durationTemplates.count
        nTokens = durationTokenEnd
    }

    public func promptToken(_ name: String) -> Int? {
        Self.tasks.firstIndex { $0.name == name }.map { promptTokenStart + $0 }
    }

    /// `[sos, prompts in schema order, out]`.
    public func promptPrefix(_ prompts: [String]) -> [Int] {
        let ordered = Self.tasks.map(\.name).filter(prompts.contains)
        return [sosToken] + ordered.compactMap(promptToken) + [outToken]
    }

    public func tokenType(_ token: Int) -> TokenType? {
        let blocks: [(TokenType, Int, Int)] = [
            (.subbeatShift, subbeatShiftTokenStart, subbeatShiftTokenEnd),
            (.time, timeTokenStart, timeTokenEnd), (.meter, meterTokenStart, meterTokenEnd),
            (.eighthPosition, eighthPositionTokenStart, eighthPositionTokenEnd),
            (.structure, structureTokenStart, structureTokenEnd), (.key, keyTokenStart, keyTokenEnd),
            (.chordMajmin, majminChordTokenStart, majminChordTokenEnd),
            (.chordFull, fullChordTokenStart, fullChordTokenEnd),
            (.pitch, pitchTokenStart, pitchTokenEnd), (.duration, durationTokenStart, durationTokenEnd),
        ]
        for (type, start, end) in blocks where start <= token && token < end { return type }
        if token >= promptTokenStart && token < promptTokenStart + Self.tasks.count { return .prompt }
        switch token {
        case padToken: return .pad
        case sosToken: return .sos
        case eosToken: return .eos
        case outToken: return .out
        default: return nil
        }
    }

    func outputField(_ type: TokenType) -> String? {
        switch type {
        case .time: "timestamp"
        case .meter, .eighthPosition: "rhythm"
        case .structure: "structure"
        case .key: "key"
        case .chordMajmin, .chordFull: "chord"
        case .pitch, .duration: "melody"
        default: nil
        }
    }
}
