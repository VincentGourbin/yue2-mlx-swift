// ABCScore.swift - reader for the native YuE2 ABC dialect (two voices, `% section` comments)
// Copyright 2026 Vincent Gourbin

import Foundation

/// A score in the dialect the YuE2 planner and SheetSage2 write: `X/T/M/L/Q` header, the two
/// voice declarations `V: Vocal` and `V: Ins`, `K:`, then `% section` comments followed by one
/// line per voice (`V: Vocal` / `V: Ins` switches), bars separated by `|`, `Zn` multi-bar rests,
/// `"Chord"` symbols, ties `-`. Durations are integer multiples of `L:`. Imposed scores (covers,
/// hummed themes) may change `M:`, `K:` or `Q:` on a line of their own inside a voice; each bar
/// keeps the meter it was written in (`L:` must not change).
///
/// Only what instrumental conversion and lyric timing need: per voice, the bars as written
/// (tokens kept verbatim so a bar can be moved without being re-spelled) and the sounding
/// notes with their onsets in `L` units.
public struct ABCScore: Equatable, Sendable {
    public struct Note: Equatable, Sendable {
        /// Onset inside the bar, in `L` units.
        public var onset: Int
        public var midi: Int
        public var duration: Int
        /// Written with a trailing `-` (tied into the next note of the same pitch).
        public var tied: Bool
    }

    public struct Bar: Equatable, Sendable {
        /// Tokens as written, spaces dropped (`"Gm7"`, `b4`, `z2`, `f2-`…).
        public var tokens: [String]
        public var length: Int
        public var notes: [Note]
        /// `(onset in L units, symbol)`.
        public var chords: [(Int, String)]
        /// Meter in force for this bar (`M:`, inline changes included).
        public var meter: (Int, Int) = (4, 4)
        public var isEmptyRest: Bool { notes.isEmpty && chords.isEmpty }

        public static func == (a: Bar, b: Bar) -> Bool {
            a.tokens == b.tokens && a.length == b.length && a.notes == b.notes && a.meter == b.meter
                && a.chords.map(\.0) == b.chords.map(\.0) && a.chords.map(\.1) == b.chords.map(\.1)
        }
    }

    /// A sounding note with ties merged (a held note is one note), positioned from the first bar.
    public struct Sounding: Equatable, Sendable {
        public var start: Int
        public var duration: Int
        public var midi: Int
        public var bar: Int
    }

    /// Header lines up to and including `K:`, voice declarations included, in order.
    public var header: [String]
    public var tempo: Double
    /// Denominator of `L:` (32 in `L:1/32`).
    public var unit: Int
    public var meter: (Int, Int)
    public var key: String
    /// `(first bar index, label)` of every `% section` comment.
    public var sections: [(Int, String)]
    public var vocal: [Bar]
    public var ins: [Bar]
    /// The music changes `M:`, `K:` or `Q:` inline (bars of different lengths are possible).
    public var hasInlineFields: Bool

    public var barUnits: Int { meter.0 * unit / meter.1 }
    /// `L` units in a quarter note.
    public var unitsPerQuarter: Double { Double(unit) / 4 }

    public static func == (a: ABCScore, b: ABCScore) -> Bool {
        a.header == b.header && a.tempo == b.tempo && a.unit == b.unit && a.meter == b.meter && a.key == b.key
            && a.sections.map(\.0) == b.sections.map(\.0) && a.sections.map(\.1) == b.sections.map(\.1)
            && a.vocal == b.vocal && a.ins == b.ins
    }

    public init(_ text: String) throws {
        var header: [String] = []
        var tempo = 120.0, unit = 8, meter = (4, 4), key = "C"
        var sections: [(Int, String)] = []
        var pendingSections: [String] = []
        var voices: [String: [Bar]] = ["Vocal": [], "Ins": []]
        var current: String?
        var hasInlineFields = false
        var liveMeter: (Int, Int)?, liveKey: String?
        for rawLine in text.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("% ") {
                pendingSections.append(String(line.dropFirst(2)).trimmingCharacters(in: .whitespaces))
                continue
            }
            if line.hasPrefix("V:") {
                let name = line.dropFirst(2).trimmingCharacters(in: .whitespaces).split(separator: " ").first.map(String.init) ?? ""
                if current == nil && (line.contains("clef") || line.contains("name=")) && !header.contains(where: { $0.hasPrefix("K:") }) {
                    header.append(line)
                    continue
                }
                guard voices[name] != nil else { throw YuE2Error.invalidRequest("ABC: unknown voice \(name)") }
                current = name
                if name == "Vocal", !pendingSections.isEmpty {
                    for label in pendingSections { sections.append((voices["Vocal"]!.count, label)) }
                    pendingSections = []
                }
                continue
            }
            if line.count >= 2, line.dropFirst().first == ":", let field = line.first, field.isLetter, field.isUppercase {
                let value = line.dropFirst(2).trimmingCharacters(in: .whitespaces)
                if current != nil {
                    // Inside the music: meter and key apply to the following bars; the header
                    // tempo stays the song's nominal one; a unit change would rescale every onset.
                    guard field != "L" else { throw YuE2Error.invalidRequest("ABC: inline unit changes (\(line)) are not supported") }
                    hasInlineFields = true
                    switch field {
                    case "M":
                        let parts = value.split(separator: "/").compactMap { Int($0) }
                        if parts.count == 2 { liveMeter = (parts[0], parts[1]) }
                    case "K": liveKey = value
                    default: break
                    }
                    continue
                }
                switch field {
                case "Q": tempo = Double(value.split(separator: "=").last ?? "") ?? tempo
                case "L": unit = Int(value.split(separator: "/").last ?? "") ?? unit
                case "M":
                    let parts = value.split(separator: "/").compactMap { Int($0) }
                    meter = parts.count == 2 ? (parts[0], parts[1]) : (4, 4)
                case "K": key = value
                default: break
                }
                header.append(line)
                continue
            }
            guard let voice = current else { throw YuE2Error.invalidRequest("ABC: music before any voice") }
            let barMeter = liveMeter ?? meter
            let bu = barMeter.0 * unit / barMeter.1
            let accidentals = try Self.keyAccidentals(liveKey ?? key)
            var body = Substring(line)
            if body.hasSuffix("|") { body = body.dropLast() }
            for chunk in body.split(separator: "|", omittingEmptySubsequences: false) {
                var bar = try Self.parseBar(String(chunk), keyAccidentals: accidentals, barUnits: bu)
                bar.meter = barMeter
                if bar.notes.isEmpty, bar.chords.isEmpty, bar.length > bu, bar.tokens.first?.hasPrefix("Z") == true {
                    for _ in 0..<(bar.length / bu) { voices[voice]!.append(Bar(tokens: ["Z"], length: bu, notes: [], chords: [], meter: barMeter)) }
                } else {
                    voices[voice]!.append(bar)
                }
            }
        }
        self.header = header
        self.tempo = tempo
        self.unit = unit
        self.meter = meter
        self.key = key
        self.sections = sections
        self.vocal = voices["Vocal"]!
        self.ins = voices["Ins"]!
        self.hasInlineFields = hasInlineFields
    }

    /// Onset of every bar in `L` units (bar lengths follow inline meter changes), plus the end
    /// of the last one. The longer voice decides, the other one fills in.
    public var barStarts: [Int] {
        var starts = [0]
        for b in 0..<max(vocal.count, ins.count) {
            let bar = b < vocal.count ? vocal[b] : ins[b]
            starts.append(starts[b] + bar.meter.0 * unit / bar.meter.1)
        }
        return starts
    }

    /// Meter of bar `b` (the header meter past the end).
    public func meter(ofBar b: Int) -> (Int, Int) {
        b < vocal.count ? vocal[b].meter : b < ins.count ? ins[b].meter : meter
    }

    /// Sounding notes of a voice, ties merged, onsets from the first bar.
    public static func sounding(_ bars: [Bar]) -> [Sounding] {
        var out: [Sounding] = []
        var open: Int?
        var base = 0
        for (index, bar) in bars.enumerated() {
            for note in bar.notes {
                if let o = open, out[o].midi == note.midi, out[o].start + out[o].duration == base + note.onset {
                    out[o].duration += note.duration
                } else {
                    out.append(Sounding(start: base + note.onset, duration: note.duration, midi: note.midi, bar: index))
                    open = out.count - 1
                }
                if !note.tied { open = nil }
            }
            base += bar.length
        }
        return out
    }

    // MARK: - Bars

    private static let steps: [Character: Int] = ["C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11]

    static func parseBar(_ text: String, keyAccidentals: [Character: Int], barUnits: Int) throws -> Bar {
        var bar = Bar(tokens: [], length: 0, notes: [], chords: [])
        var barAccidentals: [String: Int] = [:]
        let chars = Array(text)
        var i = 0
        var t = 0
        func digits() -> Int? {
            var j = i
            while j < chars.count, chars[j].isNumber { j += 1 }
            defer { i = j }
            return j > i ? Int(String(chars[i..<j])) : nil
        }
        while i < chars.count {
            let c = chars[i]
            let start = i
            if c == " " || c == "\t" {
                i += 1
                continue
            }
            if c == "\"" {
                guard let close = chars[(i + 1)...].firstIndex(of: "\"") else {
                    throw YuE2Error.invalidRequest("ABC: unterminated chord in \(text)")
                }
                bar.chords.append((t, String(chars[(i + 1)..<close])))
                i = close + 1
                bar.tokens.append(String(chars[start..<i]))
                continue
            }
            if c == "z" || c == "Z" {
                i += 1
                let n = digits() ?? 1
                bar.tokens.append(String(chars[start..<i]))
                if c == "Z" {
                    bar.length = n * barUnits
                    return bar
                }
                t += n
                continue
            }
            var alter: Int?
            while i < chars.count, "^_=".contains(chars[i]) {
                alter = (alter ?? 0) + (chars[i] == "^" ? 1 : chars[i] == "_" ? -1 : 0)
                if chars[i] == "=" { alter = 0 }
                i += 1
            }
            guard i < chars.count, let step = steps[Character(chars[i].uppercased())] else {
                throw YuE2Error.invalidRequest("ABC: unparsed text at \(String(chars[start...]).prefix(16)) in \(text)")
            }
            let letter = Character(chars[i].uppercased())
            var octave = chars[i].isLowercase ? 5 : 4
            i += 1
            while i < chars.count, chars[i] == "'" || chars[i] == "," {
                octave += chars[i] == "'" ? 1 : -1
                i += 1
            }
            let length = digits() ?? 1
            let tied = i < chars.count && chars[i] == "-"
            if tied { i += 1 }
            let slot = "\(letter)\(octave)"
            if let alter { barAccidentals[slot] = alter }
            let effective = barAccidentals[slot] ?? keyAccidentals[letter] ?? 0
            bar.notes.append(Note(onset: t, midi: 12 * (octave + 1) + step + effective, duration: length, tied: tied))
            bar.tokens.append(String(chars[start..<i]))
            t += length
        }
        bar.length = t
        return bar
    }

    // MARK: - Keys

    static func keyAccidentals(_ key: String) throws -> [Character: Int] {
        let name = key.split(separator: " ").first.map(String.init) ?? "C"
        let majors: [String: Int] = [
            "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
            "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
        ]
        let relative: [String: String] = [
            "Am": "C", "Em": "G", "Bm": "D", "F#m": "A", "C#m": "E", "G#m": "B", "D#m": "F#", "A#m": "C#",
            "Dm": "F", "Gm": "Bb", "Cm": "Eb", "Fm": "Ab", "Bbm": "Db", "Ebm": "Gb", "Abm": "Cb",
        ]
        var normalized = name.replacingOccurrences(of: "maj", with: "").replacingOccurrences(of: "min", with: "m")
        if normalized.hasSuffix("m"), let major = relative[normalized] { normalized = major }
        guard let fifths = majors[normalized] else { throw YuE2Error.invalidRequest("ABC: unsupported key \(key)") }
        var out: [Character: Int] = [:]
        if fifths >= 0 {
            for c in "FCGDAEB".prefix(fifths) { out[c] = 1 }
        } else {
            for c in "BEADGCF".prefix(-fifths) { out[c] = -1 }
        }
        return out
    }
}
