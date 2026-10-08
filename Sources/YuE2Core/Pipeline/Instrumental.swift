// Instrumental.swift - instrumental songs: move the Vocal melody to Ins, keep chords, steer away from a voice
// Copyright 2026 Vincent Gourbin

import Foundation

/// Upstream's instrumental recipe (`skills/yue2-music/instrumental`): plan a score, move every
/// Vocal note to Ins (Vocal keeps only rests and chord symbols), then render that fixed score
/// with section-tag lyrics and a "no vocals" style. Measured on 2026-10-07
/// (docs/knowledge/benchmarks/instrumental-and-karaoke-2026-10-07.md): the transfer is what
/// matters (same score without it: a voice throughout); a negative CFG branch conditioned on
/// voice tags (`negativeStyle`, guidance 1.5) removed the remaining leak on the seed that
/// still hummed the verse.
public enum Instrumental {
    /// Default tags for the negative CFG branch.
    public static let negativeStyle = "vocals, singing, female voice, male voice, choir, humming, spoken words"
    /// Appended to the user's style when it does not already say so.
    public static let styleSuffix = "no vocals, no singing, no choir, no spoken words"
    /// Guidance used with `negativeStyle`.
    public static let guidance = 1.5

    public struct Transfer: Equatable, Sendable {
        public var abc: String
        public var movedBars: Int
        public var keptInsBars: Int
        /// Bars where both voices sounded: Vocal wins, the Ins material of the bar is dropped.
        public var overlapBars: [Int]
        public var vocalNotes: Int
    }

    /// Moves the Vocal melody into Ins, bar by bar (bars are moved verbatim, so accidentals and
    /// ties keep their meaning); Vocal keeps its chord symbols over rests. Vocal priority on
    /// overlap, as upstream's default. Throws if the result does not carry every Vocal note.
    public static func transfer(abc: String) throws -> Transfer {
        let score = try ABCScore(abc)
        guard score.vocal.count == score.ins.count, !score.vocal.isEmpty else {
            throw YuE2Error.invalidRequest("ABC: Vocal and Ins must have the same number of bars")
        }
        guard !score.sections.isEmpty else {
            throw YuE2Error.invalidRequest("ABC: no % section comment")
        }
        guard !score.hasInlineFields else {
            throw YuE2Error.invalidRequest("ABC: inline M:/K:/Q: changes are not supported by the instrumental transfer")
        }
        let bu = score.barUnits
        var vocal: [String] = [], ins: [String] = []
        var moved = 0, kept = 0, overlaps: [Int] = []
        for (index, (v, i)) in zip(score.vocal, score.ins).enumerated() {
            vocal.append(restBar(keeping: v.chords, barUnits: bu))
            let notesOnly = { (bar: ABCScore.Bar) in bar.tokens.filter { !$0.hasPrefix("\"") }.joined() }
            if !v.notes.isEmpty {
                if !i.notes.isEmpty { overlaps.append(index) } else { moved += 1 }
                ins.append(notesOnly(v))
            } else if !i.notes.isEmpty {
                kept += 1
                ins.append(notesOnly(i))
            } else {
                ins.append("Z")
            }
        }
        let bounds = score.sections.map(\.0) + [score.vocal.count]
        var lines = score.header
        for (k, (start, label)) in score.sections.enumerated() {
            let end = bounds[k + 1]
            guard start < end else { continue }
            lines += ["% \(label)", "V: Vocal", compress(Array(vocal[start..<end])), "V: Ins", compress(Array(ins[start..<end]))]
        }
        let text = lines.joined(separator: "\n") + "\n"
        let check = try ABCScore(text)
        let before = ABCScore.sounding(score.vocal)
        let after = Set(ABCScore.sounding(check.ins).map { [$0.start, $0.duration, $0.midi] })
        guard ABCScore.sounding(check.vocal).isEmpty, before.allSatisfy({ after.contains([$0.start, $0.duration, $0.midi]) }) else {
            throw YuE2Error.invalidRequest("ABC: the voice transfer did not preserve every Vocal note")
        }
        return Transfer(abc: text, movedBars: moved, keptInsBars: kept, overlapBars: overlaps, vocalNotes: before.count)
    }

    /// `[Intro]\n\n[Verse]…` from the score's section comments: the planner-only lyrics upstream
    /// renders an instrumental with (no sung words).
    public static func sectionTagLyrics(abc: String) -> String {
        abc.components(separatedBy: .newlines)
            .filter { $0.hasPrefix("% ") }
            .map { "[" + $0.dropFirst(2).trimmingCharacters(in: .whitespaces).capitalized + "]" }
            .joined(separator: "\n\n")
    }

    /// The user's style with the "no vocals" tags appended once.
    public static func style(_ style: String) -> String {
        style.lowercased().contains("no vocals") ? style : style + ", " + styleSuffix
    }

    /// The render request of an instrumental: transferred score, section tags as lyrics,
    /// instrumental style, negative voice branch at `guidance`.
    public static func renderRequest(from planned: SongRequest, plannedABC: String) throws -> (SongRequest, Transfer) {
        let moved = try transfer(abc: plannedABC)
        var request = try SongRequest(
            style: style(planned.style), lyrics: sectionTagLyrics(abc: moved.abc),
            cot: planned.cot == .off ? .full : planned.cot, seed: planned.seed, abc: moved.abc,
            cfgScale: planned.cfgScale ?? guidance, id: planned.id)
        request.negativeStyle = planned.negativeStyle ?? negativeStyle
        return (request, moved)
    }

    private static func restBar(keeping chords: [(Int, String)], barUnits: Int) -> String {
        guard !chords.isEmpty else { return "Z" }
        var out = "", t = 0
        for (at, symbol) in chords {
            if at > t { out += "z\(at - t)" }
            out += "\"\(symbol)\""
            t = at
        }
        if barUnits > t { out += "z\(barUnits - t)" }
        return out
    }

    /// Bars joined with `|`, runs of empty bars written `Z`/`Z2`…`Z4` as the native dialect does.
    private static func compress(_ bars: [String]) -> String {
        var out: [String] = [], run = 0
        func flush() {
            while run > 0 {
                let n = min(run, 4)
                out.append(n == 1 ? "Z" : "Z\(n)")
                run -= n
            }
        }
        for bar in bars {
            if bar == "Z" { run += 1; continue }
            flush()
            out.append(bar)
        }
        flush()
        return out.joined(separator: "|") + "|"
    }
}
