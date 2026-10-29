// ScoreLayout.swift - where a score sings: sections, bars, first and last sung bar (ASK Q16)
// Copyright 2026 Vincent Gourbin

import Foundation

extension ABCScore {
    /// The shape of a score read before rendering it: its sections with their bars and sung
    /// notes, where the voice first and last sings, and the nominal length. The planner's length
    /// and lead-in are not bounded by anything the request says (the prompt has no duration), so
    /// a caller that caps the render must read them here rather than infer them from the lyrics.
    ///
    /// Seconds are nominal: the header `Q:` tempo from bar 1. The render does not start exactly
    /// at bar 1 and drifts (`LyricTimeline`: ~3 s over a song); add a margin before cutting.
    public struct Layout: Codable, Equatable, Sendable {
        public struct Section: Codable, Equatable, Sendable {
            /// The `% label` comment as written (`intro`, `verse`, `chorus`…).
            public var label: String
            /// 0-based index of the section's first bar.
            public var firstBar: Int
            public var bars: Int
            /// Sounding Vocal notes starting in the section, ties merged.
            public var sungNotes: Int
        }

        public var tempo: Double
        public var bars: Int
        /// Nominal length of the whole score, in seconds.
        public var seconds: Double
        public var sections: [Section]
        /// 0-based bar of the first sung note; `nil` when the Vocal voice never sings.
        public var firstSungBar: Int?
        /// Nominal onset of the first sung note, in seconds.
        public var firstSungSeconds: Double?
        /// 0-based bar of the last sung note.
        public var lastSungBar: Int?
        /// Nominal end of the last sung note, in seconds.
        public var lastSungEndSeconds: Double?
    }

    /// The layout of `abc`, or of its complete lines when the last one was cut (a plan stopped by
    /// the ABC token cap can end inside a chord symbol); `nil` when neither parses.
    public static func layout(of abc: String) -> Layout? {
        if let score = try? ABCScore(abc) { return score.layout }
        guard !abc.hasSuffix("\n"), let cut = abc.lastIndex(of: "\n") else { return nil }
        return try? ABCScore(String(abc[...cut])).layout
    }

    public var layout: Layout {
        let starts = barStarts
        let sung = Self.sounding(vocal)
        let seconds = { (units: Int) in (Double(units) / unitsPerQuarter * 60 / tempo * 1000).rounded() / 1000 }
        let total = starts.count - 1
        var parts: [Layout.Section] = []
        for (k, (first, label)) in sections.enumerated() {
            let end = k + 1 < sections.count ? sections[k + 1].0 : total
            parts.append(Layout.Section(
                label: label, firstBar: first, bars: end - first,
                sungNotes: sung.filter { $0.bar >= first && $0.bar < end }.count))
        }
        // `barStarts` follows each bar's meter, `sounding` the bars' written lengths: a note sits
        // at its bar's metered start plus its onset in that bar (an underfull bar still lasts a full bar).
        let onset = { (n: Sounding) in starts[n.bar] + (n.start - vocal[..<n.bar].reduce(0) { $0 + $1.length }) }
        return Layout(
            tempo: tempo, bars: total, seconds: seconds(starts[total]), sections: parts,
            firstSungBar: sung.first?.bar, firstSungSeconds: sung.first.map { seconds(onset($0)) },
            lastSungBar: sung.last?.bar, lastSungEndSeconds: sung.last.map { seconds(onset($0) + $0.duration) })
    }

    /// `abc` with at most `leadIn` bars before the voice's first group and at most `tail` bars
    /// after its last one, by dropping whole groups (a `V: Vocal` line and the lines up to the
    /// next one) as written: no bar is re-spelled, so the result stays in the planner's dialect.
    /// The lead-in keeps its last groups (those leading into the voice), the tail its first ones;
    /// a `% label` left without groups goes too. A group holding an inline field (`M:`, `K:`, `Q:`),
    /// or whose two voices do not cover the same bars (an `Ins` line before its `Vocal` one, a
    /// voice missing), is never dropped: the lead-in or tail may then stay longer, the voices never
    /// shift against each other. A score that never sings comes back unchanged. A last line cut by
    /// the ABC token cap is dropped.
    ///
    /// The planner writes groups of four bars: `leadIn: 4` leaves a one-group intro, `leadIn: 2`
    /// none unless a group is that short. Render the result as an imposed score (`request.abc`).
    public static func trimming(_ abc: String, leadIn: Int? = nil, tail: Int? = nil) throws -> String {
        var abc = abc
        if (try? ABCScore(abc)) == nil, !abc.hasSuffix("\n"), let cut = abc.lastIndex(of: "\n") { abc = String(abc[...cut]) }
        var header: [String] = []
        var items: [(label: String?, lines: [String])] = []  // a `% label`, or a group's lines
        for line in abc.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let isVocal = trimmed.hasPrefix("V:") && trimmed.dropFirst(2).trimmingCharacters(in: .whitespaces).hasPrefix("Vocal")
                && !(trimmed.contains("clef") || trimmed.contains("name="))
            if items.isEmpty, !isVocal, !trimmed.hasPrefix("% ") {
                header.append(line)
            } else if trimmed.hasPrefix("% ") {
                items.append((line, []))
            } else if isVocal || items.last?.label != nil {
                items.append((nil, [line]))
            } else {
                items[items.count - 1].lines.append(line)
            }
        }
        let head = header.joined(separator: "\n") + "\n"
        var groups: [(item: Int, bars: Int, sings: Bool, pinned: Bool)] = []
        for (k, item) in items.enumerated() where item.label == nil {
            let music = item.lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            guard !music.isEmpty else { continue }
            let score = try ABCScore(head + music.joined(separator: "\n") + "\n")
            let inline = music.contains {
                let t = $0.trimmingCharacters(in: .whitespaces)
                return t.count >= 2 && t.dropFirst().first == ":" && t.first.map { $0.isUppercase && $0 != "V" } == true
            }
            groups.append((k, max(score.vocal.count, score.ins.count), !sounding(score.vocal).isEmpty,
                           inline || score.vocal.count != score.ins.count))
        }
        guard let first = groups.firstIndex(where: \.sings), let last = groups.lastIndex(where: \.sings) else { return abc }
        // Walking away from the voice, the group that crosses the budget goes, and every one beyond.
        var dropped = Set<Int>()
        func drop(_ order: [Int], budget: Int) {
            var kept = 0
            for (n, g) in order.enumerated() {
                kept += groups[g].bars
                guard kept > budget else { continue }
                for h in order[n...] where !groups[h].pinned { dropped.insert(groups[h].item) }
                return
            }
        }
        if let leadIn { drop(Array((0..<first).reversed()), budget: max(0, leadIn)) }
        if let tail { drop(Array((last + 1)..<groups.count), budget: max(0, tail)) }
        guard !dropped.isEmpty else { return abc }
        var out = header
        for (k, item) in items.enumerated() where !dropped.contains(k) {
            if item.label != nil {
                // A label survives only if one of its groups does.
                let next = items[(k + 1)...].firstIndex { $0.label != nil } ?? items.count
                guard ((k + 1)..<next).contains(where: { !dropped.contains($0) }) else { continue }
            }
            out += item.label.map { [$0] } ?? item.lines
        }
        let text = out.joined(separator: "\n")
        return abc.hasSuffix("\n") && !text.hasSuffix("\n") ? text + "\n" : text
    }
}
