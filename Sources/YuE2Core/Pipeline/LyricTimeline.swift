// LyricTimeline.swift - karaoke timing: lyric lines, words and syllables placed on the sung notes of the score
// Copyright 2026 Vincent Gourbin

import Foundation

/// Lyric timing for a generated song, from what the pipeline already has: the planned score's
/// Vocal voice (ties merged, so a held note is one note and never starts a syllable) placed on
/// a clock of the rendered audio — the LM's alignment heads (`AttentionTimeline`, the default)
/// or the beat grid SheetSage2 transcribes — and the lyrics, syllabified and aligned to the
/// notes by a monotonic DP (melismas preferred on the last syllable of a line). Word-start
/// anchors (alignment head, forced aligner) pull each word to the right note; a word too far
/// from its anchor leaves the score and follows the anchor.
///
/// Measured on three songs (EN ballad, FR chanson, EN 128 BPM; 2026-10-07), word starts against
/// an MMS forced alignment: score + grid alone 81–96 % within 300 ms, line starts median 38–75 ms;
/// with MMS anchors 98–100 %. The nominal `Q:` tempo alone is unusable (the render does not
/// start at bar 1 and drifts: ~3 s off). Line-final held notes end 0.1–0.3 s before the voice
/// does (the render holds into the rest).
///
/// Everything a player needs is precomputed and sorted by time: `bars` (downbeat and beat times),
/// `notes` (the sung notes, ties merged) and `lines` → `words` → `syllables`, each carrying both
/// its seconds and its musical position (`bar` 0-based in the score, `beat` 1-based within the
/// bar, fractional: 2.5 is the "and" of beat 2). A player binary-searches these arrays at the
/// playback time; nothing is recomputed.
public struct LyricTimeline: Codable, Equatable, Sendable {
    /// Format version of the encoded file.
    public static let formatVersion = 1

    public struct Syllable: Codable, Equatable, Sendable {
        public var text: String
        public var start: Double
        public var end: Double
        /// Score notes sung on this syllable (> 1: melisma).
        public var notes: Int
        /// Index in `LyricTimeline.notes` of the first note sung on this syllable (-1: off the score).
        public var firstNote: Int
        public var bar: Int
        public var beat: Double
    }

    public struct Word: Codable, Equatable, Sendable {
        public var text: String
        public var start: Double
        public var end: Double
        public var bar: Int
        public var beat: Double
        /// False when the word is timed by the voice alone (it does not sit on the score's notes).
        public var onScore: Bool
        public var syllables: [Syllable]
    }

    public struct Line: Codable, Equatable, Sendable {
        public var text: String
        public var section: String?
        public var start: Double
        public var end: Double
        public var bar: Int
        public var beat: Double
        public var words: [Word]
    }

    /// A bar of the score: its downbeat and the start of each of its beats, in seconds.
    public struct Bar: Codable, Equatable, Sendable {
        public var index: Int
        public var start: Double
        public var beats: [Double]
        /// Score section opening at this bar (`% verse` comment), if any.
        public var section: String?
    }

    /// A sung note of the score (Vocal voice, tied chunks merged into one note).
    public struct Note: Codable, Equatable, Sendable {
        public var start: Double
        public var end: Double
        public var midi: Int
        public var bar: Int
        public var beat: Double
        /// Written length in beats.
        public var beats: Double
    }

    /// A point of the rendered audio's beat grid: global 16th-note index → seconds.
    public struct GridPoint: Codable, Equatable, Sendable {
        public var subbeat: Int
        public var time: Double
        public init(subbeat: Int, time: Double) {
            self.subbeat = subbeat
            self.time = time
        }
    }

    /// A melody onset heard in the rendered audio (any voice), used to find where bar 1 falls.
    public struct HeardNote: Codable, Equatable, Sendable {
        public var subbeat: Int
        public var midi: Int
        public init(subbeat: Int, midi: Int) {
            self.subbeat = subbeat
            self.midi = midi
        }
    }

    public enum Language: String, Codable, Sendable {
        case english = "en", french = "fr"

        /// French when the style says so, else English (the two syllabifiers we have).
        public static func guess(style: String) -> Language {
            style.lowercased().contains("french") ? .french : .english
        }
    }

    public var version: Int
    /// Score tempo (`Q:`, quarters per minute) and meter (`M:`), for display.
    public var tempo: Double
    public var meter: [Int]
    /// Seconds of audio the timeline covers (bars, notes and lines past it are dropped).
    public var duration: Double?
    public var bars: [Bar]
    public var notes: [Note]
    public var lines: [Line]
    /// Clock the score was placed on: "attention" (the LM's alignment heads), "beats" (a
    /// SheetSage2 transcription of the audio) or "nominal" (score tempo from t = 0).
    public var grid: String
    /// Where bar 1 of the score falls on the grid, in 16ths.
    public var gridOffset: Int
    /// Share of heard onsets that match the planned notes at that offset (render follows score).
    public var gridMatch: Double?
    public var alignmentCost: Double

    // MARK: - Files

    /// `timeline.json` as the engine writes it (pretty, sorted keys).
    public func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> LyricTimeline {
        try JSONDecoder().decode(LyricTimeline.self, from: Data(contentsOf: url))
    }

    // MARK: - Playback

    /// Where the song is at a playback time, for a player: indices into `bars`, `notes`, `lines`
    /// and the current line's `words` / word's `syllables`. Each is the last item started at
    /// `time` (a line stays current until the next one starts); `note` only while it sounds.
    public struct Position: Equatable, Sendable {
        public var bar: Int?
        /// Fractional beat in `bar` (1-based: 2.5 is the "and" of beat 2).
        public var beat: Double?
        public var note: Int?
        public var line: Int?
        public var word: Int?
        public var syllable: Int?
    }

    /// Binary searches of the timeline at `time` (seconds of audio).
    public func position(at time: Double) -> Position {
        func lastStarted<T>(_ items: [T], _ start: (T) -> Double) -> Int? {
            var lo = 0, hi = items.count - 1, found: Int?
            while lo <= hi {
                let mid = (lo + hi) / 2
                if start(items[mid]) <= time { found = mid; lo = mid + 1 } else { hi = mid - 1 }
            }
            return found
        }
        var p = Position()
        if let b = lastStarted(bars, \.start) {
            p.bar = b
            let beats = bars[b].beats
            let k = lastStarted(beats, { $0 }) ?? 0
            let nextBar = b + 1 < bars.count ? bars[b + 1].start : nil
            let span = beats.count > 1 ? (beats[beats.count - 1] - beats[0]) / Double(beats.count - 1) : nil
            let next = k + 1 < beats.count ? beats[k + 1] : nextBar ?? span.map { beats[k] + $0 }
            let fraction = next.map { $0 > beats[k] ? min(1, (time - beats[k]) / ($0 - beats[k])) : 0 } ?? 0
            p.beat = Double(k + 1) + max(0, fraction)
        }
        if let n = lastStarted(notes, \.start), time < notes[n].end { p.note = n }
        if let l = lastStarted(lines, \.start) {
            p.line = l
            if let w = lastStarted(lines[l].words, \.start) {
                p.word = w
                p.syllable = lastStarted(lines[l].words[w].syllables, \.start)
            }
        }
        return p
    }

    // MARK: - Building

    /// Builds the timeline of a rendered song from a SheetSage2 transcription of its audio
    /// (`events.beatGrid`, `events.melodyOnsets`; melody-only is enough).
    public static func build(
        abc: String, lyrics: String, language: Language,
        beatGrid: [(subbeat: Int, time: Double)], onsets: [(subbeat: Int, midi: Int)]
    ) throws -> LyricTimeline {
        try build(abc: abc, lyrics: lyrics, language: language,
                  grid: beatGrid.map { GridPoint(subbeat: $0.subbeat, time: $0.time) },
                  heard: onsets.map { HeardNote(subbeat: $0.subbeat, midi: $0.midi) })
    }

    public static func build(
        abc: String, lyrics: String, language: Language = .english,
        grid points: [GridPoint]? = nil, heard: [HeardNote] = [], anchors: [Double?]? = nil,
        duration: Double? = nil, clockSource: String? = nil
    ) throws -> LyricTimeline {
        let score = try ABCScore(abc)
        let clock = Clock(score: score, points: points, heard: heard)
        let sounding = ABCScore.sounding(score.vocal)
        let notes = sounding.map { n in
            AlignNote(start: n.start, duration: n.duration, t0: clock.seconds(n.start), t1: clock.seconds(n.start + n.duration))
        }
        let barStarts = score.barStarts
        func beatUnits(_ bar: Int) -> Double { Double(score.unit) / Double(score.meter(ofBar: bar).1) }
        func position(_ units: Int) -> (bar: Int, beat: Double) {
            var bar = 0
            if units >= barStarts.last! {
                // Past the written bars (a render longer than its score): header meter.
                let extra = units - barStarts.last!
                bar = barStarts.count - 1 + extra / score.barUnits
                return (bar, 1 + Double(extra % score.barUnits) / beatUnits(bar))
            }
            var lo = 0, hi = barStarts.count - 2
            while lo <= hi {
                let mid = (lo + hi) / 2
                if barStarts[mid] <= units { bar = mid; lo = mid + 1 } else { hi = mid - 1 }
            }
            return (bar, 1 + Double(units - barStarts[bar]) / beatUnits(bar))
        }
        func position(at time: Double) -> (bar: Int, beat: Double) {
            // Inverse of the clock, for syllables that share a note (their own slice of it).
            guard let i = notes.lastIndex(where: { $0.t0 <= time + 1e-9 }) else { return position(notes.first?.start ?? 0) }
            let n = notes[i]
            let fraction = n.t1 > n.t0 ? (time - n.t0) / (n.t1 - n.t0) : 0
            let p = position(n.start)
            return (p.bar, p.beat + fraction * Double(n.duration) / beatUnits(p.bar))
        }
        let parsed = parseLyrics(lyrics, language: language)
        var syllables: [AlignSyllable] = []
        var anchorIndex = 0
        for (li, line) in parsed.enumerated() {
            for (wi, word) in line.words.enumerated() {
                for (si, piece) in word.syllables.enumerated() {
                    var s = AlignSyllable(
                        line: li, word: wi, text: piece, lineStart: wi == 0 && si == 0,
                        wordEnd: si == word.syllables.count - 1,
                        lineEnd: wi == line.words.count - 1 && si == word.syllables.count - 1)
                    if si == 0, let anchors, anchorIndex < anchors.count { s.anchor = anchors[anchorIndex] }
                    syllables.append(s)
                }
                anchorIndex += 1
            }
        }
        // An instrumental (no sung note or no lyric word) still gets its bars.
        let sung = !notes.isEmpty && !syllables.isEmpty
        let (groups, cost) = sung ? align(notes: notes, syllables: syllables, unitsPerQuarter: score.unitsPerQuarter) : ([], 0)

        // Syllable times; a note shared by several syllables is split evenly between them.
        var spans: [(Double, Double, Int)] = []
        var sharing: [Int: [Int]] = [:]
        for (i, g) in groups.enumerated() where g.count == 1 { sharing[g.lowerBound, default: []].append(i) }
        for (i, g) in groups.enumerated() {
            let first = notes[g.lowerBound], last = notes[g.upperBound - 1]
            var span = (first.t0, last.t1, g.count)
            if g.count == 1, let peers = sharing[g.lowerBound], peers.count > 1, let r = peers.firstIndex(of: i) {
                let step = (first.t1 - first.t0) / Double(peers.count)
                span = (first.t0 + Double(r) * step, first.t0 + Double(r + 1) * step, 1)
            }
            spans.append(span)
        }
        let timedNotes = zip(sounding, notes).map { s, n in
            let p = position(s.start)
            return Note(start: n.t0, end: n.t1, midi: s.midi, bar: p.bar, beat: p.beat, beats: Double(s.duration) / beatUnits(p.bar))
        }
        let sectionAt = Dictionary(score.sections.map { ($0.0, $0.1) }, uniquingKeysWith: { a, _ in a })
        let bars = (0..<(barStarts.count - 1)).map { b in
            Bar(index: b, start: clock.seconds(barStarts[b]),
                beats: (0..<score.meter(ofBar: b).0).map { clock.seconds(barStarts[b] + Int((Double($0) * beatUnits(b)).rounded())) },
                section: sectionAt[b])
        }
        // Musical position of a time read off the bar table (words placed off the score).
        func position(fromBars time: Double) -> (bar: Int, beat: Double) {
            guard let b = bars.lastIndex(where: { $0.start <= time }) else { return (0, 1) }
            let beats = bars[b].beats
            let next = b + 1 < bars.count ? bars[b + 1].start : nil
            let k = beats.lastIndex(where: { $0 <= time }) ?? 0
            let k1 = k + 1 < beats.count ? beats[k + 1] : next
            guard let k1, k1 > beats[k] else { return (b, Double(k + 1)) }
            return (b, Double(k + 1) + min(1, (time - beats[k]) / (k1 - beats[k])))
        }

        // Word index (parse order) → syllable indices, and the word's anchor.
        var wordSyllables: [[Int]] = []
        for k in syllables.indices {
            if k == 0 || syllables[k].line != syllables[k - 1].line || syllables[k].word != syllables[k - 1].word {
                wordSyllables.append([])
            }
            wordSyllables[wordSyllables.count - 1].append(k)
        }
        var starts = wordSyllables.map { spans.isEmpty ? 0 : spans[$0[0]].0 }
        // A word the score places far from where the voice is heard (the render left the planned
        // melody, or the score ended before the lyrics) is timed by its anchor alone.
        var offScore = Set<Int>()
        for (w, idx) in wordSyllables.enumerated() where sung {
            if let a = syllables[idx[0]].anchor, abs(starts[w] - a) > Cost.anchorFallback {
                offScore.insert(w)
                starts[w] = a
            }
        }
        var lines: [Line] = []
        var w = 0
        for (li, line) in parsed.enumerated() where sung {
            var words: [Word] = []
            for word in line.words {
                let idx = wordSyllables[w]
                var sy: [Syllable]
                if offScore.contains(w) {
                    // Until the next word starts (at most 1.5 s), syllables in equal parts.
                    let next = w + 1 < starts.count ? starts[w + 1] : starts[w] + 1.5
                    let end = min(max(next - 0.02, starts[w] + 0.1), starts[w] + 1.5)
                    let step = (end - starts[w]) / Double(idx.count)
                    sy = idx.enumerated().map { r, k in
                        let t = starts[w] + Double(r) * step
                        let p = position(fromBars: t)
                        return Syllable(text: syllables[k].text, start: t, end: t + step, notes: 0, firstNote: -1, bar: p.bar, beat: p.beat)
                    }
                } else {
                    sy = idx.map { k in
                        let p = position(at: spans[k].0)
                        return Syllable(text: syllables[k].text, start: spans[k].0, end: spans[k].1, notes: spans[k].2,
                                        firstNote: groups[k].lowerBound, bar: p.bar, beat: p.beat)
                    }
                }
                words.append(Word(text: word.text, start: sy.first!.start, end: sy.last!.end,
                                  bar: sy.first!.bar, beat: sy.first!.beat, onScore: !offScore.contains(w), syllables: sy))
                w += 1
            }
            lines.append(Line(text: line.text, section: line.section, start: words.first!.start, end: words.last!.end,
                              bar: words.first!.bar, beat: words.first!.beat, words: words))
        }
        // What the render never reaches (a song cut to length) is dropped.
        let end = duration ?? .infinity
        return LyricTimeline(
            version: formatVersion, tempo: score.tempo, meter: [score.meter.0, score.meter.1], duration: duration,
            bars: bars.filter { $0.start < end }, notes: timedNotes.filter { $0.start < end },
            lines: lines.filter { $0.start < end },
            grid: clockSource ?? (points == nil ? "nominal" : "beats"), gridOffset: clock.offset,
            gridMatch: clock.match, alignmentCost: cost)
    }

    // MARK: - Clock

    struct Clock {
        let score: ABCScore
        let subbeats: [Int]
        let times: [Double]
        var offset = 0
        var match: Double?

        init(score: ABCScore, points: [GridPoint]?, heard: [HeardNote]) {
            self.score = score
            var unique: [Int: Double] = [:]
            for p in points ?? [] where unique[p.subbeat] == nil { unique[p.subbeat] = p.time }
            let sorted = unique.sorted { $0.key < $1.key }
            subbeats = sorted.map(\.key)
            times = sorted.map(\.value)
            guard subbeats.count >= 2, !heard.isEmpty else { return }
            // Bar 1 is where the planned onsets (pitch classes, every voice) best match the heard
            // ones; ties go to the smallest shift.
            var planned = Set<[Int]>()
            for voice in [score.vocal, score.ins] {
                for n in ABCScore.sounding(voice) { planned.insert([n.start * 16 / score.unit, n.midi % 12]) }
            }
            var best = (offset: 0, hits: -1)
            for k in (0...64).flatMap({ $0 == 0 ? [0] : [-$0, $0] }) {
                let hits = heard.reduce(0) { $0 + (planned.contains([$1.subbeat - k, $1.midi % 12]) ? 1 : 0) }
                if hits > best.hits { best = (k, hits) }
            }
            offset = best.offset
            match = Double(best.hits) / Double(heard.count)
        }

        func seconds(_ units: Int) -> Double {
            guard subbeats.count >= 2 else {
                return Double(units) / score.unitsPerQuarter * 60 / score.tempo
            }
            let s = Double(units) * 16 / Double(score.unit) + Double(offset)
            var i = 0
            var lo = 0, hi = subbeats.count - 1
            while lo <= hi {
                let mid = (lo + hi) / 2
                if Double(subbeats[mid]) <= s { i = mid; lo = mid + 1 } else { hi = mid - 1 }
            }
            i = min(max(i, 0), subbeats.count - 2)
            let s0 = Double(subbeats[i]), s1 = Double(subbeats[i + 1])
            let t = times[i] + (times[i + 1] - times[i]) * (s - s0) / (s1 - s0)
            return (t * 1000).rounded() / 1000
        }
    }

    // MARK: - Alignment

    struct AlignNote {
        var start: Int
        var duration: Int
        var t0: Double
        var t1: Double
    }

    struct AlignSyllable {
        var line: Int
        var word: Int
        var text: String
        var lineStart: Bool
        var wordEnd: Bool
        var lineEnd: Bool
        var anchor: Double?
    }

    enum Cost {
        static let melismaMid = 1.2
        static let melismaWordEnd = 0.8
        static let melismaLineEnd = 0.25
        static let share = 1.6
        static let lineStartMidPhrase = 2.5
        static let lineContinuesAcrossRest = 2.5
        static let syllableAcrossRest = 6.0
        static let anchorPer100ms = 2.0
        static let anchorTolerance = 0.08
        /// Beyond this distance from its anchor, a word leaves the score and follows the anchor.
        static let anchorFallback = 0.75
        /// A rest of at least this many quarters separates phrases.
        static let phraseGapQuarters = 1.0
        static let maxNotesPerSyllable = 8
    }

    /// Each syllable gets a contiguous group of notes, or shares the previous syllable's note.
    static func align(notes: [AlignNote], syllables: [AlignSyllable], unitsPerQuarter: Double) -> ([Range<Int>], Double) {
        let m = notes.count, s = syllables.count
        let gap = Cost.phraseGapQuarters * unitsPerQuarter
        let phraseStart = (0..<m).map { i in
            i == 0 || Double(notes[i].start - (notes[i - 1].start + notes[i - 1].duration)) >= gap
        }
        func anchorCost(_ syllable: AlignSyllable, _ t: Double) -> Double {
            guard let a = syllable.anchor else { return 0 }
            return Cost.anchorPer100ms * max(0, abs(t - a) - Cost.anchorTolerance) / 0.1
        }
        enum Step { case share, group(Int) }
        var dp = [[Double]](repeating: [Double](repeating: .infinity, count: m + 1), count: s + 1)
        var back = [[Step?]](repeating: [Step?](repeating: nil, count: m + 1), count: s + 1)
        dp[0][0] = 0
        for i in 1...s {
            let syl = syllables[i - 1]
            let perExtra = syl.lineEnd ? Cost.melismaLineEnd : syl.wordEnd ? Cost.melismaWordEnd : Cost.melismaMid
            for j in 1...m {
                if dp[i - 1][j].isFinite, !syl.lineStart {
                    let c = dp[i - 1][j] + Cost.share + anchorCost(syl, (notes[j - 1].t0 + notes[j - 1].t1) / 2)
                    if c < dp[i][j] { dp[i][j] = c; back[i][j] = .share }
                }
                for k in max(0, j - Cost.maxNotesPerSyllable)..<j where dp[i - 1][k].isFinite {
                    var c = dp[i - 1][k] + anchorCost(syl, notes[k].t0)
                    for x in (k + 1)..<j {
                        if phraseStart[x] { c += Cost.syllableAcrossRest }
                        c += perExtra
                    }
                    if syl.lineStart, !phraseStart[k] { c += Cost.lineStartMidPhrase }
                    if !syl.lineStart, phraseStart[k] { c += Cost.lineContinuesAcrossRest }
                    if c < dp[i][j] { dp[i][j] = c; back[i][j] = .group(k) }
                }
            }
        }
        // Unsung trailing notes (an instrumental doubling) cost like a melisma.
        var bestJ = 0, bestCost = Double.infinity
        for j in 0...m where dp[s][j].isFinite {
            let c = dp[s][j] + Double(m - j) * Cost.melismaMid
            if c < bestCost { bestCost = c; bestJ = j }
        }
        var groups = [Range<Int>](repeating: 0..<1, count: s)
        var j = bestJ
        for i in stride(from: s, through: 1, by: -1) {
            switch back[i][j] {
            case .share: groups[i - 1] = (j - 1)..<j
            case .group(let k): groups[i - 1] = k..<j; j = k
            case nil: groups[i - 1] = max(0, j - 1)..<max(1, j)
            }
        }
        return (groups, bestCost)
    }

    // MARK: - Lyrics

    struct LyricWord { var text: String; var syllables: [String] }
    struct LyricLine { var text: String; var section: String?; var words: [LyricWord] }

    static func parseLyrics(_ lyrics: String, language: Language) -> [LyricLine] {
        var out: [LyricLine] = []
        var section: String?
        for raw in lyrics.components(separatedBy: .newlines) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty { continue }
            if line.hasPrefix("["), line.hasSuffix("]") {
                section = String(line.dropFirst().dropLast())
                continue
            }
            let tokens = line.split(whereSeparator: \.isWhitespace).map(String.init)
            var words: [LyricWord] = []
            for (k, token) in tokens.enumerated() {
                let next = k + 1 < tokens.count ? tokens[k + 1] : nil
                let n = language == .french ? syllablesFrench(token, next: next) : syllablesEnglish(token)
                guard n > 0 else { continue }
                words.append(LyricWord(text: token, syllables: split(token, into: n)))
            }
            if !words.isEmpty { out.append(LyricLine(text: line, section: section, words: words)) }
        }
        return out
    }

    private static let vowels = Set("aeiouyàâäéèêëîïôöùûüÿœæ")

    private static func vowelGroups(_ w: [Character]) -> Int {
        var n = 0, inGroup = false
        for c in w {
            let v = vowels.contains(c)
            if v, !inGroup { n += 1 }
            inGroup = v
        }
        return n
    }

    /// Spoken-English syllables by rule: vowel groups, minus silent endings, plus common hiatus.
    static func syllablesEnglish(_ token: String) -> Int {
        let w = Array(token.lowercased().filter { $0.isLetter })
        guard !w.isEmpty else { return 0 }
        var n = vowelGroups(w)
        let s = String(w)
        if s.count > 2, s.hasSuffix("e"), !s.hasSuffix("le") || vowels.contains(w[w.count - 3]), !s.hasSuffix("ee"), n > 1 { n -= 1 }
        if s.count > 3, s.hasSuffix("es") || s.hasSuffix("ed"), n > 1 {
            let before = w[w.count - 3]
            let sounded = s.hasSuffix("ed") ? "td".contains(before) : "sxzc".contains(before) || s.hasSuffix("ges")
            if !sounded, !vowels.contains(before) { n -= 1 }
        }
        for hiatus in ["eo", "ia", "iu", "uo", "ua"] where s.contains(hiatus) { n += 1 }
        if let r = s.range(of: "io"), r.lowerBound > s.startIndex, !"tsc".contains(s[s.index(before: r.lowerBound)]) { n += 1 }
        return max(1, n)
    }

    /// Sung French syllables: vowel groups; a final mute e is sung before a consonant and elided
    /// before a vowel, an h, or at the end of a line.
    static func syllablesFrench(_ token: String, next: String?) -> Int {
        var s = token.lowercased().replacingOccurrences(of: "’", with: "'")
        if let apostrophe = s.lastIndex(of: "'") { s = String(s[s.index(after: apostrophe)...]) }
        let w = Array(s.filter { $0.isLetter })
        guard !w.isEmpty else { return 0 }
        var n = vowelGroups(w)
        let word = String(w)
        for ending in ["ent", "es", "e"] where word.hasSuffix(ending) && word.count > ending.count {
            let before = w[w.count - ending.count - 1]
            if !vowels.contains(before), n > 1 {
                let following = next?.lowercased().first(where: { $0.isLetter })
                let sung = following.map { !vowels.contains($0) && $0 != "h" } ?? false
                if !sung { n -= 1 }
            }
            break
        }
        return max(1, n)
    }

    /// Display split of a word into `n` chunks at vowel-group starts (cosmetic).
    static func split(_ word: String, into n: Int) -> [String] {
        guard n > 1 else { return [word] }
        let chars = Array(word)
        var starts: [Int] = []
        var inGroup = false
        for (i, c) in chars.enumerated() {
            let v = vowels.contains(Character(c.lowercased()))
            if v, !inGroup { starts.append(i) }
            inGroup = v
        }
        var cuts: [Int] = []
        for g in starts.dropFirst() {
            // Put one consonant before the vowel group in the next chunk ("Mor-ning").
            let c = g > 0 && !vowels.contains(Character(chars[g - 1].lowercased())) && chars[g - 1].isLetter ? g - 1 : g
            if c > (cuts.last ?? 0) { cuts.append(c) }
        }
        if cuts.count < n - 1 {
            let step = max(1, chars.count / n)
            cuts = (1..<n).map { min(chars.count, $0 * step) }
        }
        cuts = Array(cuts.prefix(n - 1))
        var parts: [String] = []
        var previous = 0
        for c in cuts + [chars.count] {
            parts.append(String(chars[previous..<max(previous, c)]))
            previous = max(previous, c)
        }
        return parts
    }
}
