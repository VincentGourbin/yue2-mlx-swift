// AttentionTimeline.swift - lyric and bar timing read from the LM's own alignment heads, no audio analysis
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// While it writes each 40 ms semantic frame, YuE2 attends to the prompt, and a few AR heads
/// follow the song: four look at the score bar being played, one at the lyric word about to be
/// sung. Reading them back after the semantic phase (a teacher-forced pass over 19 of the 28
/// layers, reusing the generation's KV cache) times the score bars and the words without
/// transcribing the audio and without another model.
///
/// Measured 2026-10-08 on eight songs (four generated on an iPhone in int4, four on the Mac; the
/// heads and their lead are identical in int4 and bf16): bar starts within 60–160 ms (median) of
/// a SheetSage2 transcription of the render; word starts 71–97 % within 300 ms of an MMS forced
/// alignment of the separated voice, including songs whose voice leaves the planned melody
/// (where score + SheetSage2 fell to 10–12 %). Heads were found on two songs and held on the six
/// others.
public enum AttentionTimeline {
    /// Heads that attend to the score bar being played (layer, head).
    public static let barHeads: [(layer: Int, head: Int)] = [(6, 8), (10, 3), (18, 7), (18, 6)]
    /// Head that attends to the next lyric word: its attention reaches word k + 1 when word k starts.
    public static let wordHead: (layer: Int, head: Int) = (14, 10)
    /// The bar heads run this far ahead of the music (0.28–0.35 s on five songs).
    public static let barLead = 0.33
    static let frameSeconds = 0.04

    /// Times read from the heads: start of each score bar and each lyric word (nil: not reached
    /// by the render, e.g. lines after the end of a song cut to length).
    public struct Reading: Equatable, Sendable {
        public var barStarts: [Double?]
        public var wordStarts: [Double?]
    }

    /// Prompt token → segment: score bar `b` is segment `b`, lyric word `w` is `bars + w`.
    struct Segments {
        var ids: [Int32]
        var bars: Int
        var words: Int
    }

    // MARK: - Public entry points

    /// The timeline of a song straight from its semantic phase (call it before the AR weights
    /// are released): score, lyrics and language from the plan, the generation's KV cache reused.
    public static func timeline(
        model: YuE2ForCausalLM, tokenizer: YuE2Tokenizer, semantic: SemanticResult,
        language: LyricTimeline.Language? = nil, cancel: (() -> Bool)? = nil
    ) throws -> LyricTimeline {
        try timeline(model: model, tokenizer: tokenizer, plan: semantic.plan, semantic: semantic.tokens,
                     language: language, cache: semantic.cache, cancel: cancel)
    }

    /// The timeline of any song from its plan and semantic tokens (no cache: the pass builds its
    /// own for 19 layers).
    public static func timeline(
        model: YuE2ForCausalLM, tokenizer: YuE2Tokenizer, plan: SymbolicPlan, semantic: [Int],
        language: LyricTimeline.Language? = nil, cache: KVCache? = nil, duration: Double? = nil,
        cancel: (() -> Bool)? = nil
    ) throws -> LyricTimeline {
        guard let abc = plan.abc else { throw YuE2Error.invalidRequest("timeline: the song has no score (cot off)") }
        return try timeline(
            model: model, tokenizer: tokenizer, prefix: plan.prefix, semantic: semantic, abc: abc,
            lyrics: plan.request.lyrics, language: language ?? .guess(style: plan.request.style),
            cache: cache, duration: duration, cancel: cancel)
    }

    /// Re-analyses a saved song (a migration of songs generated before timelines existed):
    /// `directory` holds `semantic.npy` and the plan (`plan.json` + `prefix.npy`), either beside
    /// it (`yue2 generate --out`) or in a `plan/` subdirectory (the app). The AR weights must be
    /// loaded; nothing is written. About one prefill of the song over 19 layers.
    public static func reanalyze(
        song directory: URL, model: YuE2ForCausalLM, tokenizer: YuE2Tokenizer,
        language: LyricTimeline.Language? = nil, cancel: (() -> Bool)? = nil
    ) throws -> LyricTimeline {
        let saved = try SavedSong(directory: directory)
        return try timeline(model: model, tokenizer: tokenizer, plan: saved.plan, semantic: saved.semantic,
                            language: language, cancel: cancel)
    }

    /// What a re-analysis reads from a saved song directory.
    public struct SavedSong {
        public var plan: SymbolicPlan
        public var semantic: [Int]

        /// Reads `plan.json` (from `directory/plan/` if present, else `directory/`) and
        /// `directory/semantic.npy`. The plan's hashes are not checked: timing only reads it.
        public init(directory: URL) throws {
            let fm = FileManager.default
            let nested = directory.appendingPathComponent("plan")
            let planDir = fm.fileExists(atPath: nested.appendingPathComponent("plan.json").path) ? nested : directory
            let planFile = planDir.appendingPathComponent("plan.json")
            let semanticFile = directory.appendingPathComponent("semantic.npy")
            guard fm.fileExists(atPath: planFile.path) else {
                throw YuE2Error.invalidRequest("timeline: no plan.json in \(directory.lastPathComponent) or its plan/")
            }
            guard fm.fileExists(atPath: semanticFile.path) else {
                throw YuE2Error.invalidRequest("timeline: no semantic.npy in \(directory.lastPathComponent)")
            }
            plan = try JSONDecoder().decode(SymbolicPlan.self, from: Data(contentsOf: planFile))
            semantic = try NumpyIO.readInt32(semanticFile).map(Int.init)
        }
    }

    /// The timeline of a song right after its semantic phase, while the AR weights are loaded:
    /// `prefix` is the semantic-phase prompt (`SymbolicPlan.prefix`), `semantic` the codec
    /// tokens, `cache` the semantic phase's KV cache if still at hand (`SemanticResult.cache`).
    /// `duration` (seconds of audio, default the codec length) drops what the render never reaches.
    public static func timeline(
        model: YuE2ForCausalLM, tokenizer: YuE2Tokenizer, prefix: [Int], semantic: [Int],
        abc: String, lyrics: String, language: LyricTimeline.Language,
        cache: KVCache? = nil, duration: Double? = nil, cancel: (() -> Bool)? = nil
    ) throws -> LyricTimeline {
        let score = try ABCScore(abc)
        let reading = try read(
            model: model, tokenizer: tokenizer, prefix: prefix, semantic: semantic,
            score: score, lyrics: lyrics, language: language, cache: cache, cancel: cancel)
        // Bar clock: each reached bar's start, at its 16th-note position in the score.
        let starts = score.barStarts
        var grid: [LyricTimeline.GridPoint] = []
        for (b, t) in reading.barStarts.enumerated() {
            guard let t, b < starts.count else { continue }
            let subbeat = starts[b] * 16 / score.unit
            if let last = grid.last, t <= last.time { continue }
            grid.append(.init(subbeat: subbeat, time: t))
        }
        let audio = duration ?? Double(semantic.count) * frameSeconds
        func build(_ grid: [LyricTimeline.GridPoint]) throws -> LyricTimeline {
            try LyricTimeline.build(
                abc: abc, lyrics: lyrics, language: language, grid: grid.count >= 2 ? grid : nil,
                anchors: reading.wordStarts, duration: audio, clockSource: "attention")
        }
        let first = try build(grid)
        return try build(refined(grid, by: first, anchors: reading.wordStarts, unit: score.unit, barStarts: starts))
    }

    /// The word head times words more finely (≈ 60 ms) than the bar heads time bars (≈ 100 ms):
    /// each bar is moved by the median gap between the words sung around it (±2 bars) and their
    /// anchors, so bars, notes and words stay on one clock. Bars with no word nearby keep the
    /// correction of the closest bar that has one, up to 4 bars away.
    static func refined(
        _ grid: [LyricTimeline.GridPoint], by timeline: LyricTimeline, anchors: [Double?], unit: Int, barStarts: [Int]
    ) -> [LyricTimeline.GridPoint] {
        let words = timeline.lines.flatMap(\.words)
        let gaps: [(bar: Int, gap: Double)] = zip(words, anchors).compactMap { w, a in
            guard w.onScore, let a else { return nil }
            return (w.bar, a - w.start)
        }
        guard !gaps.isEmpty else { return grid }
        var correction: [Int: Double] = [:]
        for point in grid {
            guard let bar = barStarts.firstIndex(of: point.subbeat * unit / 16) else { continue }
            let near = gaps.filter { abs($0.bar - bar) <= 2 }.map(\.gap).sorted()
            if near.count >= 2 { correction[bar] = near[near.count / 2] }
        }
        var out: [LyricTimeline.GridPoint] = []
        for point in grid {
            guard let bar = barStarts.firstIndex(of: point.subbeat * unit / 16) else { continue }
            let c = correction[bar] ?? correction.filter { abs($0.key - bar) <= 4 }
                .min(by: { abs($0.key - bar) < abs($1.key - bar) })?.value ?? 0
            let moved = LyricTimeline.GridPoint(subbeat: point.subbeat, time: max(0, point.time + c))
            if let last = out.last, moved.time <= last.time { continue }
            out.append(moved)
        }
        return out
    }

    /// Reads bar and word starts from the alignment heads.
    public static func read(
        model: YuE2ForCausalLM, tokenizer: YuE2Tokenizer, prefix: [Int], semantic: [Int],
        score: ABCScore, lyrics: String, language: LyricTimeline.Language, cache: KVCache? = nil,
        cancel: (() -> Bool)? = nil
    ) throws -> Reading {
        let segments = try segments(prefix: prefix, tokenizer: tokenizer, lyrics: lyrics, language: language)
        guard !semantic.isEmpty, segments.bars + segments.words > 0 else {
            return Reading(barStarts: [], wordStarts: [])
        }
        let tokens = prefix.map(Int32.init) + semantic.map { Int32($0 + YuE2Token.codecOffset) }
        let heads = barHeads + [wordHead]
        let mass = try model.prefixAttention(
            tokens: tokens, prefixLength: prefix.count, segments: segments.ids,
            segmentCount: segments.bars + segments.words, heads: heads, reusing: cache, cancel: cancel)
        let frames = mass.dim(1)
        let all = mass.asArray(Float.self)
        let width = segments.bars + segments.words
        func head(_ h: Int, _ range: Range<Int>) -> [[Float]] {
            (0..<frames).map { f in
                let row = Array(all[(h * frames + f) * width + range.lowerBound ..< (h * frames + f) * width + range.upperBound])
                let total = max(row.reduce(0, +), 1e-9)
                return row.map { $0 / total }
            }
        }
        var barStarts: [Double?] = []
        if segments.bars > 1 {
            let perHead = barHeads.indices.map { head($0, 0..<segments.bars) }
            let averaged = (0..<frames).map { f in
                (0..<segments.bars).map { s in perHead.reduce(Float(0)) { $0 + $1[f][s] } / Float(perHead.count) }
            }
            barStarts = segmentStarts(monotonicPath(averaged), count: segments.bars).map { $0.map { max(0, $0 + barLead) } }
            barStarts[0] = barStarts[0].map { _ in 0 }
        }
        var wordStarts: [Double?] = []
        if segments.words > 1 {
            let word = head(barHeads.count, segments.bars..<width)
            // Attention on word k + 1 begins when word k is sung; the last word has no successor.
            wordStarts = Array(segmentStarts(monotonicPath(word), count: segments.words).dropFirst()) + [nil]
        }
        return Reading(barStarts: barStarts, wordStarts: wordStarts)
    }

    // MARK: - Prompt segments

    /// Maps every prompt token to the score bar it writes (ABC region, both voices) or the lyric
    /// word it spells (words as `LyricTimeline` counts them), by exact byte offsets.
    static func segments(prefix: [Int], tokenizer: YuE2Tokenizer, lyrics: String, language: LyricTimeline.Language) throws -> Segments {
        guard let abcStart = prefix.firstIndex(of: YuE2Token.abcStart),
              let abcEnd = prefix.lastIndex(of: YuE2Token.abcEnd), abcEnd > abcStart else {
            throw YuE2Error.invalidRequest("timeline: the prompt carries no score")
        }
        var ids = [Int32](repeating: -1, count: prefix.count)

        // Score: bar index of every byte of the ABC text (bars counted per voice; voices align).
        let abcIDs = Array(prefix[(abcStart + 1)..<abcEnd])
        let abcBytes = Array(tokenizer.decode(abcIDs).utf8)
        var barOfByte = [Int](repeating: -1, count: abcBytes.count)
        var count: [String: Int] = [:]
        var voice: String?
        var lineStart = 0
        for i in 0...abcBytes.count where i == abcBytes.count || abcBytes[i] == UInt8(ascii: "\n") {
            let line = Array(abcBytes[lineStart..<i])
            defer { lineStart = i + 1 }
            let text = String(decoding: line, as: UTF8.self).trimmingCharacters(in: .whitespaces)
            if text.hasPrefix("V:") {
                let name = text.dropFirst(2).split(separator: " ").first.map(String.init)
                voice = text.contains("clef") ? nil : name
                continue
            }
            guard let v = voice, !text.isEmpty, !text.hasPrefix("%"),
                  !(text.count > 1 && text.dropFirst().first == ":" && text.first!.isUppercase) else { continue }
            var chunk: [UInt8] = []
            for (j, c) in line.enumerated() {
                barOfByte[lineStart + j] = count[v, default: 0]
                if c == UInt8(ascii: "|") {
                    let s = String(decoding: chunk, as: UTF8.self).trimmingCharacters(in: .whitespaces)
                    var bars = 1
                    if s.hasPrefix("Z"), let n = Int(s.dropFirst()) { bars = n }
                    count[v, default: 0] += bars
                    chunk = []
                } else {
                    chunk.append(c)
                }
            }
        }
        let bars = count.values.max() ?? 0
        var offset = 0
        for (k, id) in abcIDs.enumerated() {
            let n = tokenizer.byteCount(of: id)
            if let b = (offset..<min(offset + n, abcBytes.count)).lazy.map({ barOfByte[$0] }).first(where: { $0 >= 0 }) {
                ids[abcStart + 1 + k] = Int32(b)
            }
            offset += n
        }

        // Lyrics: the words of the request text, located after "[Lyrics]\n".
        let textIDs = Array(prefix[1..<abcStart])
        let textBytes = Array(tokenizer.decode(textIDs).utf8)
        let marker = Array("[Lyrics]\n".utf8)
        guard let markerAt = (0...(max(0, textBytes.count - marker.count))).first(where: {
            textBytes.count >= marker.count && Array(textBytes[$0..<($0 + marker.count)]) == marker
        }) else {
            return Segments(ids: ids, bars: bars, words: 0)
        }
        var wordOfByte = [Int](repeating: -1, count: textBytes.count)
        var words = 0
        lineStart = markerAt + marker.count
        for i in lineStart...textBytes.count where i == textBytes.count || textBytes[i] == UInt8(ascii: "\n") {
            defer { lineStart = i + 1 }
            let line = Array(textBytes[lineStart..<i])
            let text = String(decoding: line, as: UTF8.self).trimmingCharacters(in: .whitespaces)
            if text.isEmpty || (text.hasPrefix("[") && text.hasSuffix("]")) { continue }
            // Whitespace-separated tokens with their byte ranges; kept as LyricTimeline keeps them.
            var tokens: [(String, Range<Int>)] = []
            var start: Int?
            for j in 0...line.count {
                let space = j == line.count || [0x20, 0x09, 0x0D].contains(line[j])
                if space, let s = start {
                    tokens.append((String(decoding: line[s..<j], as: UTF8.self), (lineStart + s)..<(lineStart + j)))
                    start = nil
                } else if !space, start == nil {
                    start = j
                }
            }
            for (k, (token, range)) in tokens.enumerated() {
                let next = k + 1 < tokens.count ? tokens[k + 1].0 : nil
                let syllables = language == .french
                    ? LyricTimeline.syllablesFrench(token, next: next) : LyricTimeline.syllablesEnglish(token)
                guard syllables > 0 else { continue }
                for b in range { wordOfByte[b] = words }
                words += 1
            }
        }
        offset = 0
        for (k, id) in textIDs.enumerated() {
            let n = tokenizer.byteCount(of: id)
            if let w = (offset..<min(offset + n, textBytes.count)).lazy.map({ wordOfByte[$0] }).first(where: { $0 >= 0 }) {
                ids[1 + k] = Int32(bars + w)
            }
            offset += n
        }
        return Segments(ids: ids, bars: bars, words: words)
    }

    // MARK: - Monotonic path

    /// Most likely path through segments `0, 1, 2…` (one per frame, never going back, starting
    /// on segment 0 and ending wherever the song ends), as each frame's segment.
    static func monotonicPath(_ probs: [[Float]]) -> [Int] {
        guard let n = probs.first?.count, n > 0 else { return [] }
        let frames = probs.count
        let cost = probs.map { $0.map { -log(Double($0) + 1e-6) } }
        var d = [Double](repeating: .infinity, count: n)
        var moved = [[Bool]](repeating: [Bool](repeating: false, count: n), count: frames)
        d[0] = cost[0][0]
        for f in 1..<max(frames, 1) {
            var next = [Double](repeating: .infinity, count: n)
            for s in 0..<n {
                let stay = d[s], move = s > 0 ? d[s - 1] : .infinity
                moved[f][s] = move < stay
                next[s] = min(stay, move) + cost[f][s]
            }
            d = next
        }
        var s = d.indices.min(by: { d[$0] < d[$1] }) ?? 0
        var path = [Int](repeating: 0, count: frames)
        for f in stride(from: frames - 1, through: 0, by: -1) {
            path[f] = s
            if f > 0, moved[f][s] { s -= 1 }
        }
        return path
    }

    /// Time each segment is first reached on the path (nil if never).
    static func segmentStarts(_ path: [Int], count: Int) -> [Double?] {
        var starts = [Double?](repeating: nil, count: count)
        for (f, s) in path.enumerated() where s < count && starts[s] == nil {
            starts[s] = Double(f) * frameSeconds
        }
        return starts
    }
}
