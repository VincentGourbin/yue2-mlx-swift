// InstrumentalKaraokeTests.swift - ABC dialect reader, Vocal→Ins transfer, lyric timeline (weight-free)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private let score = """
X:1
T:
M:4/4
L:1/32
Q:1/4=120
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:Bb
% intro
V: Vocal
"Bb"z32|Z|
V: Ins
B,8F8f8B8|E8F8^f8B8|
% verse
V: Vocal
"Bb"d8f8f8d8|"Eb"c16-c8z8|
V: Ins
Z2|

"""

@Suite("Instrumental and karaoke")
struct InstrumentalKaraokeTests {
    @Test func readsTheNativeDialect() throws {
        let s = try ABCScore(score)
        #expect(s.vocal.count == 4 && s.ins.count == 4)
        #expect(s.sections.map(\.0) == [0, 2] && s.sections.map(\.1) == ["intro", "verse"])
        #expect(s.header.count == 8 && s.tempo == 120 && s.unit == 32 && s.barUnits == 32)
        // Key of Bb: B and E flat; ^f is F#5; B, is Bb3.
        #expect(ABCScore.sounding(s.ins).prefix(8).map(\.midi) == [58, 65, 77, 70, 63, 65, 78, 70])
        let vocal = ABCScore.sounding(s.vocal)
        #expect(vocal.map(\.start) == [64, 72, 80, 88, 96])
        #expect(vocal.last?.duration == 24) // c16-c8: one held note
    }

    @Test func movesEveryVocalNoteToIns() throws {
        let moved = try Instrumental.transfer(abc: score)
        #expect(moved.vocalNotes == 5 && moved.movedBars == 2 && moved.keptInsBars == 2 && moved.overlapBars.isEmpty)
        let s = try ABCScore(moved.abc)
        #expect(ABCScore.sounding(s.vocal).isEmpty)
        #expect(ABCScore.sounding(s.ins).count == 13)
        #expect(s.vocal.flatMap(\.chords).map(\.1) == ["Bb", "Bb", "Eb"])
        #expect(moved.abc.contains("% verse\nV: Vocal\n\"Bb\"z32|\"Eb\"z32|\nV: Ins\nd8f8f8d8|c16-c8z8|"))
        #expect(moved.abc.hasPrefix("X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=120\nV: Vocal"))
    }

    @Test func vocalWinsAnOverlap() throws {
        let overlapping = score.replacingOccurrences(of: "V: Ins\nZ2|", with: "V: Ins\nB32|Z|")
        let moved = try Instrumental.transfer(abc: overlapping)
        #expect(moved.overlapBars == [2])
        #expect(ABCScore.sounding(try ABCScore(moved.abc).ins).count == 13)
    }

    @Test func renderRequestCarriesTheRecipe() throws {
        #expect(Instrumental.sectionTagLyrics(abc: score) == "[Intro]\n\n[Verse]")
        let planned = try SongRequest(style: "warm piano pop", lyrics: "[Verse]\nla", seed: 7)
        let (request, _) = try Instrumental.renderRequest(from: planned, plannedABC: score)
        #expect(request.style == "warm piano pop, no vocals, no singing, no choir, no spoken words")
        #expect(request.lyrics == "[Intro]\n\n[Verse]" && request.seed == 7)
        #expect(request.negativeStyle == Instrumental.negativeStyle && request.guidance == 1.5)
        #expect(request.abc.map { ABCScore.sounding(try! ABCScore($0).vocal).isEmpty } == true)
    }

    @Test func oneSyllablePerNoteOnTheNominalTempo() throws {
        let t = try LyricTimeline.build(abc: score, lyrics: "[Verse]\none two three four five")
        let words = t.lines[0].words
        #expect(t.grid == "nominal" && t.lines[0].section == "Verse")
        #expect(words.map(\.start) == [4.0, 4.5, 5.0, 5.5, 6.0])
        #expect(words.last?.end == 7.5) // the held note's end, not its first written chunk
    }

    @Test func melismaGoesToTheEndOfTheLine() throws {
        let words = try LyricTimeline.build(abc: score, lyrics: "one two three four").lines[0].words
        #expect(words.map(\.start) == [4.0, 4.5, 5.0, 5.5])
        #expect(words[3].syllables[0].notes == 2 && words[3].end == 7.5)
    }

    @Test func anchorsMoveTheMelisma() throws {
        let words = try LyricTimeline.build(abc: score, lyrics: "one two three four", anchors: [4.0, 4.5, 5.0, 6.02]).lines[0].words
        #expect(words.map(\.start) == [4.0, 4.5, 5.0, 6.0])
        #expect(words[2].syllables[0].notes == 2)
    }

    @Test func beatGridFindsBarOneAndTheClock() throws {
        // A render whose bar 1 falls 4 sixteenths into the grid, 16ths every 0.125 s from 0.3 s.
        let points = (0..<200).map { LyricTimeline.GridPoint(subbeat: $0, time: 0.3 + Double($0) * 0.125) }
        let s = try ABCScore(score)
        let heard = (ABCScore.sounding(s.vocal) + ABCScore.sounding(s.ins)).map {
            LyricTimeline.HeardNote(subbeat: $0.start / 2 + 4, midi: $0.midi)
        }
        let t = try LyricTimeline.build(abc: score, lyrics: "one two three four five", grid: points, heard: heard)
        #expect(t.grid == "beats" && t.gridOffset == 4 && t.gridMatch == 1)
        #expect(abs(t.lines[0].words[0].start - 4.8) < 1e-9)
    }

    @Test func timelineCarriesBarsNotesAndMusicalPositions() throws {
        let t = try LyricTimeline.build(abc: score, lyrics: "[Verse]\none two three four five")
        #expect(t.version == LyricTimeline.formatVersion && t.meter == [4, 4] && t.tempo == 120)
        #expect(t.bars.map(\.start) == [0, 2, 4, 6] && t.bars[2].beats == [4, 4.5, 5, 5.5])
        #expect(t.bars.map(\.section) == ["intro", nil, "verse", nil])
        #expect(t.notes.count == 5 && t.notes.last.map { [$0.start, $0.end, $0.beats] } == [6, 7.5, 3])
        let words = t.lines[0].words
        #expect(words.map(\.bar) == [2, 2, 2, 2, 3] && words.map(\.beat) == [1, 2, 3, 4, 1])
        #expect(words.map { $0.syllables[0].firstNote } == [0, 1, 2, 3, 4] && words.allSatisfy { $0.onScore })
    }

    @Test func durationDropsWhatTheRenderNeverReaches() throws {
        let t = try LyricTimeline.build(abc: score, lyrics: "[Verse]\none two\nthree four five", duration: 4.9)
        #expect(t.bars.count == 3 && t.notes.count == 2 && t.lines.map(\.text) == ["one two"])
    }

    @Test func aWordFarFromItsAnchorLeavesTheScore() throws {
        // The voice sings "five" at 9 s, where the score has no note left.
        let t = try LyricTimeline.build(abc: score, lyrics: "one two three four five", anchors: [4, 4.5, 5, 5.5, 9])
        let five = t.lines[0].words[4]
        #expect(!five.onScore && five.start == 9 && five.syllables[0].firstNote == -1)
        #expect(t.lines[0].words.prefix(4).allSatisfy { $0.onScore })
    }

    @Test func inlineMeterChangesKeepEachBarsLength() throws {
        let changing = """
            X:1
            M:6/8
            L:1/32
            Q:1/4=60
            V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
            V: Ins clef=treble name="Ins Melody" snm="Inst."
            K:Am
            % verse
            V: Vocal
            A24|
            V: Ins
            Z|
            V: Vocal
            M:5/8
            B20|
            V: Ins
            M:5/8
            Z|
            V: Vocal
            M:6/8
            c24|
            V: Ins
            M:6/8
            Z|
            """
        let s = try ABCScore(changing)
        #expect(s.hasInlineFields && s.meter == (6, 8) && s.vocal.map(\.meter.0) == [6, 5, 6])
        #expect(s.barStarts == [0, 24, 44, 68])
        let t = try LyricTimeline.build(abc: changing, lyrics: "la le li")
        #expect(t.bars.map(\.beats.count) == [6, 5, 6] && t.bars.map(\.start) == [0, 3, 5.5])
        #expect(t.lines[0].words.map(\.bar) == [0, 1, 2])
        #expect(throws: YuE2Error.self) { try Instrumental.transfer(abc: changing) }
    }

    @Test func monotonicPathFollowsTheAttention() {
        // Three segments, attention moving 0 → 1 → 2 with noise; the song ends before segment 3.
        let probs: [[Float]] = [[0.9, 0.1, 0, 0], [0.6, 0.4, 0, 0], [0.2, 0.7, 0.1, 0], [0.3, 0.6, 0.1, 0],
                                [0.1, 0.2, 0.7, 0], [0.4, 0.1, 0.5, 0]]
        let path = AttentionTimeline.monotonicPath(probs)
        #expect(path == [0, 0, 1, 1, 2, 2])
        #expect(AttentionTimeline.segmentStarts(path, count: 4) == [0, 0.08, 0.16, nil])
    }

    @Test func wordAnchorsRefineTheBarClock() throws {
        // Bar heads 0.1 s late; the word head hears every word 0.1 s before the clock says.
        let grid = (0..<5).map { LyricTimeline.GridPoint(subbeat: $0 * 16, time: Double($0) * 2 + 0.5) }
        let anchors: [Double?] = [4.4, 4.9, 5.4, 5.9, nil]
        let first = try LyricTimeline.build(abc: score, lyrics: "one two three four five", grid: grid, anchors: anchors)
        let fixed = AttentionTimeline.refined(grid, by: first, anchors: anchors, unit: 32, barStarts: try ABCScore(score).barStarts)
        #expect(zip(fixed.map(\.time), [0.4, 2.4, 4.4, 6.4, 8.4]).allSatisfy { abs($0 - $1) < 1e-9 } && fixed.count == 5)
    }

    @Test func playbackPositionIsABinarySearch() throws {
        let t = try LyricTimeline.build(abc: score, lyrics: "[Verse]\none two\nthree four five")
        #expect(t.position(at: -1) == LyricTimeline.Position())
        let p = t.position(at: 4.6)  // bar 3 (index 2), just after beat 2, "two" sounding
        #expect(p.bar == 2 && abs(p.beat! - 2.2) < 1e-9 && p.note == 1 && p.line == 0 && p.word == 1 && p.syllable == 0)
        #expect(t.position(at: 7).note == 4 && t.position(at: 7).line == 1 && t.position(at: 7).word == 2)
        #expect(t.position(at: 7.9).note == nil)  // the held note ended at 7.5
        #expect(t.position(at: 1).line == nil && t.position(at: 1).bar == 0)
    }

    @Test func timelineFileRoundTrips() throws {
        let t = try LyricTimeline.build(abc: score, lyrics: "one two three four five", duration: 7)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("timeline-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try t.write(to: url)
        #expect(try LyricTimeline.load(from: url) == t)
    }

    @Test func savedSongsAreFoundInBothLayouts() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("saved-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: root) }
        let plan = SymbolicPlan(request: try SongRequest(style: "pop", lyrics: "la"), abc: score, abcIDs: [1], prefix: [1, 2, 3])
        for (name, planDir) in [("cli", ""), ("app", "plan")] {
            let song = root.appendingPathComponent(name)
            try fm.createDirectory(at: song.appendingPathComponent(planDir), withIntermediateDirectories: true)
            try JSONEncoder().encode(plan).write(to: song.appendingPathComponent(planDir).appendingPathComponent("plan.json"))
            try NumpyIO.writeInt32([5, 6, 7], url: song.appendingPathComponent("semantic.npy"))
            let saved = try AttentionTimeline.SavedSong(directory: song)
            #expect(saved.plan == plan && saved.semantic == [5, 6, 7])
        }
        #expect(throws: YuE2Error.self) { _ = try AttentionTimeline.SavedSong(directory: root) }
    }

    @Test func countsSyllables() {
        #expect(["Neon", "fades", "little", "lane", "radio", "Morning"].map(LyricTimeline.syllablesEnglish) == [2, 1, 2, 1, 3, 2])
        #expect(LyricTimeline.syllablesFrench("endormie", next: nil) == 3)
        #expect(LyricTimeline.syllablesFrench("ville", next: "endormie") == 1)
        #expect(LyricTimeline.syllablesFrench("lève", next: "sur") == 2)
        #expect(LyricTimeline.syllablesFrench("chante", next: "ici") == 1)
        #expect(LyricTimeline.syllablesFrench("fenêtres", next: "s'allument") == 3)
        #expect(LyricTimeline.syllablesFrench("s'allument", next: "une") == 2)
        #expect(LyricTimeline.split("Morning", into: 2) == ["Mo", "rning"] || LyricTimeline.split("Morning", into: 2).joined() == "Morning")
    }
}

@Suite("Alignment-head pass (tiny LM)")
struct AttentionProbeTests {
    @Test func reusingTheGenerationCacheGivesTheSameMass() throws {
        let model = try loadedTinyModel()
        let tokens: [Int32] = [5, 17, 3, 29, 12, 7, 8, 24, 23, 11, 30, 4, 31, 19]  // tiny vocabulary: 32
        let prefix = 6
        let segments: [Int32] = [-1, 0, 0, 1, 2, -1]
        let heads: [(layer: Int, head: Int)] = [(0, 0), (1, 1)]
        let own = try model.prefixAttention(tokens: tokens, prefixLength: prefix, segments: segments, segmentCount: 3, heads: heads, chunk: 4)
        #expect(own.shape == [2, tokens.count - prefix, 3])
        // Mass on the prompt segments is a share of one softmax row.
        let sums = own.sum(axis: -1).asArray(Float.self)
        #expect(sums.allSatisfy { $0 > 0 && $0 <= 1.0001 })
        let generated = TokenGenerator.prefill(model: model, tokens: tokens.dropLast().map(Int.init)).cache
        let reused = try model.prefixAttention(tokens: tokens, prefixLength: prefix, segments: segments, segmentCount: 3, heads: heads, reusing: generated)
        #expect(allClose(own, reused, rtol: 1e-3, atol: 1e-4).item(Bool.self))
    }

    @Test func aCancelledPassThrows() throws {
        let model = try loadedTinyModel()
        #expect(throws: YuE2Error.self) {
            _ = try model.prefixAttention(tokens: [1, 2, 3, 4], prefixLength: 2, segments: [0, -1], segmentCount: 1, cancel: { true })
        }
    }
}
