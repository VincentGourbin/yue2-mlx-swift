// TokenizerTests.swift - YuE2Tokenizer vs. tiktoken corpus (tier 2, real tokenizer.json)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

/// `$YUE2_MODELS_DIR/YuE2-3B/tokenizer.json`, if the directory is set and the file was
/// converted by `Scripts/convert-tokenizer.py` (T-1.5) — else this suite skips.
private func readyModelDir() -> URL? {
    guard let path = ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"], !path.isEmpty else { return nil }
    let modelDir = URL(fileURLWithPath: path).appendingPathComponent("YuE2-3B")
    guard FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("tokenizer.json").path) else { return nil }
    return modelDir
}

private struct CorpusEntry: Decodable {
    let text: String
    let ids: [Int]
}

/// `#filePath` walked up to the repo root, independent of the test runner's cwd.
private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

@Suite("Tokenizer", .enabled(if: readyModelDir() != nil))
struct TokenizerTests {
    private let tokenizer: YuE2Tokenizer
    private let corpus: [CorpusEntry]

    init() async throws {
        tokenizer = try await YuE2Tokenizer.load(modelDir: readyModelDir()!)
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("parity/tokenizer_corpus.json"))
        corpus = try JSONDecoder().decode([CorpusEntry].self, from: data)
    }

    @Test func corpusMatchesTiktokenExactly() {
        for entry in corpus {
            #expect(tokenizer.encode(entry.text) == entry.ids)
        }
    }

    @Test func promptSegmentsFindEveryBarAndWord() throws {
        let abc = """
            X:1
            T:
            M:4/4
            L:1/32
            Q:1/4=120
            V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
            V: Ins clef=treble name="Ins Melody" snm="Inst."
            K:C
            % intro
            V: Vocal
            Z2|
            V: Ins
            C32|E32|
            % verse
            V: Vocal
            "C"c8d8e8f8|g32|
            V: Ins
            Z2|
            """
        // Accents split across tokens, an elided article, a dash that is not a word.
        let lyrics = "[Verse]\nÉté brûlant – l'été s'éveille\nà côté"
        let request = try SongRequest(style: "French chanson", lyrics: lyrics, abc: abc)
        let prefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer)
        let segments = try AttentionTimeline.segments(prefix: prefix, tokenizer: tokenizer, lyrics: lyrics, language: .french)
        #expect(segments.bars == 4)
        let parsed = LyricTimeline.parseLyrics(lyrics, language: .french).flatMap(\.words)
        #expect(segments.words == parsed.count && parsed.count == 6)
        // Every segment owns at least one token, in prompt order.
        let seen = segments.ids.filter { $0 >= 0 }
        #expect(Set(seen) == Set(0..<Int32(segments.bars + segments.words)))
        let words = seen.filter { $0 >= segments.bars }
        #expect(words == words.sorted())
        // The text of each word's tokens contains the word.
        for w in 0..<segments.words {
            let ids = prefix.indices.filter { segments.ids[$0] == Int32(segments.bars + w) }.map { prefix[$0] }
            #expect(tokenizer.decode(ids).contains(parsed[w].text))
        }
    }

    @Test func roundTripsThroughDecode() {
        for entry in corpus.prefix(3) {
            #expect(tokenizer.decode(tokenizer.encode(entry.text)) == entry.text)
        }
    }

    @Test func noIDReachesTheReservedRange() {
        for entry in corpus {
            #expect(tokenizer.encode(entry.text).allSatisfy { $0 >= 0 && $0 < YuE2Token.eod })
        }
    }

    /// A score edited after its cut keeps the planner's ids up to the changed line (ASK Q18).
    @Test func scoreSpliceKeepsTheUnchangedIDs() {
        let old = "X:1\nM:2/4\nL:1/32\nK:Bb\n% chorus\nV: Vocal\n\"Bb\"f16-|\"F\"f8d6c2|\n% outro\nV: Vocal\n\"Bb\"z16|\"F\"z16|\nV: Ins\nB2z6d2c2B4|c8d2c2B4|\n"
        let new = old.replacingOccurrences(of: "\"Bb\"z16|\"F\"z16|", with: "\"Bb\"z4B4c4d4|\"F\"c16|")
        let oldIDs = tokenizer.encode(old)
        let ids = SymbolicPlan.spliceScoreIDs(old: old, oldIDs: oldIDs, new: new, tokenizer: tokenizer)
        #expect(tokenizer.decode(ids) == new)
        // Every old token that ends before the changed line is kept as it was.
        let o = Array(old.utf8), n = Array(new.utf8)
        var lineStart = zip(o, n).prefix { $0 == $1 }.count
        while lineStart > 0, o[lineStart - 1] != 0x0A { lineStart -= 1 }
        var bytes = 0, kept = 0
        while kept < oldIDs.count, bytes + tokenizer.byteCount(of: oldIDs[kept]) <= lineStart {
            bytes += tokenizer.byteCount(of: oldIDs[kept]); kept += 1
        }
        #expect(kept > 10 && Array(ids.prefix(kept)) == Array(oldIDs.prefix(kept)))
        // The plan keeps its request and wraps the spliced ids in the usual prompt.
        let request = try! SongRequest(style: "pop", lyrics: "[verse]\nla la", cot: .full, seed: 1)
        let plan = SymbolicPlan(
            request: request, abc: old, abcIDs: oldIDs,
            prefix: try! Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer, abcIDs: oldIDs))
        let edited = try! plan.replacingScore(new, tokenizer: tokenizer)
        #expect(edited.abc == new && edited.abcIDs == ids && edited.request == request)
        #expect(edited.prefix == (try! Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer, abcIDs: ids)))
        #expect(try! plan.replacingScore(old, tokenizer: tokenizer).prefix == plan.prefix)
        // Ids that do not spell the old score: encoded whole.
        #expect(SymbolicPlan.spliceScoreIDs(old: old, oldIDs: [1, 2], new: new, tokenizer: tokenizer) == tokenizer.encode(new))
    }
}
