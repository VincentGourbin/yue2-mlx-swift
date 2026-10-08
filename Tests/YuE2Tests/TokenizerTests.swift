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
}
