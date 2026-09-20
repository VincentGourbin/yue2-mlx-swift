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
