// ProtocolTests.swift - protocol.py parity tests (constants, request, sampling, prefixes)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

/// One id per character, `Int(scalar) % 1000` — enough to exercise prefix shapes
/// without depending on the real tokenizer (arrives T-1.5).
private struct FakeTokenizer: TextEncoding {
    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) % 1000 }
    }
}

@Suite("Protocol")
struct ProtocolTests {
    private let tokenizer = FakeTokenizer()

    @Test func instructionsAreExact() {
        #expect(CoTMode.off.instruction == "Generate music with codec tokens from the given conditions.")
        #expect(CoTMode.melody.instruction == "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.")
        #expect(CoTMode.full.instruction == "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.")
    }

    @Test func textMatchesFormat() throws {
        let request = try SongRequest(style: "piano pop", lyrics: "line one\nline two")
        #expect(request.text() == "\(CoTMode.full.instruction)\n[Tags]\npiano pop\n[Lyrics]\nline one\nline two\n")
    }

    @Test func guidanceDefaults() throws {
        #expect(try SongRequest(style: "a", lyrics: "b", cot: .full).guidance == 1.0)
        #expect(try SongRequest(style: "a", lyrics: "b", cot: .off).guidance == 1.01)
        #expect(try SongRequest(style: "a", lyrics: "b", cfgScale: 3.5).guidance == 3.5)
    }

    @Test func tokenPrefixOffMode() throws {
        let request = try SongRequest(style: "a", lyrics: "b", cot: .off)
        let prefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer)
        #expect(prefix.first == YuE2Token.eod)
        #expect(Array(prefix.suffix(3)) == [YuE2Token.abcStart, YuE2Token.abcEnd, YuE2Token.musicStart])
        #expect(Array(prefix.dropFirst().dropLast(3)) == tokenizer.encode(request.text()))
    }

    @Test func tokenPrefixWithoutABCStopsAtABCStart() throws {
        let request = try SongRequest(style: "a", lyrics: "b", cot: .full)
        let prefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer)
        #expect(prefix.last == YuE2Token.abcStart)
    }

    @Test func tokenPrefixWithABCIDs() throws {
        let request = try SongRequest(style: "a", lyrics: "b", cot: .full)
        let prefix = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer, abcIDs: [17, 19, 23])
        #expect(Array(prefix.suffix(6)) == [YuE2Token.abcStart, 17, 19, 23, YuE2Token.abcEnd, YuE2Token.musicStart])
    }

    @Test func negativePrefixMatchesPositiveTail() throws {
        for cot in [CoTMode.full, CoTMode.melody] {
            let request = try SongRequest(style: "STYLE", lyrics: "LYRIC", cot: cot, cfgScale: 1.2)
            let ids = [17, 19, 23]
            let positive = try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer, abcIDs: ids)
            let negative = try Prefixes.negativePrefix(request: request, tokenizer: tokenizer, abcIDs: ids)
            let tail = [YuE2Token.abcStart, 17, 19, 23, YuE2Token.abcEnd, YuE2Token.musicStart]
            #expect(Array(positive.suffix(6)) == tail)
            #expect(Array(negative.suffix(6)) == tail)
            #expect(negative == [YuE2Token.eod] + tokenizer.encode(cot.instruction) + tail)
            #expect(positive != negative)
        }
    }

    @Test func negativePrefixOffMode() throws {
        let request = try SongRequest(style: "a", lyrics: "b", cot: .off)
        let negative = try Prefixes.negativePrefix(request: request, tokenizer: tokenizer)
        #expect(!negative.contains(YuE2Token.abcStart))
        #expect(negative.last == YuE2Token.musicStart)
    }

    @Test func negativePrefixRequiresABCIDsWhenNotOff() throws {
        let request = try SongRequest(style: "a", lyrics: "b", cot: .full)
        #expect(throws: YuE2Error.self) {
            try Prefixes.negativePrefix(request: request, tokenizer: tokenizer)
        }
    }

    @Test func chunkRangesSplitsEvenly() throws {
        let ranges = try Prefixes.chunkRanges(frames: 11, prefixTokens: 2, context: 15)
        #expect(ranges.map(\.0) == [0, 5, 10])
        #expect(ranges.map(\.1) == [5, 10, 11])
    }

    @Test func invalidCotIsRejected() throws {
        let json = Data(#"{"style":"a","lyrics":"b","cot":"bogus"}"#.utf8)
        #expect(throws: YuE2Error.self) {
            try JSONDecoder().decode(SongRequest.self, from: json)
        }
    }

    @Test func abcRequiresNotOffAndNonEmpty() throws {
        #expect(throws: YuE2Error.self) {
            try SongRequest(style: "a", lyrics: "b", cot: .off, abc: "X:1")
        }
        #expect(throws: YuE2Error.self) {
            try SongRequest(style: "a", lyrics: "b", cot: .full, abc: "   ")
        }
    }

    @Test func cfgScaleOutOfRangeIsRejected() throws {
        for value in [Double.nan, .infinity, -1, 21] {
            #expect(throws: YuE2Error.self) {
                try SongRequest(style: "a", lyrics: "b", cfgScale: value)
            }
        }
    }

    @Test func idMustBeFilenameSafe() throws {
        #expect(throws: YuE2Error.self) {
            try SongRequest(style: "a", lyrics: "b", id: "..")
        }
    }

    @Test func samplingRejectsInvalidTopK() throws {
        #expect(throws: YuE2Error.self) {
            try Sampling(topK: 0)
        }
    }

    @Test func samplingRejectsMinAboveMax() throws {
        #expect(throws: YuE2Error.self) {
            try Sampling(minTokens: 5, maxTokens: 4)
        }
    }

    @Test func abcIDsMustStayBelowEOD() throws {
        let request = try SongRequest(style: "a", lyrics: "b", cot: .full)
        #expect(throws: YuE2Error.self) {
            try Prefixes.tokenPrefixes(request: request, tokenizer: tokenizer, abcIDs: [YuE2Token.abcEnd])
        }
    }
}
