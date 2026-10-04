// StyleKeyTests.swift - key and tempo read from the style line, forced score header (ASK Q12)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

@Suite("StyleKey")
struct StyleKeyTests {
    @Test func readsKeyAndTempoFromStyleLines() {
        #expect(StyleKey.parse("upbeat pop, key of D major, 120 BPM").key == "D")
        #expect(StyleKey.parse("upbeat pop, key of D major, 120 BPM").bpm == 120)
        #expect(StyleKey.parse("French, afro-house, 131 BPM, A minor, deep bass").key == "Am")
        #expect(StyleKey.parse("ballad in F# minor").key == "F#m")
        #expect(StyleKey.parse("jazz, key of Bb").key == "Bb")
        #expect(StyleKey.parse("synthwave, E♭ major, 98 bpm").key == "Eb")
        #expect(StyleKey.parse("A# major anthem").key == "Bb")      // enharmonic signature
        #expect(StyleKey.parse("dark, D# minor").key == "D#m")       // a standard signature, kept
        #expect(StyleKey.parse("A minimal techno track, 128 BPM").key == nil)  // "A minimal" is not a key
        #expect(StyleKey.parse("rock with a big chorus").key == nil)
        #expect(StyleKey.parse("rock with a big chorus").bpm == nil)
    }

    @Test func forcedHeaderIsThePlannerDialectAndOnlyWhenFree() throws {
        let header = try #require(StyleKey.scoreHeader(style: "pop, key of G major, 72 BPM"))
        #expect(header == """
            X:1
            T:
            M:4/4
            L:1/16
            Q:1/4=72
            V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
            V: Ins clef=treble name="Ins Melody" snm="Inst."
            K:G

            """)
        let free = try SongRequest(style: "pop, key of G major, 72 BPM", lyrics: "[Verse]\nla", cot: .full, seed: 1)
        #expect(free.followingStyleKey().abcPrefix == header)
        var imposed = free
        imposed.abc = "X:1\n"
        #expect(imposed.followingStyleKey().abcPrefix == nil)
        let plain = try SongRequest(style: "rock", lyrics: "[Verse]\nla", cot: .full, seed: 1)
        #expect(plain.followingStyleKey().abcPrefix == nil)
    }
}
