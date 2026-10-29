// ScoreLayoutTests.swift - score layout and lead-in/tail trimming (ASK Q16, weight-free tier 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

@Suite("Score layout")
struct ScoreLayoutTests {
    /// The shape of the iPhone's plan for ASK Q16 (`s20261017142348`), shortened: three silent
    /// intro groups (12 bars), a sung verse group, a chorus group, a two-group interlude.
    static let planned = """
        X:1
        T:
        M:4/4
        L:1/16
        Q:1/4=128
        V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
        V: Ins clef=treble name="Ins Melody" snm="Inst."
        K:Dm
        % intro
        V: Vocal
        "Dm"z16|"Dm"z16|"Dm"z16|"Dm"z16|
        V: Ins
        D4D4D4D4|D4D4D2D2D2DD|DDDDDDDDDDDDDDDD|DDDDDDDDDDDDDDDD|
        V: Vocal
        "Dm"z16|"Dm"z16|"Dm"z16|"Dm"z16|
        V: Ins
        D4z12|z8D2D2D4|D4z12|z8z2D2z4|
        V: Vocal
        "Dm"z16|"Dm"z16|"Dm"z16|"Dm"z16|
        V: Ins
        D4z12|z8D2D2D4|DDDDDDDDDDDDDDDD|DDDDDDDDDDDDDDD2|
        % verse
        V: Vocal
        "Dm"z4A4G4F4|"Dm"G4F2G4F2A4|"Bb"z16|"Bb"z16|
        V: Ins
        G4z12|Z|z12z2B,2|B,2B,2B,2B,2B,2B,2B,2B,2|
        % chorus
        V: Vocal
        "Dm"z6d2f2d4f2-|"Dm"f4d4f4d4|"Bb"a6g6f4|"Bb"d8z8|
        V: Ins
        Z4|
        % interlude
        V: Vocal
        "Dm"z16|"Dm"z16|"Dm"z16|"Dm"z16|
        V: Ins
        DDDDDDDDDDDDDDDD|DDDDDDDDDDDDDDDD|DDDDDDDDDDDDDDDD|DDDDDDDDD2D2D2D2|
        V: Vocal
        Z|
        V: Ins
        D12z4|

        """

    @Test func layoutFindsTheFirstSungBarAfterTheIntro() throws {
        let layout = try ABCScore(Self.planned).layout
        #expect(layout.bars == 25)
        #expect(layout.sections.map(\.label) == ["intro", "verse", "chorus", "interlude"])
        #expect(layout.sections.map(\.bars) == [12, 4, 4, 5])
        #expect(layout.sections.map(\.sungNotes) == [0, 8, 11, 0])
        #expect(layout.firstSungBar == 12)
        // 128 BPM, 4/4: a bar is 1.875 s; the voice enters on beat 2 of bar 13.
        #expect(abs((layout.firstSungSeconds ?? 0) - 22.969) < 1e-9)
        #expect(layout.lastSungBar == 19)
        #expect(abs((layout.lastSungEndSeconds ?? 0) - 36.563) < 1e-9)
        #expect(layout.seconds == 25 * 1.875)
    }

    @Test func layoutOfAnInstrumentalScoreHasNoSungBar() throws {
        let layout = try ABCScore("X:1\nM:4/4\nL:1/8\nK:C\n% intro\nV: Vocal\nZ2|\nV: Ins\nC8|C8|\n").layout
        #expect(layout.firstSungBar == nil && layout.lastSungEndSeconds == nil)
        #expect(layout.bars == 2)
    }

    @Test func aPlanCutInsideAChordIsLaidOutFromItsCompleteLines() throws {
        let cut = Self.planned.replacingOccurrences(of: "V: Vocal\nZ|\nV: Ins\nD12z4|\n", with: "V: Vocal\n\"D")
        #expect((try? ABCScore(cut)) == nil)
        let layout = try #require(ABCScore.layout(of: cut))
        #expect(layout.bars == 24 && layout.firstSungBar == 12)
        let request = try SongRequest(style: "techno", lyrics: "la", cot: .full)
        #expect(SymbolicPlan(request: request, abc: cut, abcIDs: [], prefix: [], truncated: true).layout == layout)
        #expect(try ABCScore(ABCScore.trimming(cut, tail: 0)).layout.bars == 20)
    }

    @Test func trimmingKeepsTheLastIntroGroupVerbatim() throws {
        let short = try ABCScore.trimming(Self.planned, leadIn: 4)
        let layout = try ABCScore(short).layout
        #expect(layout.firstSungBar == 4)
        #expect(layout.sections.map(\.bars) == [4, 4, 4, 5])
        #expect(short.contains("% intro\nV: Vocal\n\"Dm\"z16|\"Dm\"z16|\"Dm\"z16|\"Dm\"z16|\nV: Ins\nD4z12|z8D2D2D4|DDDDDDDDDDDDDDDD|"))
        #expect(!short.contains("D4D4D4D4|"))
        // Everything sung is untouched.
        let before = try ABCScore(Self.planned), after = try ABCScore(short)
        #expect(ABCScore.sounding(after.vocal).map(\.midi) == ABCScore.sounding(before.vocal).map(\.midi))
        #expect(short.hasSuffix("D12z4|\n"))
    }

    @Test func trimmingToNoLeadInDropsTheIntroLabel() throws {
        let short = try ABCScore.trimming(Self.planned, leadIn: 2)
        #expect(!short.contains("% intro"))
        #expect(try ABCScore(short).layout.firstSungBar == 0)
        #expect(short.hasPrefix("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=128\n"))
    }

    @Test func trimmingTheTailKeepsItsFirstGroup() throws {
        let short = try ABCScore.trimming(Self.planned, tail: 4)
        let layout = try ABCScore(short).layout
        #expect(layout.sections.map(\.bars) == [12, 4, 4, 4])
        #expect(!short.contains("D12z4|"))
        #expect(short.hasSuffix("\n"))
        #expect(try ABCScore.trimming(Self.planned, tail: 0).contains("% interlude") == false)
    }

    @Test func trimmingLeavesWhatAlreadyFitsAndInstrumentalsAlone() throws {
        #expect(try ABCScore.trimming(Self.planned, leadIn: 12, tail: 5) == Self.planned)
        let instrumental = "X:1\nM:4/4\nL:1/8\nK:C\n% intro\nV: Vocal\nZ2|\nV: Ins\nC8|C8|\n"
        #expect(try ABCScore.trimming(instrumental, leadIn: 0, tail: 0) == instrumental)
    }

    @Test func trimmingNeverDropsAGroupWithAnInlineField() throws {
        let abc = Self.planned.replacingOccurrences(
            of: "V: Ins\nD4z12|z8D2D2D4|D4z12|z8z2D2z4|", with: "V: Ins\nK:D\nD4z12|z8D2D2D4|D4z12|z8z2D2z4|")
        let short = try ABCScore.trimming(abc, leadIn: 4)
        #expect(short.contains("K:D\nD4z12|z8D2D2D4|D4z12|z8z2D2z4|"))
        #expect(try ABCScore(short).layout.firstSungBar == 8)
    }

    @Test func trimmingNeverSplitsTheVoicesOfAGroup() throws {
        // Ins written before Vocal: each line reads as a one-voice group, so nothing may go.
        let swapped = Self.planned.replacingOccurrences(
            of: "V: Vocal\n\"Dm\"z16|\"Dm\"z16|\"Dm\"z16|\"Dm\"z16|\nV: Ins\nD4D4D4D4|D4D4D2D2D2DD|DDDDDDDDDDDDDDDD|DDDDDDDDDDDDDDDD|",
            with: "V: Ins\nD4D4D4D4|D4D4D2D2D2DD|DDDDDDDDDDDDDDDD|DDDDDDDDDDDDDDDD|\nV: Vocal\n\"Dm\"z16|\"Dm\"z16|\"Dm\"z16|\"Dm\"z16|")
        #expect(swapped != Self.planned)
        let short = try ABCScore.trimming(swapped, leadIn: 0)
        #expect(short.contains("V: Ins\nD4D4D4D4|"))
        let layout = try ABCScore(short).layout
        #expect(layout.firstSungBar == 4)
        #expect(try ABCScore(short).vocal.count == ABCScore(short).ins.count)
    }

    @Test func planJSONCarriesTheLayout() throws {
        let request = try SongRequest(style: "techno", lyrics: "[Verse]\nla la", cot: .full, abc: Self.planned)
        let plan = SymbolicPlan(request: request, abc: Self.planned, abcIDs: [1, 2], prefix: [3])
        #expect(plan.layout?.firstSungBar == 12)
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(plan)) as? [String: Any]
        let layout = json?["layout"] as? [String: Any]
        #expect(layout?["firstSungBar"] as? Int == 12)
        // A plan.json written before the field existed still decodes, layout recomputed.
        var old = json ?? [:]
        old["layout"] = nil
        let decoded = try JSONDecoder().decode(SymbolicPlan.self, from: JSONSerialization.data(withJSONObject: old))
        #expect(decoded == plan)
    }
}
