// PlannerTests.swift - Planner + SymbolicPlan save/load (T-2.8, weight-free tier 1)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

/// One id per character, `Int(scalar) % 1000` — same recipe as `ProtocolTests`' `FakeTokenizer`,
/// plus a `decode` that never runs here (the provided-ABC and `cot == .off` branches never call
/// it; `Planner`'s own ABC-generation branch, which does, needs real vocabulary bounds and is
/// exercised against the real checkpoint at T-2.9, not against this fake).
private struct FakeTokenizer: TextDecoding {
    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) % 1000 }
    }
    func decode(_ ids: [Int]) -> String {
        String(ids.map { Character(UnicodeScalar($0 % 128) ?? " ") })
    }
}

@Suite("Planner")
struct PlannerTests {
    private let tokenizer = FakeTokenizer()

    private func makeRequest(abc: String? = nil, cot: CoTMode = .full) throws -> SongRequest {
        try SongRequest(style: "piano pop", lyrics: "line one", cot: cot, abc: abc)
    }

    private func makePlanner() throws -> Planner {
        Planner(model: try loadedTinyModel(), tokenizer: tokenizer, config: try GenerationConfig())
    }

    /// `parity/protocol.json`'s real-tokenizer prefix shape doesn't apply here (`FakeTokenizer`
    /// doesn't produce those ids) — checking the *shape* instead: `eod` first, `abcEnd`/
    /// `musicStart` last, exactly as `Prefixes.tokenPrefixes` builds it for a provided score.
    @Test func planWithProvidedABCHasExpectedPrefixShape() throws {
        let request = try makeRequest(abc: "X:1\nK:C\nC4|")
        let plan = try makePlanner().plan(request: request)
        #expect(plan.prefix.first == YuE2Token.eod)
        #expect(Array(plan.prefix.suffix(2)) == [YuE2Token.abcEnd, YuE2Token.musicStart])
        #expect(plan.abc == request.abc)
        #expect(plan.abcIDs == tokenizer.encode(request.abc!))
        #expect(plan.timing == nil)
        #expect(plan.truncated == false)
    }

    /// `cot == .off` never generates or re-tokenizes anything: empty `abc`/`abcIDs`, and the
    /// prefix still ends `[abcEnd, musicStart]` (empty ABC span) per `Prefixes.tokenPrefixes`.
    @Test func planOffModeSkipsGenerationEntirely() throws {
        let plan = try makePlanner().plan(request: try makeRequest(cot: .off))
        #expect(plan.abc == nil)
        #expect(plan.abcIDs.isEmpty)
        #expect(plan.prefix.first == YuE2Token.eod)
        #expect(Array(plan.prefix.suffix(2)) == [YuE2Token.abcEnd, YuE2Token.musicStart])
    }

    @Test func saveThenLoadRoundTrips() throws {
        let plan = try makePlanner().plan(request: try makeRequest(abc: "X:1\nK:C\nC4|"))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        try plan.save(to: dir)
        #expect(try SymbolicPlan.load(from: dir) == plan)
    }

    /// Flipping one byte in `prefix.npy`'s payload must be caught by the manifest's SHA-256,
    /// not silently accepted (piège n°16: a saved plan is a trust boundary once on disk).
    @Test func loadRejectsTamperedPrefixArray() throws {
        let plan = try makePlanner().plan(request: try makeRequest(abc: "X:1\nK:C\nC4|"))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        try plan.save(to: dir)

        let prefixURL = dir.appendingPathComponent("prefix.npy")
        var bytes = try Data(contentsOf: prefixURL)
        bytes[bytes.count - 1] ^= 0xFF
        try bytes.write(to: prefixURL)

        #expect(throws: YuE2Error.self) {
            _ = try SymbolicPlan.load(from: dir)
        }
    }

    @Test func loadRejectsIncompleteManifest() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try JSONSerialization.data(withJSONObject: ["plan.json": "deadbeef"])
            .write(to: dir.appendingPathComponent("plan_manifest.json"))
        #expect(throws: YuE2Error.self) {
            _ = try SymbolicPlan.load(from: dir)
        }
    }

    @Test func loadRejectsMissingManifest() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: YuE2Error.self) {
            _ = try SymbolicPlan.load(from: dir)
        }
    }
}
