// SheetSage2SymbolicTests.swift - vocabulary, grammar decoding and ABC notation parity, plan/14 S-4/S-5
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import SheetSage2Core

@Suite("SheetSage2Vocabulary")
struct SheetSage2VocabularyTests {
    /// Block bounds of the released 300 s / 100 Hz checkpoint, as exported by
    /// `sheetsage2_fixtures.py convert` (`vocabulary.json`).
    @Test func layoutMatchesTheReleasedCheckpoint() {
        let t = SheetSage2Tokenizer()
        #expect(t.subbeatShiftTokenStart == 260 && t.subbeatShiftTokenEnd == 517)
        #expect(t.timeTokenStart == 517 && t.timeTokenEnd == 30517)
        #expect(t.meterTokenStart == 30517 && t.meterTokenEnd == 30709)
        #expect(t.eighthPositionTokenStart == 30709 && t.eighthPositionTokenEnd == 30965)
        #expect(t.structureTokenStart == 30965 && t.structureTokenEnd == 30988)
        #expect(t.keyTokenStart == 30988 && t.keyTokenEnd == 31012)
        #expect(t.majminChordTokenStart == 31012 && t.majminChordTokenEnd == 31037)
        #expect(t.fullChordTokenStart == 31037 && t.fullChordTokenEnd == 31398)
        #expect(t.pitchTokenStart == 31398 && t.pitchTokenEnd == 31654)
        #expect(t.durationTokenStart == 31654 && t.nTokens == 31678)
        #expect(t.promptPrefix(SheetSage2Tokenizer.fullTaskPrompts) == [1, 4, 5, 6, 7, 9, 11, 3])
    }

    @Test func keyNamesAndChordSpellingFollowTheLM() {
        #expect(ChordSpelling.normalizeKeyName("D#:major") == "Eb:major")
        #expect(ChordSpelling.normalizeKeyName("A#:minor") == "Bb:minor")
        #expect(ChordSpelling.correctChordSpelling("D#:maj", keyName: "C:minor") == "Eb:maj")
        #expect(ChordSpelling.correctChordSpelling("C:maj", keyName: "E:major") == "C:maj")
        #expect(ChordSpelling.correctChordSpelling("A#:maj/3", keyName: "G:major") == "Bb:maj/3")
    }

    @Test func midiRoundTripMatchesPrettyMIDI() {
        // The written tempo is int(6e7 / (60 / (writeScale · 960))) = 500 000 exactly, so the
        // scale read back equals the write scale (checked against Python float64).
        #expect(MidiRoundTrip.readScale == MidiRoundTrip.writeScale)
        #expect(MidiRoundTrip.tick(1.0) == 1920)
        #expect(MidiRoundTrip.quantize([NoteRow(start: 0.1, end: 0.1001, pitch: 60, track: 0)]).isEmpty)
    }
}

private struct Sidecar: Decodable {
    let durationSeconds: Double
    let abcMelodyOnly: String?
    let abcFull: String?

    enum CodingKeys: String, CodingKey {
        case durationSeconds = "duration_seconds"
        case abcMelodyOnly = "abc_melody_only"
        case abcFull = "abc_full"
    }
}

private func fixtureReady(_ name: String) -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(atPath: dir.appendingPathComponent("parity/sheetsage2_\(name).json").path)
}

private func sidecar(_ name: String) throws -> Sidecar {
    let dir = try #require(modelsDir())
    let data = try Data(contentsOf: dir.appendingPathComponent("parity/sheetsage2_\(name).json"))
    return try JSONDecoder().decode(Sidecar.self, from: data)
}

/// Single-window stitching, exactly as `Transcriber.analyze` does for a song shorter than the window.
private func singleWindowEvents(_ tokens: [Int], duration: Double) throws -> [SheetSage2Event] {
    let tokenizer = SheetSage2Tokenizer()
    let decoded = try tokenizer.decodeGenerated(tokens)
    let timeMap = EventTimeMap(events: decoded.events, targetSeconds: 300)
    return stitchedWindowEvents(
        decoded.events, timeMap: timeMap, windowStart: 0, acceptStart: 0, acceptEnd: duration,
        songDuration: duration, windowIndex: 0, globalSubbeatBase: 0
    ).sorted { ($0.time!, $0.globalSubbeat) < ($1.time!, $1.globalSubbeat) }
}

/// S-5 gate on the reference tokens alone (CPU, no model): events → ABC identical to upstream.
@Suite("SheetSage2NotationParity", .enabled(if: fixtureReady("beethoven")))
struct SheetSage2NotationParityTests {
    private func check(_ name: String) throws {
        let reference = try sidecar(name)
        let fixture = try loadFixture(name: "sheetsage2_\(name)")
        let tokens = try #require(fixture["decoder.tokens"]).asArray(Int64.self).map(Int.init)
        let events = try singleWindowEvents(tokens, duration: reference.durationSeconds)
        let melodyOnly = try AbcNotation.abc(events: events, duration: reference.durationSeconds, melodyOnly: true)
        #expect(melodyOnly == reference.abcMelodyOnly, "\(name) melody-only ABC differs:\n\(melodyOnly)")
        let full = try AbcNotation.abc(events: events, duration: reference.durationSeconds, melodyOnly: false)
        #expect(full == reference.abcFull, "\(name) full ABC differs:\n\(full)")
    }

    @Test func beethoven() throws { try check("beethoven") }
    @Test(.enabled(if: fixtureReady("magic"))) func magic() throws { try check("magic") }
}

/// Measurement suites (minutes of GPU): run with `YUE2_SHEETSAGE2_MEASURE=1`.
private func measuring() -> Bool { ProcessInfo.processInfo.environment["YUE2_SHEETSAGE2_MEASURE"] == "1" }

private func modelReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return fixtureReady("beethoven")
        && FileManager.default.fileExists(atPath: dir.appendingPathComponent("SheetSage2-merged/model.safetensors").path)
}

/// S-4 gate (GPU): greedy grammar-constrained tokens identical to upstream, then the whole
/// transcriber's ABC identical.
@Suite("SheetSage2GenerationParity", .enabled(if: modelReady()))
struct SheetSage2GenerationParityTests {
    @Test func beethovenFloat32TokensAndABC() throws {
        let dir = try #require(modelsDir())
        let fixture = try loadFixture(name: "sheetsage2_beethoven")
        let reference = try sidecar("beethoven")
        let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: .float32)
        let waveform = try #require(fixture["input_waveform"])
        let expected = try #require(fixture["decoder.tokens"]).asArray(Int64.self).map(Int.init)

        let result = try SheetSage2Transcriber(model: model).transcribe(waveform, melodyOnly: true)
        let tokens = try #require(result.tokens.first)
        let firstDifference = zip(tokens, expected).enumerated().first { $0.element.0 != $0.element.1 }?.offset
        #expect(tokens == expected, "first difference at \(firstDifference.map(String.init) ?? "length") (\(tokens.count) vs \(expected.count))")
        #expect(result.abc == reference.abcMelodyOnly, "ABC differs:\n\(result.abc ?? result.abcError ?? "")")
    }
}

/// Reduced precision is judged on what it changes downstream: token agreement with the float32
/// reference and the ABC itself (measurement, recorded in docs/knowledge; not a parity gate).
@Suite("SheetSage2ReducedPrecision", .enabled(if: modelReady() && measuring()))
struct SheetSage2ReducedPrecisionTests {
    private func agreement(_ name: String, dtype: DType, subsamplingFloat32: Bool) throws {
        let dir = try #require(modelsDir())
        let fixture = try loadFixture(name: "sheetsage2_\(name)")
        let reference = try sidecar(name)
        let model = try SheetSage2Model.load(
            directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: dtype, subsamplingFloat32: subsamplingFloat32)
        let expected = try #require(fixture["decoder.tokens"]).asArray(Int64.self).map(Int.init)
        let result = try SheetSage2Transcriber(model: model).transcribe(try #require(fixture["input_waveform"]), melodyOnly: true)
        let tokens = try #require(result.tokens.first)
        let common = zip(tokens, expected).prefix { $0.0 == $0.1 }.count
        print("PRECISION \(name) dtype=\(dtype) subsampling_fp32=\(subsamplingFloat32) tokens=\(tokens.count)/\(expected.count) common_prefix=\(common) abc_identical=\(result.abc == reference.abcMelodyOnly)")
        if result.abc != reference.abcMelodyOnly { print("ABC \(name) \(dtype):\n\(result.abc ?? result.abcError ?? "")") }
    }

    @Test func reducedPrecisionAgreement() throws {
        for name in ["beethoven", "magic"] where fixtureReady(name) {
            try agreement(name, dtype: .bfloat16, subsamplingFloat32: false)
            try agreement(name, dtype: .bfloat16, subsamplingFloat32: true)
            try agreement(name, dtype: .float16, subsamplingFloat32: true)
        }
    }
}

private struct SongSidecar: Decodable {
    let windowTokens: [[Int]]
    let abcFull: String?
    let abcMelodyOnly: String?

    enum CodingKeys: String, CodingKey {
        case windowTokens = "window_tokens"
        case abcFull = "abc_full"
        case abcMelodyOnly = "abc_melody_only"
    }
}

private func songReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return modelReady() && FileManager.default.fileExists(atPath: dir.appendingPathComponent("parity/sheetsage2_song_b5mvt1.json").path)
}

/// Songs longer than one window: sliding windows, overlap prefixes and stitching (float32 gate).
@Suite("SheetSage2SongParity", .enabled(if: songReady()))
struct SheetSage2SongParityTests {
    @Test func beethovenFirstMovementFloat32() throws {
        let dir = try #require(modelsDir())
        let reference = try JSONDecoder().decode(
            SongSidecar.self, from: Data(contentsOf: dir.appendingPathComponent("parity/sheetsage2_song_b5mvt1.json")))
        let waveform = try #require(try loadFixture(name: "sheetsage2_song_b5mvt1")["input_waveform"])
        let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: .float32, subsamplingFloat32: false)
        let result = try SheetSage2Transcriber(model: model).transcribe(waveform, melodyOnly: false)
        #expect(result.tokens.count == reference.windowTokens.count, "windows \(result.tokens.count) vs \(reference.windowTokens.count)")
        for (index, (ours, theirs)) in zip(result.tokens, reference.windowTokens).enumerated() {
            let common = zip(ours, theirs).prefix { $0.0 == $0.1 }.count
            #expect(ours == theirs, "window \(index): common prefix \(common), \(ours.count) vs \(theirs.count) tokens")
        }
        #expect(result.abc == reference.abcFull, "full ABC differs:\n\(result.abc ?? result.abcError ?? "")")
        let melodyOnly = try AbcNotation.abc(events: result.events, duration: result.durationSeconds, melodyOnly: true)
        #expect(melodyOnly == reference.abcMelodyOnly, "melody-only ABC differs")
    }
}

/// float16 on a whole song (4 windows of up to 4 000 greedy steps): agreement with the float32
/// reference, per window, and the melody notes kept (measurement, not a gate).
@Suite("SheetSage2SongReducedPrecision", .enabled(if: songReady() && measuring()))
struct SheetSage2SongReducedPrecisionTests {
    @Test func float16Agreement() throws {
        let dir = try #require(modelsDir())
        let reference = try JSONDecoder().decode(
            SongSidecar.self, from: Data(contentsOf: dir.appendingPathComponent("parity/sheetsage2_song_b5mvt1.json")))
        let waveform = try #require(try loadFixture(name: "sheetsage2_song_b5mvt1")["input_waveform"])
        let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: .float16)
        let result = try SheetSage2Transcriber(model: model).transcribe(waveform, melodyOnly: true)
        for (index, (ours, theirs)) in zip(result.tokens, reference.windowTokens).enumerated() {
            let common = zip(ours, theirs).prefix { $0.0 == $0.1 }.count
            print("SONGFP16 window \(index) tokens \(ours.count)/\(theirs.count) common_prefix \(common)")
        }
        func lines(_ abc: String?) -> Set<Substring> { Set((abc ?? "").split(separator: "\n")) }
        let (a, b) = (lines(result.abc), lines(reference.abcMelodyOnly))
        print("SONGFP16 abc_identical=\(result.abc == reference.abcMelodyOnly) shared_lines=\(a.intersection(b).count)/\(b.count)")

        // Note-level agreement (pitch, track, onset within 50 ms) against the float32 transcription,
        // the metric of upstream's melody benchmarks; bfloat16 for scale.
        let fp32 = try SheetSage2Transcriber(model: try SheetSage2Model.load(
            directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: .float32, subsamplingFloat32: false)).transcribe(waveform)
        #expect(fp32.tokens == reference.windowTokens)
        let bf16 = try SheetSage2Transcriber(model: try SheetSage2Model.load(
            directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: .bfloat16)).transcribe(waveform)
        for (label, candidate) in [("fp16", result), ("bf16", bf16)] {
            let f1 = Self.noteF1(candidate.events, fp32.events)
            print(String(format: "SONGNOTES %@ precision %.3f recall %.3f f1 %.3f (notes %d vs %d)", label, f1.0, f1.1, f1.2, f1.3, f1.4))
        }
    }

    /// Greedy one-to-one matching on (track, pitch) with |Δonset| ≤ 50 ms.
    static func noteF1(_ test: [SheetSage2Event], _ reference: [SheetSage2Event]) -> (Double, Double, Double, Int, Int) {
        func notes(_ events: [SheetSage2Event]) -> [(Double, Int, Int)] {
            events.flatMap { e in (e.values.melody ?? []).map { (e.time!, $0.pitch, $0.track) } }
        }
        let (t, r) = (notes(test), notes(reference))
        var used = [Bool](repeating: false, count: r.count)
        var matched = 0
        for note in t {
            if let j = r.indices.first(where: { !used[$0] && r[$0].1 == note.1 && r[$0].2 == note.2 && abs(r[$0].0 - note.0) <= 0.05 }) {
                used[j] = true
                matched += 1
            }
        }
        let p = t.isEmpty ? 0 : Double(matched) / Double(t.count)
        let rc = r.isEmpty ? 0 : Double(matched) / Double(r.count)
        return (p, rc, p + rc == 0 ? 0 : 2 * p * rc / (p + rc), t.count, r.count)
    }
}

/// Profiles: the lean mechanics (chunked STFT and ConvNeXt, encoder released before decoding) must
/// not change the result in float32 (gate); each profile's fidelity is then measured.
@Suite("SheetSage2Profiles", .enabled(if: modelReady()))
struct SheetSage2ProfileTests {
    @Test func leanMechanicsKeepFloat32Parity() throws {
        let dir = try #require(modelsDir())
        for name in ["beethoven", "magic"] where fixtureReady(name) {
            let fixture = try loadFixture(name: "sheetsage2_\(name)")
            let reference = try sidecar(name)
            let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: .float32, subsamplingFloat32: false)
            model.chunkFrames = 2048
            let taps = try model.encodeWithTaps(try #require(fixture["input_waveform"]))
            #expect(relativeError(taps.memory[0], try #require(fixture["encoder.memory"])) <= 1e-3)
            var transcriber = SheetSage2Transcriber(model: model)
            transcriber.releaseEncoderAfterEncoding = true
            let result = try transcriber.transcribe(try #require(fixture["input_waveform"]))
            #expect(!model.encoderResident)
            #expect(result.tokens.first == (try #require(fixture["decoder.tokens"])).asArray(Int64.self).map(Int.init), "\(name) tokens")
            #expect(result.abc == reference.abcMelodyOnly, "\(name) ABC")
        }
    }

    @Test(.enabled(if: measuring())) func profileFidelity() throws {
        let dir = try #require(modelsDir())
        for profile in SheetSage2Profile.all {
            var line = "PROFILE \(profile.id)"
            for name in ["beethoven", "magic"] where fixtureReady(name) {
                let fixture = try loadFixture(name: "sheetsage2_\(name)")
                let expected = try #require(fixture["decoder.tokens"]).asArray(Int64.self).map(Int.init)
                let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-merged"), profile: profile)
                let result = try SheetSage2Transcriber(model: model, profile: profile).transcribe(try #require(fixture["input_waveform"]))
                let tokens = result.tokens.first ?? []
                let common = zip(tokens, expected).prefix { $0.0 == $0.1 }.count
                line += " | \(name) prefix \(common)/\(expected.count) abc_identical \(result.abc == (try sidecar(name)).abcMelodyOnly)"
            }
            if songReady() {
                let waveform = try #require(try loadFixture(name: "sheetsage2_song_b5mvt1")["input_waveform"])
                let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-merged"), profile: profile)
                let ours = try SheetSage2Transcriber(model: model, profile: profile).transcribe(waveform)
                let fp32 = try SheetSage2Transcriber(model: try SheetSage2Model.load(
                    directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: .float32, subsamplingFloat32: false)).transcribe(waveform)
                let f1 = SheetSage2SongReducedPrecisionTests.noteF1(ours.events, fp32.events)
                line += String(format: " | song note F1 %.3f", f1.2)
            }
            print(line)
        }
    }
}
