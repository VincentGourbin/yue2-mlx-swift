// AbcNotation.swift - timed events → validated two-voice ABC (exports_ + notation_sheetsage2.py port)
// Copyright 2026 Vincent Gourbin

import Foundation

public struct AbcNotationError: Error, CustomStringConvertible {
    public let description: String
    init(_ message: String) { description = message }
}

/// `[start, end, pitch, track]` note row of `export_result`.
struct NoteRow: Comparable {
    var start: Double
    var end: Double
    var pitch: Int
    var track: Int

    static func < (a: NoteRow, b: NoteRow) -> Bool {
        (a.start, a.end, a.pitch, a.track) < (b.start, b.end, b.pitch, b.track)
    }
}

/// `[time, beat_id, numerator, denominator]` beat row.
struct BeatRow {
    var time: Double
    var beatID: Int
    var numerator: Int
    var denominator: Int
}

/// Upstream exports the notation melody to a MIDI file (pretty_midi, 960 ticks per quarter at
/// 120 BPM) and reads it back, so note times are quantized exactly as below before notation.
enum MidiRoundTrip {
    static let resolution = 960.0
    static let writeScale = 60.0 / (120.0 * resolution)
    static let readScale: Double = {
        let tempo = Double(Int(6e7 / (60.0 / (writeScale * resolution))))  // int() truncates
        let bpm = 6e7 / tempo
        return 60.0 / (bpm * resolution)
    }()

    static func tick(_ time: Double) -> Int {
        time <= 0 ? 0 : Int((time / writeScale).rounded(.toNearestOrEven))
    }

    /// Notes per voice after the round trip; zero-length notes are dropped as pretty_midi does.
    static func quantize(_ notes: [NoteRow]) -> [NoteRow] {
        notes.compactMap { note in
            let (a, b) = (tick(note.start), tick(note.end))
            guard b != a else { return nil }
            return NoteRow(start: readScale * Double(a), end: readScale * Double(b), pitch: note.pitch, track: note.track)
        }
    }
}

/// Port of `export_result`'s ABC branch and of `notation_sheetsage2.py` (no file I/O).
public enum AbcNotation {
    // MARK: export_result

    static func intervalRows(_ events: [SheetSage2Event], _ field: String, duration: Double) -> [IntervalRow] {
        var rows = [IntervalRow]()
        for event in events {
            let value: String?
            switch field {
            case "chord": value = event.values.chord
            case "key": value = event.values.key
            default: value = event.values.structure
            }
            if let value { rows.append(IntervalRow(start: event.time!, end: 0, label: value)) }
        }
        for i in rows.indices { rows[i].end = i + 1 < rows.count ? rows[i + 1].start : duration }
        return rows.filter { $0.end > $0.start }
    }

    static func rhythmRows(_ events: [SheetSage2Event]) throws -> [BeatRow] {
        var rows = [BeatRow]()
        var meter: (Int, Int)?
        for event in events {
            if let m = event.values.meter { meter = m }
            guard let eighth = event.values.eighthPosition, let (numerator, denominator) = meter else { continue }
            let scaled = eighth * denominator
            guard scaled % 8 == 0 else {
                throw AbcNotationError("Eighth position \(eighth) is off the \(numerator)/\(denominator) beat grid")
            }
            let position = scaled / 8
            guard position >= 0 && position < numerator else {
                throw AbcNotationError("Eighth position \(eighth) is outside meter (\(numerator), \(denominator))")
            }
            rows.append(BeatRow(time: event.time!, beatID: position + 1, numerator: numerator, denominator: denominator))
        }
        return rows
    }

    /// `notation_notes`: per track, clip each note to the next onset, drop the empty ones.
    static func notationNotes(_ notes: [NoteRow]) -> [NoteRow] {
        var result = [NoteRow]()
        for track in 0...1 {
            var ordered = notes.filter { $0.track == track }.sorted { ($0.start, $0.pitch, $0.end) < ($1.start, $1.pitch, $1.end) }
            for i in ordered.indices {
                if i + 1 < ordered.count && ordered[i].end > ordered[i + 1].start + 1e-6 { ordered[i].end = ordered[i + 1].start }
                if ordered[i].end > ordered[i].start + 1e-6 { result.append(ordered[i]) }
            }
        }
        return result.sorted()
    }

    /// `export_result` (ABC part): stitched events of a song → ABC text.
    public static func abc(events: [SheetSage2Event], duration: Double, melodyOnly: Bool) throws -> String {
        var notes = [NoteRow]()
        for event in events {
            let start = event.time!
            for note in event.values.melody ?? [] {
                let end = min(duration, note.endTime!)
                if end > start { notes.append(NoteRow(start: start, end: end, pitch: note.pitch, track: note.track)) }
            }
        }
        notes.sort()

        let beats = try rhythmRows(events)
        guard beats.count >= 2 else { throw AbcNotationError("At least two decoded beats are required for ABC") }
        var abcBeats = beats
        let tail = beats.suffix(9).map(\.time)
        let period = EventTimeMap.median((1..<tail.count).map { tail[$0] - tail[$0 - 1] })
        guard period > 0 else { throw AbcNotationError("Decoded beats must increase in time") }
        let end = max(duration, notes.map(\.end).max() ?? 0)
        while abcBeats.last!.time < end - 1e-6 {
            let previous = abcBeats.last!
            abcBeats.append(BeatRow(
                time: previous.time + period, beatID: previous.beatID % previous.numerator + 1,
                numerator: previous.numerator, denominator: previous.denominator))
        }
        let (domainStart, domainEnd) = (abcBeats[0].time, abcBeats[abcBeats.count - 1].time)
        func clip(_ rows: [IntervalRow]) -> [IntervalRow] {
            rows.filter { $0.end > domainStart && $0.start < domainEnd }
                .map { IntervalRow(start: max(domainStart, $0.start), end: min(domainEnd, $0.end), label: $0.label) }
        }
        let rawKeys = intervalRows(events, "key", duration: duration)
        let chords = clip(ChordSpelling.correctChordRows(intervalRows(events, "chord", duration: duration), keys: rawKeys))
        let keys = clip(rawKeys.map { IntervalRow(start: $0.start, end: $0.end, label: ChordSpelling.normalizeKeyName($0.label)) })
        let structures = clip(intervalRows(events, "structure", duration: duration))
        guard !rawKeys.isEmpty else { throw AbcNotationError("No key was decoded; cannot construct a keyed ABC score") }

        let melody = MidiRoundTrip.quantize(notationNotes(notes))
        let score = try buildScore(
            melody: melody, beats: abcBeats, chords: chords, keys: keys, structures: structures, melodyOnly: melodyOnly)
        return try scoreToABC(score)
    }

    // MARK: notation_sheetsage2.py

    static let subbeatDivision = 4
    static let voiceIDs = ["Vocal", "Ins"]
    static let noChords: Set<String> = ["N", "X", "?"]

    struct Beat {
        var time: Double
        var beatID: Int
        var declaredNumerator: Int
        var denominator: Int
        var lineNo: Int
    }

    struct Measure {
        var index: Int
        var startBeat: Int
        var endBeat: Int
        var numerator: Int
        var denominator: Int
        var pickup = false
        var partial = false
        var inferred = false
        var notatedNumerator: Int?
        var notatedDenominator: Int?
        var padBefore = false

        var startT: Int { startBeat * AbcNotation.subbeatDivision }
        var endT: Int { endBeat * AbcNotation.subbeatDivision }
        var abcNumerator: Int { notatedNumerator ?? numerator }
        var abcDenominator: Int { notatedDenominator ?? denominator }
    }

    struct Score {
        var measures: [Measure]
        var subbeatTimes: [Double]
        var subbeatQuarters: [Double]
        var subbeatDenominators: [Int]
        var keyArr: [String]
        var chordArr: [String]
        var structureEvents: [(Int, String)]
        var voiceArrs: [String: [Int]]
    }

    struct MeasureGroup {
        var measures: [Measure]
        var structureLabels: [String]
        var meterChanged: Bool
        var keyChanged: Bool
    }

    static let qualityToABC: [String: String] = [
        "maj": "", "min": "m", "dim": "dim", "aug": "aug", "7": "7", "maj7": "maj7", "min7": "m7",
        "dim7": "dim7", "hdim7": "m7b5", "sus4": "sus4", "sus2": "sus2", "maj6": "6", "min6": "m6",
        "sus4(b7)": "7sus4", "minmaj7": "m(maj7)",
    ]
    static let naturalPitchClass: [Character: Int] = ["C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11]
    static let letters = Array("CDEFGAB")
    static let sharpPitchNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    static let flatPitchNames = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    static let keySignatureAccidentals: [String: Int] = [
        "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7, "F": -1, "Bb": -2, "Eb": -3,
        "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7, "Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5,
        "D#m": 6, "A#m": 7, "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7,
    ]
    static let keyRelativePitchNames: [Int: [String]] = [
        7: ["B#", "C#", "C##", "D#", "D##", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
        6: ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "G##", "A#", "B"],
        5: ["B#", "C#", "C##", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        4: ["B#", "C#", "D", "D#", "E", "E#", "F#", "F##", "G#", "A", "A#", "B"],
        3: ["B#", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        2: ["C", "C#", "D", "D#", "E", "E#", "F#", "G", "G#", "A", "A#", "B"],
        1: ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
        0: ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        -1: ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"],
        -2: ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        -3: ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        -4: ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"],
        -5: ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        -6: ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "A", "Bb", "Cb"],
        -7: ["C", "Db", "D", "Eb", "Fb", "F", "Gb", "G", "Ab", "Bbb", "Bb", "Cb"],
    ]

    // Parsing (`_parse_beats`, `_parse_keys`, …): same validations as upstream.

    static func parseBeats(_ rows: [BeatRow]) throws -> [Beat] {
        var beats = [Beat]()
        for (i, row) in rows.enumerated() {
            let beat = Beat(time: row.time, beatID: row.beatID, declaredNumerator: row.numerator, denominator: row.denominator, lineNo: i + 1)
            guard beat.beatID >= 1 else { throw AbcNotationError("beats:\(i + 1): beat ID must be positive") }
            guard beat.declaredNumerator >= 1 else { throw AbcNotationError("beats:\(i + 1): meter numerator must be positive") }
            guard beat.denominator >= 1, beat.denominator & (beat.denominator - 1) == 0 else {
                throw AbcNotationError("beats:\(i + 1): meter denominator must be a positive power of two")
            }
            if let last = beats.last, beat.time <= last.time {
                throw AbcNotationError("beats:\(i + 1): beat times must be strictly increasing")
            }
            beats.append(beat)
        }
        guard beats.count >= 2 else { throw AbcNotationError("beats: at least two beat events are required") }
        return beats
    }

    static func parseIntervals(_ rows: [IntervalRow], kind: String, transform: (String) throws -> String) throws -> [IntervalRow] {
        var result = [IntervalRow]()
        var previousEnd: Double?
        for (i, row) in rows.enumerated() {
            let label = row.label.trimmingCharacters(in: .whitespaces)
            guard row.end > row.start else { throw AbcNotationError("\(kind):\(i + 1): \(kind) end must be after start") }
            if let previousEnd, row.start < previousEnd - 1e-6 {
                throw AbcNotationError("\(kind):\(i + 1): overlapping \(kind) intervals")
            }
            result.append(IntervalRow(start: row.start, end: row.end, label: try transform(label)))
            previousEnd = row.end
        }
        return result
    }

    static func modeWithFirstTiebreak(_ values: [Int]) -> Int {
        var counts = [Int: Int]()
        for v in values { counts[v, default: 0] += 1 }
        let maximum = counts.values.max()!
        return values.first { counts[$0] == maximum }!
    }

    /// `infer_measures(meter_conflict="infer")`.
    static func inferMeasures(_ beats: [Beat]) throws -> [Measure] {
        let downbeats = beats.indices.filter { beats[$0].beatID == 1 }
        guard let firstDownbeat = downbeats.first, let lastDownbeat = downbeats.last else {
            throw AbcNotationError("No downbeat (beat ID 1) exists in the beat lab")
        }
        var spans = [(Int, Int, Bool, Bool)]()
        if firstDownbeat > 0 { spans.append((0, firstDownbeat, true, false)) }
        for (a, b) in zip(downbeats, downbeats.dropFirst()) { spans.append((a, b, false, false)) }
        if lastDownbeat < beats.count - 1 { spans.append((lastDownbeat, beats.count - 1, false, true)) }
        guard !spans.isEmpty else { throw AbcNotationError("No positive-length measure exists between downbeats") }

        var measures = [Measure]()
        for (index, (start, end, pickup, partial)) in spans.enumerated() {
            let events = Array(beats[start..<end])
            let count = events.count
            guard count >= 1 else { throw AbcNotationError("Measure \(index): empty downbeat span") }
            let ids = events.map(\.beatID)
            guard ids == Array(ids[0] ..< ids[0] + count) else {
                throw AbcNotationError("Measure \(index): non-consecutive beat IDs \(ids)")
            }
            if !pickup && ids[0] != 1 { throw AbcNotationError("Measure \(index): full measure does not start at beat ID 1") }
            let denominators = events.map(\.denominator)
            let denominator = modeWithFirstTiebreak(denominators)
            let declared = events.map(\.declaredNumerator)
            let declaredNumerator = modeWithFirstTiebreak(declared)
            let numeratorConflict = declared.contains { $0 != count }
            let denominatorConflict = denominators.contains { $0 != denominator }
            let padFinalPartial = partial && Set(declared).count == 1 && !denominatorConflict && declaredNumerator >= count
            measures.append(Measure(
                index: index, startBeat: start, endBeat: end, numerator: count, denominator: denominator,
                pickup: pickup, partial: partial,
                inferred: pickup || partial || numeratorConflict || denominatorConflict,
                notatedNumerator: padFinalPartial ? declaredNumerator : count))
        }
        if measures.count >= 2 {
            let (first, following) = (measures[0], measures[1])
            let firstDuration = Double(first.numerator) / Double(first.denominator)
            let followingDuration = Double(following.abcNumerator) / Double(following.abcDenominator)
            if firstDuration < followingDuration {
                measures[0].inferred = true
                measures[0].notatedNumerator = following.abcNumerator
                measures[0].notatedDenominator = following.abcDenominator
                measures[0].padBefore = true
            }
        }
        return measures
    }

    static func buildGrid(_ beats: [Beat], _ measures: [Measure]) throws -> ([Double], [Double], [Int]) {
        var intervalDenominators = [Int](repeating: 0, count: beats.count - 1)
        for measure in measures {
            for i in measure.startBeat..<measure.endBeat { intervalDenominators[i] = measure.denominator }
        }
        guard !intervalDenominators.contains(0) else { throw AbcNotationError("Downbeat spans do not cover every beat interval") }
        var times = [Double]()
        var denominators = [Int]()
        var quarters = [0.0]
        var current = 0.0
        for index in 0..<(beats.count - 1) {
            let (start, end) = (beats[index].time, beats[index + 1].time)
            let denominator = intervalDenominators[index]
            times.append(contentsOf: linspaceExcludingEnd(start, end, subbeatDivision + 1))
            denominators.append(contentsOf: [Int](repeating: denominator, count: subbeatDivision))
            let step = 4.0 / Double(denominator) / Double(subbeatDivision)
            for _ in 0..<subbeatDivision {
                current += step
                quarters.append(current)
            }
        }
        times.append(beats[beats.count - 1].time)
        denominators.append(intervalDenominators[intervalDenominators.count - 1])
        return (times, quarters, denominators)
    }

    /// `np.linspace(start, end, num)[:-1]` with numpy's exact arithmetic
    /// (`start + i·step`, `step = (end − start) / (num − 1)`).
    static func linspaceExcludingEnd(_ start: Double, _ end: Double, _ num: Int) -> [Double] {
        let div = Double(num - 1)
        let delta = end - start
        let step = delta / div
        return (0..<(num - 1)).map { start + Double($0) * step }
    }

    static func boundaries(_ times: [Double]) -> [Double] {
        (1..<times.count).map { (times[$0 - 1] + times[$0]) / 2 }
    }

    static func fillIntervals(_ rows: [IntervalRow], boundaries: [Double], count: Int, default value: String) throws -> [String] {
        var result = [String](repeating: value, count: count)
        for row in rows {
            let start = max(0, min(searchSortedLeft(boundaries, row.start), count - 1))
            let end = max(0, min(searchSortedLeft(boundaries, row.end), count - 1))
            guard end > start else {
                throw AbcNotationError(String(format: "Interval %.6f-%.6f (%@) is shorter than the ABC subbeat grid", row.start, row.end, row.label))
            }
            for t in start..<end { result[t] = row.label }
        }
        if count > 1 { result[count - 1] = result[count - 2] }
        return result
    }

    static func notesToArr(_ notes: [NoteRow], boundaries: [Double], count: Int, voice: String) throws -> [Int] {
        var result = [Int](repeating: 0, count: count)
        for note in notes.sorted(by: { ($0.start, $0.end, $0.pitch) < ($1.start, $1.end, $1.pitch) }) {
            let start = max(0, min(searchSortedLeft(boundaries, note.start), count - 1))
            let end = max(0, min(searchSortedLeft(boundaries, note.end), count - 1))
            guard end > start else {
                throw AbcNotationError(String(format: "%@: MIDI note pitch=%d at %.6f-%.6f cannot be represented on the decoded subbeat grid", voice, note.pitch, note.start, note.end))
            }
            guard result[start..<end].allSatisfy({ $0 == 0 }) else {
                throw AbcNotationError("\(voice): overlapping quantized melody notes at subbeats \(start):\(end)")
            }
            let sustain = note.pitch * 2 + 2
            for t in start..<end { result[t] = sustain }
            result[start] = sustain + 1
        }
        return result
    }

    static func pitchClass(_ root: String) throws -> (Int, Character, String) {
        guard let letter = root.first, let natural = naturalPitchClass[letter] else {
            throw AbcNotationError("Invalid pitch spelling '\(root)'")
        }
        let accidental = String(root.dropFirst())
        guard ["", "#", "##", "b", "bb"].contains(accidental) else { throw AbcNotationError("Invalid pitch spelling '\(root)'") }
        let offset = accidental.filter { $0 == "#" }.count - accidental.filter { $0 == "b" }.count
        return (floorMod(natural + offset, 12), letter, accidental)
    }

    static func portablePitchName(_ root: String, preserveDouble: Bool = false) throws -> String {
        let (pc, _, accidental) = try pitchClass(root)
        if preserveDouble || accidental.count <= 1 { return root }
        return (accidental.hasPrefix("#") ? sharpPitchNames : flatPitchNames)[pc]
    }

    static func bassDegreeToPitch(root: String, degree text: String) throws -> String {
        if (try? pitchClass(text)) != nil { return try portablePitchName(text, preserveDouble: true) }
        let accidental = String(text.prefix { $0 == "#" || $0 == "b" })
        guard ["", "#", "##", "b", "bb"].contains(accidental), let degree = Int(text.dropFirst(accidental.count)),
            (1...13).contains(degree)
        else { throw AbcNotationError("Invalid chord bass degree '\(text)'") }
        let (rootPC, rootLetter, rootAccidental) = try pitchClass(root)
        let scaleSemitones = [0, 2, 4, 5, 7, 9, 11]
        var interval = scaleSemitones[(degree - 1) % 7] + 12 * ((degree - 1) / 7)
        interval += accidental.filter { $0 == "#" }.count - accidental.filter { $0 == "b" }.count
        let targetPC = floorMod(rootPC + interval, 12)
        let targetLetter = letters[(letters.firstIndex(of: rootLetter)! + degree - 1) % 7]
        let difference = floorMod(targetPC - naturalPitchClass[targetLetter]! + 6, 12) - 6
        if let suffix = [-2: "bb", -1: "b", 0: "", 1: "#", 2: "##"][difference] { return String(targetLetter) + suffix }
        return ((rootAccidental + accidental).contains("#") ? sharpPitchNames : flatPitchNames)[targetPC]
    }

    static func chordSymbolToABC(_ chord: String) throws -> String? {
        let chord = chord.trimmingCharacters(in: .whitespaces)
        if noChords.contains(chord) { return nil }
        guard let colon = chord.firstIndex(of: ":") else { throw AbcNotationError("Chord '\(chord)' is missing the ':' quality separator") }
        let root = String(chord[..<colon])
        let descriptor = String(chord[chord.index(after: colon)...])
        var quality = descriptor
        var bass: String?
        if let slash = descriptor.firstIndex(of: "/") {
            quality = String(descriptor[..<slash])
            bass = String(descriptor[descriptor.index(after: slash)...])
        }
        guard let abcQuality = qualityToABC[quality] else {
            throw AbcNotationError("Unsupported chord quality '\(quality)' in '\(chord)'; refusing to rewrite it as major")
        }
        var text = try portablePitchName(root, preserveDouble: true) + abcQuality
        if let bass, !bass.isEmpty { text += "/" + (try bassDegreeToPitch(root: root, degree: bass)) }
        return text
    }

    static func keySymbolToABC(_ key: String) throws -> String {
        let key = key.trimmingCharacters(in: .whitespaces)
        var root: String
        var minor: Bool
        if let colon = key.firstIndex(of: ":") {
            root = String(key[..<colon])
            let mode = String(key[key.index(after: colon)...])
            guard mode == "major" || mode == "minor" else { throw AbcNotationError("Unsupported key mode '\(mode)' in '\(key)'") }
            minor = mode == "minor"
        } else if key.hasSuffix("m") {
            (root, minor) = (String(key.dropLast()), true)
        } else {
            (root, minor) = (key, false)
        }
        let (pc, _, accidental) = try pitchClass(root)
        let suffix = minor ? "m" : ""
        var candidate = try portablePitchName(root) + suffix
        if keySignatureAccidentals[candidate] != nil { return candidate }
        let names = accidental.contains("b") ? flatPitchNames : sharpPitchNames
        candidate = names[pc] + suffix
        if keySignatureAccidentals[candidate] == nil {
            candidate = (names == flatPitchNames ? sharpPitchNames : flatPitchNames)[pc] + suffix
        }
        guard keySignatureAccidentals[candidate] != nil else { throw AbcNotationError("Cannot encode portable ABC key for '\(key)'") }
        return candidate
    }

    static func keyAccidentals(_ key: String) throws -> [Int] {
        guard let count = keySignatureAccidentals[key] else { throw AbcNotationError("Unsupported ABC key signature '\(key)'") }
        var accidentals = [Int](repeating: 0, count: 7)
        let order = Array(count > 0 ? "FCGDAEB" : "BEADGCF")
        for letter in order.prefix(abs(count)) { accidentals[letters.firstIndex(of: letter)!] = count > 0 ? 1 : -1 }
        return accidentals
    }

    static func noteToABC(_ note: Int, keyAccidentals: [Int], measureAccidentals: inout [Int: Int]) throws -> String {
        let count = keyAccidentals.reduce(0, +)
        guard let names = keyRelativePitchNames[count] else {
            throw AbcNotationError("Unsupported key signature accidental count \(count)")
        }
        let pitchName = names[floorMod(note, 12)]
        var letter = String(pitchName.prefix(1))
        let accidentalNumber = ["": 0, "#": 1, "##": 2, "b": -1, "bb": -2][String(pitchName.dropFirst())]!
        var octave = floorDiv(note - 60, 12)
        if floorMod(note, 12) == 11 && accidentalNumber == -1 { octave += 1 }
        else if floorMod(note, 12) == 0 && accidentalNumber == 1 { octave -= 1 }
        let scaleIndex = letters.firstIndex(of: Character(letter))!
        let current = measureAccidentals[scaleIndex] ?? keyAccidentals[scaleIndex]
        var accidentalText = ""
        if current != accidentalNumber {
            measureAccidentals[scaleIndex] = accidentalNumber
            accidentalText = [-2: "__", -1: "_", 0: "=", 1: "^", 2: "^^"][accidentalNumber]!
        }
        if octave > 0 {
            letter = letter.lowercased()
            if octave > 1 { letter += String(repeating: "'", count: octave - 1) }
        } else if octave < 0 {
            letter += String(repeating: ",", count: -octave)
        }
        return accidentalText + letter
    }

    static func buildScore(
        melody: [NoteRow], beats beatRows: [BeatRow], chords chordRows: [IntervalRow], keys keyRows: [IntervalRow],
        structures structureRows: [IntervalRow], melodyOnly: Bool
    ) throws -> Score {
        let beats = try parseBeats(beatRows)
        let keys = try parseIntervals(keyRows, kind: "key", transform: keySymbolToABC)
        guard !keys.isEmpty else { throw AbcNotationError("keys: at least one key interval is required") }
        let structures = try parseIntervals(structureRows, kind: "structure") { $0 }
        let chords = melodyOnly ? [] : try parseIntervals(chordRows, kind: "chord") { label in
            _ = try chordSymbolToABC(label)
            return label
        }
        let measures = try inferMeasures(beats)
        let (times, quarters, denominators) = try buildGrid(beats, measures)
        let bounds = boundaries(times)
        var voices = [String: [Int]]()
        for (track, voice) in voiceIDs.enumerated() {
            voices[voice] = try notesToArr(melody.filter { $0.track == track }, boundaries: bounds, count: times.count, voice: voice)
        }
        let keyArr = try fillIntervals(keys, boundaries: bounds, count: times.count, default: keys[0].label)
        let chordArr = melodyOnly
            ? [String](repeating: "N", count: times.count)
            : try fillIntervals(chords, boundaries: bounds, count: times.count, default: "N")
        let structureEvents = structures.map { row in
            (max(0, min(searchSortedLeft(bounds, row.start), times.count - 1)), row.label)
        }
        return Score(
            measures: measures, subbeatTimes: times, subbeatQuarters: quarters, subbeatDenominators: denominators,
            keyArr: keyArr, chordArr: chordArr, structureEvents: structureEvents, voiceArrs: voices)
    }

    // MARK: serialization

    static func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }

    static func unitDenominator(_ score: Score) throws -> Int {
        var value = 1
        for measure in score.measures {
            for d in [measure.denominator, measure.abcDenominator] {
                let x = d * subbeatDivision
                value = value / gcd(value, x) * x
            }
        }
        guard value <= 1024 else { throw AbcNotationError("Required ABC unit length 1/\(value) is unreasonably small") }
        return value
    }

    static func paddingUnits(_ m: Measure, _ unit: Int) -> Int {
        m.abcNumerator * unit / m.abcDenominator - m.numerator * unit / m.denominator
    }

    static func durationUnits(_ score: Score, _ start: Int, _ end: Int, _ unit: Int) throws -> Int {
        var units = 0
        for t in start..<max(start, end) {
            let divisor = score.subbeatDenominators[t] * subbeatDivision
            guard unit % divisor == 0 else { throw AbcNotationError("ABC L:1/\(unit) cannot express a 1/\(divisor) subbeat exactly") }
            units += unit / divisor
        }
        return units
    }

    static func estimateTempo(_ score: Score) throws -> Double {
        let seconds = score.subbeatTimes.last! - score.subbeatTimes.first!
        let quarters = score.subbeatQuarters.last! - score.subbeatQuarters.first!
        guard seconds > 0 && quarters > 0 else { throw AbcNotationError("Cannot estimate tempo from a zero-duration score") }
        return quarters / seconds * 60.0
    }

    static func continuesPitch(_ value: Int, _ next: Int) -> Bool {
        value > 0 && next == (value / 2 - 1) * 2 + 2
    }

    static func sameNoteSegment(_ value: Int, _ next: Int) -> Bool {
        value == 0 ? next == 0 : next == (value / 2 - 1) * 2 + 2
    }

    static let supportedDurations = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]

    static func splitDuration(_ duration: Int) throws -> [Int] {
        guard duration > 0 else { throw AbcNotationError("Cannot serialize non-positive duration \(duration)") }
        var result = [Int]()
        var remaining = duration
        while remaining > 0 {
            if supportedDurations.contains(remaining) {
                result.append(remaining)
                break
            }
            guard let chunk = supportedDurations.filter({ $0 < remaining }).max() else {
                throw AbcNotationError("Duration \(duration) cannot be split into representable ABC values")
            }
            result.append(chunk)
            remaining -= chunk
        }
        return result
    }

    static func durationTokens(prefix: String, note: String, duration: Int, tieOut: Bool) throws -> [String] {
        let chunks = try splitDuration(duration)
        return chunks.enumerated().map { index, chunk in
            let continues = note != "z" && (index + 1 < chunks.count || tieOut)
            return (index == 0 ? prefix : "") + note + (chunk == 1 ? "" : String(chunk)) + (continues ? "-" : "")
        }
    }

    static func renderVoiceMeasure(_ score: Score, voice voiceID: String, measure: Measure, unit: Int) throws -> String {
        let voice = score.voiceArrs[voiceID]!
        let showChords = voiceID == "Vocal"
        var measureAccidentals = [Int: Int]()
        var currentKey = score.keyArr[measure.startT]
        var keyAccidentals = try self.keyAccidentals(currentKey)
        var parts = [String]()
        let padding = paddingUnits(measure, unit)
        guard padding >= 0 else { throw AbcNotationError("Measure \(measure.index): notated meter is shorter than its decoded span") }
        var leading = measure.padBefore ? padding : 0
        var trailing = measure.padBefore ? 0 : padding
        var t = measure.startT
        while t < measure.endT {
            var changes = [measure.endT]
            if let probe = ((t + 1)..<measure.endT).first(where: { !sameNoteSegment(voice[t], voice[$0]) }) { changes.append(probe) }
            if let probe = ((t + 1)..<measure.endT).first(where: { score.keyArr[$0] != score.keyArr[$0 - 1] }) { changes.append(probe) }
            if showChords, let probe = ((t + 1)..<measure.endT).first(where: { score.chordArr[$0] != score.chordArr[$0 - 1] }) {
                changes.append(probe)
            }
            let next = changes.min()!

            var prefix = ""
            let key = score.keyArr[t]
            if t > measure.startT && key != currentKey {
                currentKey = key
                keyAccidentals = try self.keyAccidentals(currentKey)
                measureAccidentals = [:]
                prefix += "[K:\(currentKey)]"
            }
            if showChords && (t == measure.startT || score.chordArr[t] != score.chordArr[t - 1]),
                let chordText = try chordSymbolToABC(score.chordArr[t])
            {
                prefix += "\"\(chordText)\""
            }
            let value = voice[t]
            let noteText = value == 0 ? "z" : try noteToABC(value / 2 - 1, keyAccidentals: keyAccidentals, measureAccidentals: &measureAccidentals)
            var duration = try durationUnits(score, t, next, unit)
            if t == measure.startT && leading > 0 {
                if value == 0 && prefix.isEmpty {
                    duration += leading
                } else {
                    parts.append(contentsOf: try durationTokens(prefix: "", note: "z", duration: leading, tieOut: false))
                }
                leading = 0
            }
            if value == 0 && next == measure.endT && trailing > 0 {
                duration += trailing
                trailing = 0
            }
            guard duration > 0 else { throw AbcNotationError("Non-positive ABC duration at subbeats \(t):\(next)") }
            let tieOut = value > 0 && next < voice.count && continuesPitch(value, voice[next])
            parts.append(contentsOf: try durationTokens(prefix: prefix, note: noteText, duration: duration, tieOut: tieOut))
            t = next
        }
        guard leading == 0 else { throw AbcNotationError("Measure \(measure.index): leading rest padding was not serialized") }
        if trailing > 0 { parts.append(contentsOf: try durationTokens(prefix: "", note: "z", duration: trailing, tieOut: false)) }
        return parts.joined()
    }

    /// Whether a rendered measure contains only untied rests (`_is_compressible_full_rest`).
    static func isCompressibleFullRest(_ rendered: String) -> Bool {
        guard !rendered.isEmpty else { return false }
        var index = rendered.startIndex
        while index < rendered.endIndex {
            guard rendered[index] == "z" else { return false }
            index = rendered.index(after: index)
            while index < rendered.endIndex, rendered[index].isASCII, rendered[index].isNumber {
                index = rendered.index(after: index)
            }
            // A rest is never tied (`durationTokens` only ties notes), so no "-" can follow.
        }
        return true
    }

    static func renderVoiceGroup(_ score: Score, voice: String, measures: [Measure], unit: Int) throws -> String {
        let rendered = try measures.map { try renderVoiceMeasure(score, voice: voice, measure: $0, unit: unit) }
        var parts = [String]()
        var index = 0
        while index < rendered.count {
            if !isCompressibleFullRest(rendered[index]) {
                parts.append(rendered[index] + "|")
                index += 1
                continue
            }
            var end = index + 1
            while end < rendered.count && isCompressibleFullRest(rendered[end]) { end += 1 }
            let count = end - index
            parts.append("Z" + (count > 1 ? String(count) : "") + "|")
            index = end
        }
        return parts.joined()
    }

    static func measureGroups(_ score: Score) -> [MeasureGroup] {
        let first = score.measures[0]
        var activeMeter = (first.abcNumerator, first.abcDenominator)
        var activeKey = score.keyArr[first.startT]
        var activeStructure = ""
        var groups = [MeasureGroup]()
        for measure in score.measures {
            let meter = (measure.abcNumerator, measure.abcDenominator)
            let key = score.keyArr[measure.startT]
            let meterChanged = meter != activeMeter
            let keyChanged = key != activeKey
            var labels = [String]()
            for (t, label) in score.structureEvents where measure.startT <= t && t < measure.endT {
                let clean = label.split(whereSeparator: \.isWhitespace).joined(separator: " ")
                if !clean.isEmpty && clean != activeStructure {
                    labels.append(clean)
                    activeStructure = clean
                }
            }
            if groups.isEmpty || groups[groups.count - 1].measures.count >= 4 || meterChanged || keyChanged || !labels.isEmpty {
                groups.append(MeasureGroup(measures: [measure], structureLabels: labels, meterChanged: meterChanged, keyChanged: keyChanged))
            } else {
                groups[groups.count - 1].measures.append(measure)
            }
            activeMeter = meter
            activeKey = score.keyArr[measure.endT - 1]
        }
        return groups
    }

    static func scoreToABC(_ score: Score) throws -> String {
        let unit = try unitDenominator(score)
        let first = score.measures[0]
        let tempo = Int(try estimateTempo(score).rounded(.toNearestOrEven))
        var lines = [
            "X:1", "T:", "M:\(first.abcNumerator)/\(first.abcDenominator)", "L:1/\(unit)", "Q:1/4=\(tempo)",
            "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"", "V: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"",
            "K:\(score.keyArr[first.startT])",
        ]
        for group in measureGroups(score) {
            lines.append(contentsOf: group.structureLabels.map { "% \($0)" })
            let groupFirst = group.measures[0]
            for voice in voiceIDs {
                lines.append("V: \(voice)")
                if group.meterChanged { lines.append("M:\(groupFirst.abcNumerator)/\(groupFirst.abcDenominator)") }
                if group.keyChanged { lines.append("K:\(score.keyArr[groupFirst.startT])") }
                lines.append(try renderVoiceGroup(score, voice: voice, measures: group.measures, unit: unit))
            }
        }
        return lines.joined(separator: "\n") + "\n"
    }
}
