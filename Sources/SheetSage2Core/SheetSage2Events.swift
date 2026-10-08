// SheetSage2Events.swift - token sequence → timed events (decode_sequence, event_time_map, stitching)
// Copyright 2026 Vincent Gourbin

import Foundation

/// One melody note of an event (`_decode_field("melody")` upstream); `endTime` is set by stitching.
public struct SheetSage2Note: Equatable, Sendable {
    public var pitch: Int
    public var track: Int
    public var durationBin: Int
    public var durationSteps: Int
    public var endTime: Double?
}

/// Decoded field values of one event.
public struct SheetSage2EventValues: Equatable, Sendable {
    public var timestamp: Double?
    public var meter: (Int, Int)?
    public var eighthPosition: Int?
    public var hasRhythm = false
    public var structure: String?
    public var key: String?
    public var chord: String?
    public var melody: [SheetSage2Note]?

    public static func == (a: Self, b: Self) -> Bool {
        a.timestamp == b.timestamp && a.meter?.0 == b.meter?.0 && a.meter?.1 == b.meter?.1
            && a.eighthPosition == b.eighthPosition && a.hasRhythm == b.hasRhythm
            && a.structure == b.structure && a.key == b.key && a.chord == b.chord && a.melody == b.melody
    }
}

/// One event: its subbeat step, raw tokens per field, values; `time` once stitched.
public struct SheetSage2Event: Equatable, Sendable {
    public var subbeat: Int
    public var tokensByField: [String: [Int]]
    public var values: SheetSage2EventValues
    public var time: Double?
    public var windowIndex = 0
    public var windowStart = 0.0
    public var sourceSubbeat = 0
    public var globalSubbeat = 0
}

extension SheetSage2Tokenizer {
    func activeFields(_ prompts: [String]) -> Set<String> {
        Set(Self.tasks.filter { prompts.contains($0.name) }.map(\.field))
    }

    /// `_decode_field` for every non-empty field of an event.
    func decodeValues(_ tokensByField: [String: [Int]]) -> SheetSage2EventValues {
        var values = SheetSage2EventValues()
        for (field, tokens) in tokensByField where !tokens.isEmpty {
            switch field {
            case "timestamp":
                values.timestamp = Double(tokens[0] - timeTokenStart) / Double(timeHz)
            case "rhythm":
                values.hasRhythm = true
                for token in tokens {
                    if tokenType(token) == .meter { values.meter = Self.meterPairs[token - meterTokenStart] }
                    else if tokenType(token) == .eighthPosition { values.eighthPosition = token - eighthPositionTokenStart }
                }
            case "structure":
                values.structure = Self.structureLabels[tokens[0] - structureTokenStart]
            case "key":
                let id = tokens[0] - keyTokenStart
                values.key = ChordSpelling.normalizeKeyName("\(Self.chromaticSharps[id % 12]):\(id >= 12 ? "minor" : "major")")
            case "chord":
                values.chord = tokenType(tokens[0]) == .chordMajmin
                    ? Self.majminChordLabels[tokens[0] - majminChordTokenStart]
                    : Self.fullChordLabels[tokens[0] - fullChordTokenStart]
            case "melody":
                var notes = [SheetSage2Note]()
                var index = 0
                while index < tokens.count {
                    let pitchID = tokens[index] - pitchTokenStart
                    var bin = 0
                    if index + 1 < tokens.count, tokenType(tokens[index + 1]) == .duration {
                        bin = tokens[index + 1] - durationTokenStart
                        index += 2
                    } else {
                        index += 1
                    }
                    notes.append(SheetSage2Note(
                        pitch: pitchID % 128, track: pitchID >= 128 ? 1 : 0, durationBin: bin,
                        durationSteps: Self.durationTemplates[bin]))
                }
                values.melody = notes
            default:
                break
            }
        }
        return values
    }

    /// `decode_sequence`: parse one prompt-conditioned sequence into typed events.
    public func decodeSequence(_ input: [Int], strict: Bool) throws -> (prompts: [String], events: [SheetSage2Event]) {
        var tokens = input
        while tokens.last == padToken { tokens.removeLast() }
        guard tokens.first == sosToken else { throw SheetSage2Error.invalidSequence("sequence must begin with <|sos|>") }
        guard let outIndex = tokens.dropFirst().firstIndex(of: outToken) else {
            throw SheetSage2Error.invalidSequence("sequence is missing <|out|>")
        }
        let prompts = tokens[1..<outIndex].map { Self.tasks[$0 - promptTokenStart].name }
        let active = activeFields(prompts)
        var events = [SheetSage2Event]()
        var position = outIndex + 1
        var step = 0
        var sawEOS = false
        while position < tokens.count {
            if tokens[position] == eosToken {
                sawEOS = true
                position += 1
                break
            }
            guard tokenType(tokens[position]) == .subbeatShift else {
                throw SheetSage2Error.invalidSequence("event at token index \(position) has no subbeat shift")
            }
            while position < tokens.count, tokenType(tokens[position]) == .subbeatShift {
                step += tokens[position] - subbeatShiftTokenStart
                position += 1
            }
            var byField = [String: [Int]]()
            while position < tokens.count {
                let token = tokens[position]
                guard let type = tokenType(token), type != .subbeatShift, token != eosToken else { break }
                guard let field = outputField(type) else {
                    throw SheetSage2Error.invalidSequence("token \(token) (\(type.rawValue)) has no field in schema v1")
                }
                if strict && !active.contains(field) {
                    throw SheetSage2Error.invalidSequence("token \(token) belongs to inactive output field '\(field)'")
                }
                byField[field, default: []].append(token)
                position += 1
            }
            if byField.isEmpty {
                if strict { throw SheetSage2Error.invalidSequence("empty event at subbeat \(step)") }
                continue
            }
            if strict { try validate(byField) }
            events.append(SheetSage2Event(subbeat: step, tokensByField: byField, values: decodeValues(byField)))
        }
        if strict && !sawEOS { throw SheetSage2Error.invalidSequence("sequence is missing <|eos|>") }
        if strict && position != tokens.count { throw SheetSage2Error.invalidSequence("non-padding tokens follow <|eos|>") }
        return (prompts, events)
    }

    private func validate(_ byField: [String: [Int]]) throws {
        for (field, values) in byField {
            let types = values.map { tokenType($0)! }
            switch field {
            case "timestamp" where types != [.time]:
                throw SheetSage2Error.invalidSequence("timestamp event must contain exactly one time token")
            case "rhythm" where types != [.eighthPosition] && types != [.meter, .eighthPosition]:
                throw SheetSage2Error.invalidSequence("invalid rhythm payload")
            case "structure", "key", "chord":
                if values.count != 1 { throw SheetSage2Error.invalidSequence("field '\(field)' must contain exactly one token") }
            case "melody":
                var index = 0
                while index < types.count {
                    guard types[index] == .pitch else {
                        throw SheetSage2Error.invalidSequence("melody payload must contain pitch tokens with optional duration")
                    }
                    index += index + 1 < types.count && types[index + 1] == .duration ? 2 : 1
                }
            default:
                break
            }
        }
    }

    /// `decode_generated_tokens`: strict first, lenient on the two recoverable errors.
    public func decodeGenerated(_ tokens: [Int]) throws -> (prompts: [String], events: [SheetSage2Event], warning: String?) {
        do {
            let decoded = try decodeSequence(tokens, strict: true)
            return (decoded.prompts, decoded.events, nil)
        } catch let SheetSage2Error.invalidSequence(message)
            where message.contains("empty event at subbeat") || message.contains("belongs to inactive output field")
        {
            let decoded = try decodeSequence(tokens, strict: false)
            return (decoded.prompts, decoded.events, message)
        }
    }
}

/// `event_time_map`: subbeat → seconds, interpolating between timestamped events.
struct EventTimeMap {
    let steps: [Double]
    let times: [Double]
    let stepSeconds: Double
    let target: Double

    init(events: [SheetSage2Event], targetSeconds: Double) {
        target = targetSeconds
        var anchors = [Int: Double]()
        for event in events {
            if let value = event.values.timestamp { anchors[event.subbeat] = value }
        }
        let sorted = anchors.sorted { $0.key < $1.key }
        let steps = sorted.map { Double($0.key) }
        let times = sorted.map(\.value)
        self.steps = steps
        self.times = times
        if steps.count >= 2 {
            let ratios = (1..<steps.count).map { (times[$0] - times[$0 - 1]) / max(steps[$0] - steps[$0 - 1], 1) }
            let median = Self.median(ratios)
            stepSeconds = median.isFinite && median > 0 ? median : 0.125
        } else {
            stepSeconds = 0.125
        }
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let n = sorted.count
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }

    func callAsFunction(_ stepInt: Int) -> Double {
        let step = Double(stepInt)
        guard let first = steps.first, let last = steps.last else {
            return min(target, max(0, step * 0.125))
        }
        if step <= first { return min(max(times[0] + (step - first) * stepSeconds, 0), target) }
        if step >= last { return min(max(times[times.count - 1] + (step - last) * stepSeconds, 0), target) }
        // np.interp: right-closed segment search.
        var index = 1
        while index < steps.count && steps[index] < step { index += 1 }
        if steps[index] == step { return times[index] }
        // Same operation order as numpy's compiled interp, so the result is bit-identical.
        let (x0, x1, y0, y1) = (steps[index - 1], steps[index], times[index - 1], times[index])
        let slope = (y1 - y0) / (x1 - x0)
        return slope * (step - x0) + y0
    }
}

/// `stitched_window_events`: keep a window's events inside its acceptance range, in song time.
func stitchedWindowEvents(
    _ events: [SheetSage2Event], timeMap: EventTimeMap, windowStart: Double, acceptStart: Double,
    acceptEnd: Double, songDuration: Double, windowIndex: Int, globalSubbeatBase: Int
) -> [SheetSage2Event] {
    let eps = 1e-4
    var accepted = [SheetSage2Event]()
    for event in events {
        let absolute = windowStart + timeMap(event.subbeat)
        if absolute < acceptStart - eps { continue }
        if absolute >= acceptEnd - eps || absolute >= songDuration - eps { continue }
        var output = event
        let time = min(max(absolute, 0), songDuration)
        output.time = time
        output.windowIndex = windowIndex
        output.windowStart = windowStart
        output.sourceSubbeat = event.subbeat
        output.globalSubbeat = globalSubbeatBase + event.subbeat
        if output.values.timestamp != nil { output.values.timestamp = time }
        if let notes = output.values.melody {
            output.values.melody = notes.map { note in
                var fixed = note
                let end = windowStart + timeMap(event.subbeat + note.durationSteps)
                fixed.endTime = min(songDuration, max(time + 0.04, end))
                return fixed
            }
        }
        accepted.append(output)
    }
    return accepted
}

extension Array where Element == SheetSage2Event {
    /// The transcription's beat grid: global 16th-note index → seconds in the audio, the clock a
    /// lyric or score timeline is placed on (`LyricTimeline.build(…beatGrid:onsets:)`).
    public var beatGrid: [(subbeat: Int, time: Double)] {
        compactMap { e in e.time.map { (e.globalSubbeat, $0) } }
    }

    /// Every melody onset heard (any track), as global 16th and MIDI pitch.
    public var melodyOnsets: [(subbeat: Int, midi: Int)] {
        flatMap { e in (e.values.melody ?? []).map { (e.globalSubbeat, $0.pitch) } }
    }
}
