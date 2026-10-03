// ChordSpelling.swift - key names and local-key chord spelling (chord_spelling_sheetsage2.py port)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Python-style integer helpers: `//` and `%` floor toward −∞ (Swift truncates toward zero).
@inline(__always) func floorDiv(_ a: Int, _ b: Int) -> Int {
    let q = a / b
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
}

@inline(__always) func floorMod(_ a: Int, _ b: Int) -> Int {
    let r = a % b
    return r != 0 && (r < 0) != (b < 0) ? r + b : r
}

/// Local-key chord spelling, ported from SheetSage2 LM (`chord_spelling_sheetsage2.py`).
enum ChordSpelling {
    static let keyMap: [[String]] = [
        ["C:major", "Db:major", "D:major", "Eb:major", "E:major", "F:major", "F#:major", "G:major", "Ab:major", "A:major", "Bb:major", "B:major"],
        ["D:dorian", "Eb:dorian", "E:dorian", "F:dorian", "F#:dorian", "G:dorian", "G#:dorian", "A:dorian", "Bb:dorian", "B:dorian", "C:dorian", "C#:dorian"],
        ["E:phrygian", "F:phrygian", "F#:phrygian", "G:phrygian", "G#:phrygian", "A:phrygian", "A#:phrygian", "B:phrygian", "C:phrygian", "C#:phrygian", "D:phrygian", "D#:phrygian"],
        ["F:lydian", "Gb:lydian", "G:lydian", "Ab:lydian", "A:lydian", "Bb:lydian", "B:lydian", "C:lydian", "Db:lydian", "D:lydian", "Eb:lydian", "E:lydian"],
        ["G:mixolydian", "Ab:mixolydian", "A:mixolydian", "Bb:mixolydian", "B:mixolydian", "C:mixolydian", "C#:mixolydian", "D:mixolydian", "Eb:mixolydian", "E:mixolydian", "F:mixolydian", "F#:mixolydian"],
        ["A:minor", "Bb:minor", "B:minor", "C:minor", "C#:minor", "D:minor", "D#:minor", "E:minor", "F:minor", "F#:minor", "G:minor", "G#:minor"],
        ["B:locrian", "C:locrian", "C#:locrian", "D:locrian", "D#:locrian", "E:locrian", "E#:locrian", "F#:locrian", "G:locrian", "G#:locrian", "A:locrian", "A#:locrian"],
    ]
    static let modeNames = ["major", "dorian", "phrygian", "lydian", "mixolydian", "minor", "locrian"]
    static let modeStarts = [0, 2, 4, 5, 7, 9, 11]
    static let noteNames = Array("CDEFGAB")
    static let circleOfFifthInv = [1, 3, 5, 0, 2, 4, 6]
    static let majorScaleFifths = [0, 2, 4, -1, 1, 3, 5]
    static let defaultSpelling = ["1", "b2", "2", "b3", "3", "4", "b5", "5", "#5", "6", "b7", "7"]

    static let qualities: [String: [Int]] = [
        "maj": [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], "min": [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "aug": [2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], "dim": [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        "sus4": [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], "sus4(b7)": [2, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        "sus4(b7,9)": [2, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0], "sus2": [2, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "7": [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], "maj7": [2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "min7": [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], "minmaj7": [2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        "maj6": [2, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0], "min6": [2, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
        "9": [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0], "maj9": [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "min9": [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], "7(b9)": [2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "7(#9)": [2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0], "maj6(9)": [2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "min6(9)": [2, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], "maj(9)": [2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "min(9)": [2, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0], "maj(11)": [2, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        "min(11)": [2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1], "11": [2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        "maj9(11)": [2, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1], "min11": [2, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        "13": [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], "maj13": [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "min13": [2, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], "dim7": [2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        "hdim7": [2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    ]

    /// `mir_eval.chord.pitch_class_to_semitone`: letter plus `#`/`b` offsets, modulo 12.
    static func pitchClassToSemitone(_ name: String) -> Int {
        let base = ["C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11][String(name.prefix(1))] ?? 0
        let offset = name.dropFirst().reduce(0) { $0 + ($1 == "#" ? 1 : $1 == "b" ? -1 : 0) }
        return floorMod(base + offset, 12)
    }

    /// `normalize_key_name`: canonical LM/NoLM tonal spelling.
    static func normalizeKeyName(_ keyName: String) -> String {
        let parts = keyName.split(separator: ":", maxSplits: 1).map(String.init)
        let modeID = modeNames.firstIndex(of: parts[1])!
        let scale = floorMod(pitchClassToSemitone(parts[0]) - modeStarts[modeID], 12)
        return keyMap[modeID][scale]
    }

    static func scaleDegreeTuple(_ degree: String) -> (Int, Int) {
        var text = degree
        var offset = 0
        if text.hasPrefix("#") {
            offset = text.filter { $0 == "#" }.count
            text = text.replacingOccurrences(of: "#", with: "")
        } else if text.hasPrefix("b") {
            offset = -text.filter { $0 == "b" }.count
            text = text.replacingOccurrences(of: "b", with: "")
        }
        return (Int(text)! - 1, offset)
    }

    static func noteNameTuple(_ name: String) -> (Int, Int) {
        let degree = noteNames.firstIndex(of: name.first!)!
        var offset = 0
        if name.hasSuffix("#") { offset = name.filter { $0 == "#" }.count }
        else if name.hasSuffix("b") { offset = -name.dropFirst().filter { $0 == "b" }.count }
        return (degree, offset)
    }

    static let spellingTable: [String: [(Int, Int)]] = {
        var table = [String: [(Int, Int)]]()
        for (quality, chroma) in qualities {
            var spelling = [String]()
            for i in 0..<12 where chroma[i] > 0 {
                if quality.contains("#9") && i == defaultSpelling.firstIndex(of: "b3")! { spelling.append("#2") }
                else if quality.contains("dim7") && i == defaultSpelling.firstIndex(of: "6")! { spelling.append("bb7") }
                else { spelling.append(defaultSpelling[i]) }
            }
            table[quality] = spelling.map(scaleDegreeTuple)
        }
        return table
    }()

    static func spellChordTones(root: (Int, Int), quality: String) -> [(Int, Int)] {
        let intervals = spellingTable[quality]!
        let rootPos = circleOfFifthInv[root.0] + root.1 * 7
        return intervals.map { interval in
            let letter = floorMod(root.0 + interval.0, 7)
            let tonePos = rootPos + majorScaleFifths[interval.0] + interval.1 * 7
            return (letter, floorDiv(tonePos - circleOfFifthInv[letter], 7))
        }
    }

    static func score(_ spelling: [(Int, Int)], key: (Int, Int)) -> Int {
        let keyPos = circleOfFifthInv[key.0] + key.1 * 7
        let relative = spelling.map { tone in
            max(abs(circleOfFifthInv[floorMod(tone.0, 7)] + tone.1 * 7 - (keyPos + 2)) - 3, 0)
        }
        return relative.reduce(0, +) + relative[0]
    }

    /// `correct_chord_spelling`: re-spell the root so the chord fits `keyName` best.
    static func correctChordSpelling(_ label: String, keyName: String) -> String {
        if label == "N" || label == "X" { return label }
        let keyParts = keyName.split(separator: ":", maxSplits: 1).map(String.init)
        let keyMode = modeNames.firstIndex(of: keyParts[1])!
        let scaleSemitone = floorMod(pitchClassToSemitone(keyParts[0]) - modeStarts[keyMode], 12)
        let keySpelling = noteNameTuple(String(keyMap[0][scaleSemitone].split(separator: ":")[0]))
        var body = label
        var inversion = ""
        if let slash = label.firstIndex(of: "/") {
            body = String(label[..<slash])
            inversion = String(label[label.index(after: slash)...])
        }
        let parts = body.split(separator: ":", maxSplits: 1).map(String.init)
        let (root, quality) = (parts[0], parts[1])
        let rootSemitone = pitchClassToSemitone(root)
        func rootSpelling(_ id: Int) -> (Int, Int) {
            (id, floorMod(rootSemitone - pitchClassToSemitone(String(noteNames[id])) + 6, 12) - 6)
        }
        var best = 0
        var bestScore = Int.max
        for id in 0..<7 {
            let s = score(spellChordTones(root: rootSpelling(id), quality: quality), key: keySpelling)
            if s < bestScore { (best, bestScore) = (id, s) }
        }
        let spelled = rootSpelling(best)
        let name = String(noteNames[best]) + (spelled.1 < 0 ? String(repeating: "b", count: -spelled.1) : String(repeating: "#", count: spelled.1))
        return inversion.isEmpty ? "\(name):\(quality)" : "\(name):\(quality)/\(inversion)"
    }

    /// `correct_chord_rows`: spell each chord in the key active at its midpoint (left-boundary ties).
    static func correctChordRows(_ chords: [IntervalRow], keys: [IntervalRow]) -> [IntervalRow] {
        guard !keys.isEmpty else { return chords }
        let boundaries = keys.dropLast().map(\.end)
        return chords.map { row in
            let midpoint = (row.start + row.end) / 2
            let index = searchSortedLeft(boundaries, midpoint)
            return IntervalRow(start: row.start, end: row.end, label: correctChordSpelling(row.label, keyName: keys[index].label))
        }
    }
}

/// `[start, end, label]` interval row (`interval_rows` upstream).
public struct IntervalRow: Equatable, Sendable {
    public var start: Double
    public var end: Double
    public var label: String
}

/// `np.searchsorted(values, x, side="left")`: first index whose value is ≥ x.
func searchSortedLeft(_ values: [Double], _ x: Double) -> Int {
    var (low, high) = (0, values.count)
    while low < high {
        let mid = (low + high) / 2
        if values[mid] < x { low = mid + 1 } else { high = mid }
    }
    return low
}
