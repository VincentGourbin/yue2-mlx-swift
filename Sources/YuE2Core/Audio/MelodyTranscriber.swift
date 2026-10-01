// MelodyTranscriber.swift - hummed / sung monophonic audio -> ABC vocal line the model can take as its score
// Copyright 2026 Vincent Gourbin
//
// The model accepts an imposed ABC score (`SongRequest.abc`): this turns a short monophonic
// recording into one, in the exact dialect the model writes itself (`X:1 … L:1/16 … V: Vocal /
// V: Ins` blocks of four bars, chord symbols, `Z4|` instrumental rests, `% verse` sections) so
// that the semantic and acoustic stages see a score shaped like their training data. Pure CPU
// (Accelerate): a YIN pitch tracker, median smoothing, note segmentation, quantization to a
// sixteenth-note grid at a caller-supplied tempo, key estimation (Krumhansl profiles) and a
// one-chord-per-bar diatonic harmonization.

import Accelerate
import Foundation
import MLX

public struct MelodyNote: Equatable, Sendable {
    /// MIDI pitch (60 = C4).
    public let midi: Int
    /// Start and length on the sixteenth-note grid (16 per 4/4 bar).
    public let startSixteenth: Int
    public let lengthSixteenths: Int
}

public struct MelodyTranscription: Sendable {
    public let bpm: Double
    /// ABC key field (`C`, `Am`, `Bb`, `F#m`…).
    public let key: String
    public let notes: [MelodyNote]
    /// Fraction of analysed frames that carried a pitch — low values mean the recording was
    /// mostly silence or noise, and the score is not to be trusted.
    public let voicedRatio: Double
    /// The score, ready for `SongRequest.abc`.
    public let abc: String
}

public enum MelodyTranscriber {
    public struct Options: Sendable {
        public var bpm: Double = 100
        /// Section label written before the bars (`% verse` by default).
        public var section: String = "verse"
        /// Minimum note length, in milliseconds, before segmentation keeps it.
        public var minimumNoteMs: Double = 80
        /// Pitch range searched, in Hz (a sung or hummed melody).
        public var minHz: Double = 70
        public var maxHz: Double = 1000
        public init() {}
    }

    /// `audio`: `[1, S, C]` (what `AudioImporter.loadAudio` returns) or `[S]`, any sample rate.
    public static func transcribe(audio: MLXArray, sampleRate: Double, options: Options = Options()) -> MelodyTranscription {
        let mono: MLXArray = audio.ndim == 3 ? audio[0].mean(axis: -1) : audio
        return transcribe(samples: mono.asType(.float32).asArray(Float.self), sampleRate: sampleRate, options: options)
    }

    public static func transcribe(samples: [Float], sampleRate: Double, options: Options = Options()) -> MelodyTranscription {
        let f0 = pitchTrack(samples: samples, sampleRate: sampleRate, minHz: options.minHz, maxHz: options.maxHz)
        let hopSeconds = Double(hop(for: sampleRate)) / sampleRate
        let voiced = f0.filter { $0 > 0 }.count
        let voicedRatio = f0.isEmpty ? 0 : Double(voiced) / Double(f0.count)
        let segments = segment(f0: f0, hopSeconds: hopSeconds, minimumSeconds: options.minimumNoteMs / 1000)
        let notes = quantize(segments: segments, bpm: options.bpm)
        let key = estimateKey(notes: notes)
        let abc = render(notes: notes, key: key, bpm: options.bpm, section: options.section)
        return MelodyTranscription(bpm: options.bpm, key: key.abcName, notes: notes, voicedRatio: voicedRatio, abc: abc)
    }

    // MARK: - Pitch tracking (YIN, de Cheveigné & Kawahara 2002)

    static func hop(for sampleRate: Double) -> Int { max(1, Int(sampleRate / 100)) }  // 10 ms

    /// One f0 (Hz) per hop, 0 for unvoiced frames.
    static func pitchTrack(samples: [Float], sampleRate: Double, minHz: Double, maxHz: Double) -> [Float] {
        let window = max(1024, Int(sampleRate * 0.04))  // 40 ms
        let hop = hop(for: sampleRate)
        let tauMin = max(2, Int(sampleRate / maxHz))
        let tauMax = min(window / 2, Int(sampleRate / minHz))
        guard samples.count >= window + tauMax, tauMax > tauMin else { return [] }
        let threshold: Float = 0.15
        var track: [Float] = []
        var frame = [Float](repeating: 0, count: window + tauMax)
        var squares = [Float](repeating: 0, count: window + tauMax)
        var cumulative = [Float](repeating: 0, count: window + tauMax + 1)
        var cmndf = [Float](repeating: 0, count: tauMax + 1)
        var start = 0
        while start + window + tauMax <= samples.count {
            for i in 0..<(window + tauMax) { frame[i] = samples[start + i] }
            var rms: Float = 0
            vDSP_rmsqv(frame, 1, &rms, vDSP_Length(window))
            if rms < 0.01 {
                track.append(0); start += hop; continue
            }
            // Difference function d(τ) = Σ (x[i] − x[i+τ])² = E(0) + E(τ) − 2·x·x_τ, the energies
            // from one prefix sum of squares; then cumulative-mean normalisation (the full curve
            // first, the dip search after — stopping at the threshold crossing reads a period that
            // is too short, a semitone sharp on a sung tone).
            vDSP_vsq(frame, 1, &squares, 1, vDSP_Length(window + tauMax))
            cumulative[0] = 0
            for i in 0..<(window + tauMax) { cumulative[i + 1] = cumulative[i] + squares[i] }
            let e1 = cumulative[window]
            var running: Float = 0
            cmndf[0] = 1
            for tau in 1...tauMax {
                var dot: Float = 0
                frame.withUnsafeBufferPointer { p in
                    vDSP_dotpr(p.baseAddress!, 1, p.baseAddress! + tau, 1, &dot, vDSP_Length(window))
                }
                let e2 = cumulative[tau + window] - cumulative[tau]
                let d = max(0, e1 + e2 - 2 * dot)
                running += d
                cmndf[tau] = running > 0 ? d * Float(tau) / running : 1
            }
            // First dip under the threshold, followed to its local minimum; otherwise the global
            // minimum of the search range if it is convincing enough.
            var best = 0
            var bestValue: Float = .greatestFiniteMagnitude
            var tau = tauMin
            while tau <= tauMax {
                if cmndf[tau] < threshold {
                    while tau + 1 <= tauMax, cmndf[tau + 1] < cmndf[tau] { tau += 1 }
                    best = tau; bestValue = cmndf[tau]
                    break
                }
                if cmndf[tau] < bestValue { bestValue = cmndf[tau]; best = tau }
                tau += 1
            }
            if bestValue < 0.5, best > 0 {
                // parabolic interpolation around the minimum
                var refined = Float(best)
                if best > 1, best < tauMax {
                    let a = cmndf[best - 1], b = cmndf[best], c = cmndf[best + 1]
                    let denom = a - 2 * b + c
                    if abs(denom) > 1e-9 { refined += 0.5 * (a - c) / denom }
                }
                track.append(Float(sampleRate) / refined)
            } else {
                track.append(0)
            }
            start += hop
        }
        return medianSmooth(track, radius: 2)
    }

    static func medianSmooth(_ values: [Float], radius: Int) -> [Float] {
        guard values.count > 2 * radius else { return values }
        var out = values
        for i in 0..<values.count {
            let lo = max(0, i - radius), hi = min(values.count - 1, i + radius)
            let voiced = values[lo...hi].filter { $0 > 0 }
            if values[i] == 0 {
                // a one-frame dropout inside a note is filled, a real gap is kept
                out[i] = voiced.count >= 2 * radius ? voiced.sorted()[voiced.count / 2] : 0
            } else {
                out[i] = voiced.count > radius ? voiced.sorted()[voiced.count / 2] : values[i]
            }
        }
        return out
    }

    // MARK: - Segmentation

    struct Segment { let midi: Int; let start: Double; let end: Double }

    static func midi(_ hz: Float) -> Double { 69 + 12 * log2(Double(hz) / 440) }

    static func segment(f0: [Float], hopSeconds: Double, minimumSeconds: Double) -> [Segment] {
        var segments: [Segment] = []
        var current: (pitches: [Double], start: Int)?
        func close(at index: Int) {
            guard let c = current else { return }
            let duration = Double(index - c.start) * hopSeconds
            if duration >= minimumSeconds {
                let median = c.pitches.sorted()[c.pitches.count / 2]
                segments.append(Segment(midi: Int(median.rounded()), start: Double(c.start) * hopSeconds, end: Double(index) * hopSeconds))
            }
            current = nil
        }
        var unvoicedRun = 0
        for (i, hz) in f0.enumerated() {
            guard hz > 0 else {
                unvoicedRun += 1
                if Double(unvoicedRun) * hopSeconds >= 0.03 { close(at: i - unvoicedRun + 1) }
                continue
            }
            unvoicedRun = 0
            let m = midi(hz)
            if var c = current {
                let reference = c.pitches.sorted()[c.pitches.count / 2]
                if abs(m - reference) < 0.6 {
                    c.pitches.append(m); current = c
                } else {
                    close(at: i); current = ([m], i)
                }
            } else {
                current = ([m], i)
            }
        }
        close(at: f0.count)
        // Two pieces of one held note split by a wobble (same pitch, no silence between) are one note.
        var merged: [Segment] = []
        for s in segments {
            if let last = merged.last, last.midi == s.midi, s.start - last.end < 0.02 {
                merged[merged.count - 1] = Segment(midi: last.midi, start: last.start, end: s.end)
            } else {
                merged.append(s)
            }
        }
        return merged
    }

    // MARK: - Quantization (sixteenth grid)

    static func quantize(segments: [Segment], bpm: Double) -> [MelodyNote] {
        let sixteenth = 60.0 / bpm / 4
        guard let first = segments.first else { return [] }
        let origin = first.start
        // The tracker loses the attack and the decay of every note (≈ 30 ms on a sung tone).
        let edgeCompensation = 0.03
        var notes: [MelodyNote] = []
        var cursor = 0
        for (i, s) in segments.enumerated() {
            var start = Int(((s.start - origin) / sixteenth).rounded())
            if start < cursor { start = cursor }
            var length = max(1, Int(((s.end - s.start + edgeCompensation) / sixteenth).rounded()))
            if i + 1 < segments.count {
                let next = segments[i + 1]
                let nextStart = max(start + 1, Int(((next.start - origin) / sixteenth).rounded()))
                // a breath or a consonant shorter than a sixteenth: the note is held to the next one
                if next.start - s.end < 0.6 * sixteenth || start + length > nextStart { length = nextStart - start }
            }
            notes.append(MelodyNote(midi: s.midi, startSixteenth: start, lengthSixteenths: length))
            cursor = start + length
        }
        return notes
    }

    // MARK: - Key estimation and harmonization

    struct Key { let tonic: Int; let minor: Bool
        var abcName: String { (Self.names(flat: usesFlats)[tonic]) + (minor ? "m" : "") }
        /// F, Bb, Eb, Ab, Db majors (and their relative minors) are written with flats; F# / D#m with sharps.
        var usesFlats: Bool { [5, 10, 3, 8, 1].contains(minor ? (tonic + 3) % 12 : tonic) }
        static func names(flat: Bool) -> [String] {
            flat ? ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
                 : ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        }
        /// Pitch classes altered by the key signature (sharps or flats) — notes on those
        /// letters are written bare, naturals on them need `=`.
        var signature: Set<Int> {
            let major = minor ? (tonic + 3) % 12 : tonic
            let sharps = [6, 1, 8, 3, 10, 5, 0]   // F# C# G# D# A# E# B#
            let flats = [10, 3, 8, 1, 6, 11, 4]   // Bb Eb Ab Db Gb Cb Fb
            let counts: [Int: Int] = [0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 5: -1, 10: -2, 3: -3, 8: -4, 1: -5]
            let n = counts[major] ?? 0
            return n >= 0 ? Set(sharps.prefix(n)) : Set(flats.prefix(-n))
        }
    }

    static func estimateKey(notes: [MelodyNote]) -> Key {
        var histogram = [Double](repeating: 0, count: 12)
        for n in notes { histogram[((n.midi % 12) + 12) % 12] += Double(n.lengthSixteenths) }
        for n in [notes.first, notes.last].compactMap({ $0 }) { histogram[((n.midi % 12) + 12) % 12] += 4 }
        let majorProfile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        let minorProfile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        var best = Key(tonic: 0, minor: false), bestScore = -Double.infinity
        for tonic in 0..<12 {
            for (minor, profile) in [(false, majorProfile), (true, minorProfile)] {
                var score = 0.0
                for pc in 0..<12 { score += profile[(pc - tonic + 12) % 12] * histogram[pc] }
                if score > bestScore { bestScore = score; best = Key(tonic: tonic, minor: minor) }
            }
        }
        return best
    }

    /// One diatonic triad per bar: the one covering the most note duration in the bar.
    static func chord(forBar notes: [MelodyNote], key: Key) -> String {
        let major = key.minor ? (key.tonic + 3) % 12 : key.tonic
        // (root offset from the major tonic, minor?) for I ii iii IV V vi
        let triads: [(Int, Bool)] = [(0, false), (2, true), (4, true), (5, false), (7, false), (9, true)]
        var best = triads[key.minor ? 5 : 0], bestScore = -1.0
        for (offset, isMinor) in triads {
            let root = (major + offset) % 12
            let pcs: Set<Int> = [root, (root + (isMinor ? 3 : 4)) % 12, (root + 7) % 12]
            var score = 0.0
            for n in notes where pcs.contains(((n.midi % 12) + 12) % 12) { score += Double(n.lengthSixteenths) }
            if score > bestScore { bestScore = score; best = (offset, isMinor) }
        }
        let root = (major + best.0) % 12
        return Key.names(flat: key.usesFlats)[root] + (best.1 ? "m" : "")
    }

    // MARK: - ABC rendering

    static func noteName(midi: Int, key: Key) -> String {
        let pc = ((midi % 12) + 12) % 12
        let octave = midi / 12 - 1          // C4 = 60 -> 4
        let letters = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
        let naturals: Set<Int> = [0, 2, 4, 5, 7, 9, 11]
        var letter = letters[pc]
        var accidental = ""
        if naturals.contains(pc) {
            // a natural on a letter the signature alters needs an explicit natural
            let sharpened = (pc + 1) % 12, flattened = (pc + 11) % 12
            if !key.usesFlats && key.signature.contains(sharpened) { accidental = "=" }
            if key.usesFlats && key.signature.contains(flattened) { accidental = "=" }
        } else if key.signature.contains(pc) {
            // written bare: the key signature supplies the accidental
            letter = key.usesFlats ? letters[(pc + 1) % 12] : letters[(pc + 11) % 12]
        } else {
            letter = key.usesFlats ? letters[(pc + 1) % 12] : letters[(pc + 11) % 12]
            accidental = key.usesFlats ? "_" : "^"
        }
        var name = accidental + letter
        if octave >= 5 { name = accidental + letter.lowercased() + String(repeating: "'", count: octave - 5) }
        else if octave < 4 { name += String(repeating: ",", count: 4 - octave) }
        return name
    }

    static func render(notes: [MelodyNote], key: Key, bpm: Double, section: String) -> String {
        let barLength = 16
        let totalSixteenths = max(barLength, notes.map { $0.startSixteenth + $0.lengthSixteenths }.max() ?? 0)
        let bars = (totalSixteenths + barLength - 1) / barLength
        var barTexts: [String] = []
        for bar in 0..<bars {
            let barStart = bar * barLength, barEnd = barStart + barLength
            let inBar = notes.filter { $0.startSixteenth < barEnd && $0.startSixteenth + $0.lengthSixteenths > barStart }
            var text = "\"" + chord(forBar: inBar, key: key) + "\""
            var cursor = barStart
            for n in inBar {
                let s = max(n.startSixteenth, barStart), e = min(n.startSixteenth + n.lengthSixteenths, barEnd)
                if s > cursor { text += "z\(s - cursor)" }
                text += noteName(midi: n.midi, key: key) + "\(e - s)"
                if n.startSixteenth + n.lengthSixteenths > barEnd { text += "-" }  // tied into the next bar
                cursor = e
            }
            if cursor < barEnd { text += "z\(barEnd - cursor)" }
            barTexts.append(text + "|")
        }
        var abc = "X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=\(Int(bpm.rounded()))\n"
        abc += "V: Vocal clef=treble name=\"Vocal Melody\" snm=\"Vocal\"\nV: Ins clef=treble name=\"Ins Melody\" snm=\"Inst.\"\n"
        abc += "K:\(key.abcName)\n% \(section)\n"
        var i = 0
        while i < barTexts.count {
            let group = barTexts[i..<min(i + 4, barTexts.count)]
            abc += "V: Vocal\n" + group.joined() + "\nV: Ins\nZ\(group.count)|\n"
            i += 4
        }
        return abc
    }
}
