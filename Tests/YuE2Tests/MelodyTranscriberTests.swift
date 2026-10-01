// MelodyTranscriberTests.swift - synthetic melodies -> notes and ABC (CPU only)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

@Suite("MelodyTranscriber")
struct MelodyTranscriberTests {
    /// A sung-like tone: fundamental plus two harmonics, slight vibrato, soft attack.
    private func tone(midi: Int, seconds: Double, sampleRate: Double) -> [Float] {
        let hz = 440 * pow(2, Double(midi - 69) / 12)
        let n = Int(seconds * sampleRate)
        return (0..<n).map { i in
            let t = Double(i) / sampleRate
            // 5.5 Hz vibrato of ±0.3 % (integrated into the phase, not multiplied: a multiplied
            // factor sweeps the frequency wider and wider with time)
            let phase = 2 * .pi * hz * (t + 0.003 / (2 * .pi * 5.5) * (1 - cos(2 * .pi * 5.5 * t)))
            let env = min(1, t / 0.02) * min(1, (seconds - t) / 0.02)
            let s = sin(phase) + 0.4 * sin(2 * phase) + 0.2 * sin(3 * phase)
            return Float(0.4 * env * s)
        }
    }

    @Test func transcribesAFourNoteMelodyAt120BPM() {
        let sr = 48_000.0
        // 120 BPM: a sixteenth is 125 ms, a quarter 500 ms. C4 E4 G4 (quarters) C5 (half), each note
        // sung 60 ms short of its value (the breath between notes).
        var samples: [Float] = []
        let gap = [Float](repeating: 0, count: Int(0.06 * sr))
        for (midi, seconds) in [(60, 0.44), (64, 0.44), (67, 0.44), (72, 0.94)] {
            samples += tone(midi: midi, seconds: seconds, sampleRate: sr) + gap
        }
        var options = MelodyTranscriber.Options()
        options.bpm = 120
        let t = MelodyTranscriber.transcribe(samples: samples, sampleRate: sr, options: options)
        #expect(t.notes.map(\.midi) == [60, 64, 67, 72], "\(t.notes)")
        #expect(t.notes.map(\.startSixteenth) == [0, 4, 8, 12], "\(t.notes)")
        #expect(t.notes.map(\.lengthSixteenths) == [4, 4, 4, 8], "\(t.notes)")
        #expect(t.voicedRatio > 0.8)
        #expect(t.key == "C")
        #expect(t.abc.contains("Q:1/4=120"))
        #expect(t.abc.contains("K:C"))
        // bar 1: C major triad under C E G c, bar 2: the held c
        #expect(t.abc.contains("V: Vocal\n\"C\"C4E4G4c4-|\"C\"c4z12|"), "\(t.abc)")
        #expect(t.abc.contains("V: Ins\nZ2|"))
    }

    @Test func silenceGivesNoNotesAndAnEmptyScore() {
        let t = MelodyTranscriber.transcribe(samples: [Float](repeating: 0, count: 48_000), sampleRate: 48_000)
        #expect(t.notes.isEmpty)
        #expect(t.voicedRatio == 0)
        #expect(t.abc.contains("Z1|"))
    }

    @Test func spellsAccidentalsFromTheKeySignature() {
        // F# major-ish melody: F#4 G#4 A#4 -> key F# (sharps), notes written bare under the signature
        let sr = 48_000.0
        var samples: [Float] = []
        for m in [66, 68, 70, 66] { samples += tone(midi: m, seconds: 0.4, sampleRate: sr) + [Float](repeating: 0, count: 2_400) }
        let t = MelodyTranscriber.transcribe(samples: samples, sampleRate: sr)
        #expect(t.notes.map(\.midi) == [66, 68, 70, 66], "\(t.notes)")
        #expect(t.key == "F#", "\(t.key)")
        #expect(!t.abc.contains("^"), "a melody inside its own key signature needs no explicit sharps: \(t.abc)")
    }
}
