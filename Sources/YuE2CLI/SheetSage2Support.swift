// SheetSage2Support.swift - shared SheetSage2 plumbing for transcribe, karaoke and the instrumental leak check
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import SheetSage2Core
import YuE2Core

enum SheetSage2Support {
    /// `--model`, else the profile's pack under `$YUE2_MODELS_DIR` (SheetSage2-q8 / -q4), else
    /// SheetSage2-fp16, else SheetSage2.
    static func directory(model: String?, profile: String, unquantized: Bool = false) throws -> URL {
        if let model { return URL(fileURLWithPath: model) }
        guard let root = ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"], !root.isEmpty else {
            throw ValidationError("pass --model or set YUE2_MODELS_DIR")
        }
        let base = URL(fileURLWithPath: root)
        let profileBits = SheetSage2Profile.named(profile)?.bits ?? 16
        let packName = ["SheetSage2-q8", "SheetSage2-q4"][safe: profileBits == 8 ? 0 : profileBits == 4 ? 1 : -1]
        let candidates = (unquantized ? [] : [packName].compactMap { $0 }) + ["SheetSage2-fp16", "SheetSage2"]
        return candidates.map { base.appendingPathComponent($0) }
            .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.safetensors").path) }
            ?? base.appendingPathComponent("SheetSage2")
    }

    /// Loads the profile's model and transcribes a file (melody only).
    static func transcribe(audio: URL, model: String?, profile: String) throws -> SheetSage2Transcription {
        guard let reference = SheetSage2Profile.named(profile) else {
            throw ValidationError("unknown SheetSage2 profile \(profile)")
        }
        let sheetsage = try SheetSage2Model.load(directory: directory(model: model, profile: profile), profile: reference)
        let stereo = try AudioImporter.loadAudio(url: audio, sampleRate: Double(sheetsage.config.samplingRate))
        return try SheetSage2Transcriber(model: sheetsage, profile: reference)
            .transcribe(stereo[0].mean(axis: -1), melodyOnly: true) { _, _ in }
    }

    /// Beat grid and heard onsets of a transcription, for `LyricTimeline`.
    static func grid(_ events: [SheetSage2Event]) -> ([LyricTimeline.GridPoint], [LyricTimeline.HeardNote]) {
        (events.beatGrid.map { .init(subbeat: $0.subbeat, time: $0.time) },
         events.melodyOnsets.map { .init(subbeat: $0.subbeat, midi: $0.midi) })
    }

    /// Same, from the `events.json` that `yue2 transcribe` writes.
    static func grid(eventsJSON url: URL) throws -> ([LyricTimeline.GridPoint], [LyricTimeline.HeardNote]) {
        guard let rows = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]] else {
            throw ValidationError("\(url.path): not an events.json array")
        }
        var points: [LyricTimeline.GridPoint] = [], heard: [LyricTimeline.HeardNote] = []
        for row in rows {
            guard let subbeat = row["subbeat"] as? Int else { continue }
            if let time = row["time"] as? Double { points.append(.init(subbeat: subbeat, time: time)) }
            for note in row["melody"] as? [[String: Any]] ?? [] {
                if let pitch = note["pitch"] as? Int { heard.append(.init(subbeat: subbeat, midi: pitch)) }
            }
        }
        return (points, heard)
    }

    /// Melody notes SheetSage2 attributes to a singing voice (track 0): 0 on a clean
    /// instrumental, 25–33 on a render that hums or sings a verse, ~66 on a sung minute.
    static func vocalNotes(_ events: [SheetSage2Event]) -> Int {
        events.reduce(0) { $0 + ($1.values.melody ?? []).filter { $0.track == 0 }.count }
    }
}
