// TranscribeCommand.swift - `yue2 transcribe`: recording → ABC score with SheetSage2 (plan/14-sheetsage2.md)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import SheetSage2Core
import YuE2Core

struct TranscribeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe a recording (a mixed song) into an ABC score with SheetSage2, for generate --abc-file / --abc-prefix-file with --cot melody."
    )

    @Option(name: .long, help: "SheetSage2 weights directory (default: $YUE2_MODELS_DIR/SheetSage2-fp16 if present, else $YUE2_MODELS_DIR/SheetSage2).")
    var model: String?

    @Option(name: .long, help: "Path to the recording (WAV, M4A, MP3…).")
    var audio: String

    @Option(name: .long, help: "Reference profile: \(SheetSage2Profile.all.map(\.id).joined(separator: ", ")) (default 16bit-fast).")
    var profile: String = SheetSage2Profile.default.id

    @Option(name: .long, help: "Override the profile's precision with an unquantized fp32 or bf16 run (parity and comparisons).")
    var precision: String?

    @Flag(name: .long, help: "Keep the chord symbols in the score (default: melody only, as for covers).")
    var chords = false

    @Option(name: .long, help: "Output directory for score.abc, events.json and tokens.json.")
    var out: String

    func run() async throws {
        let directory: URL
        if let model {
            directory = URL(fileURLWithPath: model)
        } else if let root = ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"], !root.isEmpty {
            let pack = URL(fileURLWithPath: root).appendingPathComponent("SheetSage2-fp16")
            directory = FileManager.default.fileExists(atPath: pack.appendingPathComponent("model.safetensors").path)
                ? pack : URL(fileURLWithPath: root).appendingPathComponent("SheetSage2")
        } else {
            throw ValidationError("pass --model or set YUE2_MODELS_DIR")
        }
        guard let reference = SheetSage2Profile.named(profile) else {
            throw ValidationError("unknown profile \(profile); one of \(SheetSage2Profile.all.map(\.id).joined(separator: ", "))")
        }
        let dtypes: [String: DType] = ["fp32": .float32, "bf16": .bfloat16]
        if let precision, dtypes[precision] == nil { throw ValidationError("--precision must be fp32 or bf16") }

        let started = Date()
        let loadSampler = FootprintSampler()
        loadSampler.start()
        let sheetsage = try precision.flatMap { dtypes[$0] }.map {
            try SheetSage2Model.load(directory: directory, dtype: $0, subsamplingFloat32: $0 != .float32)
        } ?? SheetSage2Model.load(directory: directory, profile: reference)
        let loadPeak = loadSampler.stop()
        let afterLoad = FootprintSampler.current()
        let loaded = Date()
        let runSampler = FootprintSampler()
        runSampler.start()
        // Upstream mixes to mono and resamples to 24 kHz; AVAudioConverter does the resampling.
        let stereo = try AudioImporter.loadAudio(url: URL(fileURLWithPath: audio), sampleRate: Double(sheetsage.config.samplingRate))
        let waveform = stereo[0].mean(axis: -1)
        var count = 0
        let transcriber = precision == nil ? SheetSage2Transcriber(model: sheetsage, profile: reference) : SheetSage2Transcriber(model: sheetsage)
        let result = try transcriber.transcribe(waveform, melodyOnly: !chords) { window, _ in
            count += 1
            if count % 256 == 0 { print("window \(window + 1): \(count) tokens…") }
        }
        let finished = Date()
        let runPeak = runSampler.stop()

        let dir = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let abc = result.abc {
            try abc.write(to: dir.appendingPathComponent("score.abc"), atomically: true, encoding: .utf8)
        }
        try JSONSerialization.data(withJSONObject: result.tokens, options: []).write(to: dir.appendingPathComponent("tokens.json"))
        let summary: [String: Any] = [
            "audio": URL(fileURLWithPath: audio).lastPathComponent, "duration_seconds": result.durationSeconds,
            "events": result.events.count, "windows": result.tokens.count, "tokens": result.tokens.map(\.count),
            "abc_error": result.abcError as Any, "warnings": result.warnings, "profile": precision == nil ? reference.id : "unquantized-\(precision!)",
            "load_seconds": loaded.timeIntervalSince(started), "transcribe_seconds": finished.timeIntervalSince(loaded),
            "footprint_load_peak_mb": loadPeak / 1_048_576, "footprint_after_load_mb": afterLoad / 1_048_576,
            "footprint_transcribe_peak_mb": runPeak / 1_048_576,
        ]
        try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
            .write(to: dir.appendingPathComponent("result.json"))
        if let abc = result.abc {
            print(abc)
        } else {
            print("no ABC score: \(result.abcError ?? "unknown error")")
        }
        print(String(format: "transcribed %.1f s of audio in %.1f s (load %.1f s), %d tokens",
                     result.durationSeconds, finished.timeIntervalSince(loaded), loaded.timeIntervalSince(started),
                     result.tokens.map(\.count).reduce(0, +)))
        print("FOOTPRINT load_peak_mb=\(loadPeak / 1_048_576) after_load_mb=\(afterLoad / 1_048_576) transcribe_peak_mb=\(runPeak / 1_048_576)")
    }
}
