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

    @Option(name: .long, help: "SheetSage2 weights directory (default: the profile's pack under $YUE2_MODELS_DIR — SheetSage2-q8 / -q4 / -fp16 — if present, else SheetSage2-fp16, else SheetSage2).")
    var model: String?

    @Option(name: .long, help: "Path to the recording (WAV, M4A, MP3…).")
    var audio: String?

    @Option(name: .long, help: "Reference profile: \(SheetSage2Profile.all.map(\.id).joined(separator: ", ")) (default 16bit-fast).")
    var profile: String = SheetSage2Profile.default.id

    @Option(name: .long, help: "Override the profile's precision with an unquantized fp32 or bf16 run (parity and comparisons).")
    var precision: String?

    @Flag(name: .long, help: "Keep the chord symbols in the score (default: melody only, as for covers).")
    var chords = false

    @Flag(name: .long, help: "Load the model, print its footprint and load time, and stop (no audio, no GPU work).")
    var loadOnly = false

    @Option(name: .long, help: "Output directory for score.abc, events.json and tokens.json.")
    var out: String = "."

    func run() async throws {
        let directory: URL
        if let model {
            directory = URL(fileURLWithPath: model)
        } else if let root = ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"], !root.isEmpty {
            let base = URL(fileURLWithPath: root)
            let profileBits = SheetSage2Profile.named(profile)?.bits ?? 16
            let packName = ["SheetSage2-q8", "SheetSage2-q4"][safe: profileBits == 8 ? 0 : profileBits == 4 ? 1 : -1]
            let candidates = (precision == nil ? [packName].compactMap { $0 } : []) + ["SheetSage2-fp16", "SheetSage2"]
            directory = candidates.map { base.appendingPathComponent($0) }
                .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.safetensors").path) }
                ?? base.appendingPathComponent("SheetSage2")
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
        if loadOnly {
            print(String(format: "LOAD %@ from %@: %.2f s, footprint peak %d MB, after %d MB, MLX active %d MB",
                         reference.id, directory.lastPathComponent, loaded.timeIntervalSince(started),
                         loadPeak / 1_048_576, afterLoad / 1_048_576, Memory.activeMemory / 1_048_576))
            return
        }
        let runSampler = FootprintSampler()
        runSampler.start()
        // Upstream mixes to mono and resamples to 24 kHz; AVAudioConverter does the resampling.
        guard let audio else { throw ValidationError("--audio is required (except with --load-only)") }
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

extension Array {
    subscript(safe index: Int) -> Element? { indices.contains(index) ? self[index] : nil }
}

/// `yue2 sheetsage2-pack`: writes a prequantized pack for the 8/4-bit transcription profiles. The
/// quantization runs on the GPU by default, like the profiles' on-the-fly quantization on a Mac or an
/// iPhone: the CPU kernel rounds differently and would give a pack that is not bit-identical to it.
struct SheetSage2PackCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sheetsage2-pack",
        abstract: "Write a prequantized SheetSage2 pack (8 or 4 bits) from the fp16 pack or the upstream release."
    )

    @Option(name: .long, help: "Source weights directory (fp16 pack, merged snapshot or upstream release).")
    var model: String

    @Option(name: .long, help: "Bits of the Conformer's linear layers: 8 or 4.")
    var bits: Int

    @Flag(name: .long, help: "Quantize on the CPU (not bit-identical to the profiles' GPU quantization).")
    var cpu = false

    @Option(name: .long, help: "Output directory (config.json + model.safetensors).")
    var out: String

    func run() async throws {
        guard let profile = SheetSage2Profile.named("\(bits)bit-fast"), profile.quantizationBits != nil else {
            throw ValidationError("--bits must be 8 or 4")
        }
        try Device.withDefaultDevice(cpu ? .cpu : .gpu) {
            let started = Date()
            let model = try SheetSage2Model.load(directory: URL(fileURLWithPath: model), profile: profile)
            try SheetSage2Pack.save(model, bits: bits, to: URL(fileURLWithPath: out))
            print(String(format: "%d-bit pack written to %@ in %.1f s", bits, out, Date().timeIntervalSince(started)))
        }
    }
}
