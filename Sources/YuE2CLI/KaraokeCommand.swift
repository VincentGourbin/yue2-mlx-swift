// KaraokeCommand.swift - `yue2 karaoke`: lyric lines, words and syllables timed on a generated song
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

struct KaraokeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "karaoke",
        abstract: "Time the score bars and the lyrics (lines, words, syllables) of a generated song.",
        discussion: """
            Default: the LM's own alignment heads (needs the saved plan and semantic.npy, loads the AR \
            weights once for every --song). --sheetsage places the score on a SheetSage2 beat grid of \
            audio.wav instead. Writes timeline.json beside each song.
            """
    )

    @OptionGroup var modelOptions: ModelOptions

    @Flag(name: .long, help: "Clock from a SheetSage2 transcription of audio.wav instead of the LM's attention.")
    var sheetsage = false

    @Option(name: .long, help: "A saved song: a `yue2 generate --out` directory or an app song directory (plan/ + semantic.npy). Repeat for a batch (the model loads once).")
    var song: [String]

    @Flag(name: .long, help: "Skip songs that already have the output file (migrations).")
    var skipExisting = false

    @Option(name: .long, help: "Reuse an events.json written by `yue2 transcribe` instead of transcribing the audio.")
    var events: String?

    @Option(name: .long, help: "Lyrics language for syllabification: en or fr (default: fr when the style says French, else en).")
    var language: String?

    @Option(name: .long, help: "EXPERIMENTAL: word starts from a forced aligner, JSON {\"words\":[{\"t0\":seconds},…]} in lyric order.")
    var anchors: String?

    @Flag(name: .long, help: "Ignore the audio: place the score on its nominal tempo from t = 0 (comparison only, ~3 s off on a real render).")
    var nominal = false

    @Option(name: .long, help: "SheetSage2 weights directory (default: under $YUE2_MODELS_DIR).")
    var sheetsageModel: String?

    @Option(name: .long, help: "SheetSage2 profile for the beat grid (default 16bit-fast).")
    var profile: String = "16bit-fast"

    @Option(name: .long, help: "Output file for a single --song (default: <song>/timeline.json).")
    var out: String?

    private var usesAttention: Bool { !sheetsage && !nominal && events == nil }

    func validate() throws {
        if out != nil, song.count != 1 { throw ValidationError("--out needs exactly one --song") }
        if song.isEmpty { throw ValidationError("pass at least one --song") }
        if let language, LyricTimeline.Language(rawValue: language) == nil { throw ValidationError("--language must be en or fr") }
    }

    func run() async throws {
        var session: ModelSession?
        if usesAttention {
            YuE2MemoryManager.configure(for: .ar)
            session = try await ModelSession.load(
                modelsDir: try resolveModelsDir(modelOptions.modelsDir), quant: modelOptions.quant,
                quantizeHead: modelOptions.quantHead, precision: modelOptions.precision, residency: [.ar])
        }
        var failures = 0
        for path in song {
            let dir = URL(fileURLWithPath: path)
            let destination = out.map { URL(fileURLWithPath: $0) } ?? dir.appendingPathComponent("timeline.json")
            if skipExisting, FileManager.default.fileExists(atPath: destination.path) {
                print("\(dir.lastPathComponent): \(destination.lastPathComponent) exists, skipped")
                continue
            }
            do {
                let timeline = try analyze(dir, session: session)
                try timeline.write(to: destination)
                if song.count == 1 {
                    for line in timeline.lines {
                        print(String(format: "%6.2f–%6.2f  bar %3d  ", line.start, line.end, line.bar + 1)
                            + line.words.map { w in w.syllables.count > 1 ? w.syllables.map(\.text).joined(separator: "·") : w.text }.joined(separator: " "))
                    }
                }
                print(String(format: "%@: clock %@, %d bars, %d notes, %d lines → %@",
                             dir.lastPathComponent, timeline.grid, timeline.bars.count, timeline.notes.count,
                             timeline.lines.count, destination.path))
            } catch {
                failures += 1
                FileHandle.standardError.write(Data("\(dir.lastPathComponent): \(error)\n".utf8))
            }
        }
        if failures > 0 { throw ExitCode(1) }
    }

    private func analyze(_ dir: URL, session: ModelSession?) throws -> LyricTimeline {
        let forced = language.flatMap(LyricTimeline.Language.init(rawValue:))
        if let session {
            let started = Date()
            let timeline = try AttentionTimeline.reanalyze(
                song: dir, model: session.model, tokenizer: session.tokenizer, language: forced)
            FileHandle.standardError.write(Data(String(format: "attention timeline in %.1f s\n", Date().timeIntervalSince(started)).utf8))
            return timeline
        }
        // SheetSage2 / nominal clock: the score and the request, from the plan or loose files.
        let abc: String, request: SongRequest
        if let saved = try? AttentionTimeline.SavedSong(directory: dir), let planned = saved.plan.abc {
            (abc, request) = (planned, saved.plan.request)
        } else {
            abc = try String(contentsOf: dir.appendingPathComponent("score.abc"), encoding: .utf8)
            request = try JSONDecoder().decode(SongRequest.self, from: Data(contentsOf: dir.appendingPathComponent("request.json")))
        }
        return try placeOnGrid(dir: dir, abc: abc, request: request, language: forced ?? .guess(style: request.style))
    }

    private func placeOnGrid(dir: URL, abc: String, request: SongRequest, language lang: LyricTimeline.Language) throws -> LyricTimeline {
        var points: [LyricTimeline.GridPoint]?
        var heard: [LyricTimeline.HeardNote] = []
        if !nominal {
            if let events {
                (points, heard) = try SheetSage2Support.grid(eventsJSON: URL(fileURLWithPath: events))
            } else {
                let started = Date()
                let result = try SheetSage2Support.transcribe(audio: dir.appendingPathComponent("audio.wav"), model: sheetsageModel, profile: profile)
                (points, heard) = SheetSage2Support.grid(result.events)
                FileHandle.standardError.write(Data(String(format: "beat grid: %d points in %.1f s\n", points?.count ?? 0, Date().timeIntervalSince(started)).utf8))
            }
        }
        var anchorTimes: [Double?]?
        if let anchors {
            let object = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: anchors))) as? [String: Any]
            anchorTimes = (object?["words"] as? [[String: Any]])?.map { $0["t0"] as? Double }
        }
        return try LyricTimeline.build(
            abc: abc, lyrics: request.lyrics, language: lang, grid: points, heard: heard, anchors: anchorTimes)
    }
}
