// PackSupportTests.swift - a pack plus its support files is a loadable models directory (ASK Q13)
// Copyright 2026 Vincent Gourbin

import Foundation
import Testing
@testable import YuE2Core

private func packReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return FileManager.default.fileExists(
        atPath: dir.appendingPathComponent(YuE2Pack.int4MixedHead.localDirectory + "/model.safetensors").path)
}

@Suite("PackSupport", .enabled(if: packReady()))
struct PackSupportTests {
    /// Exactly what `ModelDownloader.download(pack:)` leaves in an empty directory — the pack and
    /// `YuE2Pack.supportFiles` — must be enough for `ModelSession.load` (no bf16 checkpoint, no
    /// weights manifest, no tiktoken file).
    @Test func packAndSupportFilesAloneLoad() async throws {
        let source = try #require(modelsDir())
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: dir) }
        let lm = dir.appendingPathComponent(YuE2Pack.supportLocalDirectory)
        let pack = dir.appendingPathComponent(YuE2Pack.int4MixedHead.localDirectory)
        try fm.createDirectory(at: pack, withIntermediateDirectories: true)
        for file in YuE2Pack.supportFiles {
            try fm.copyItem(at: source.appendingPathComponent("YuE2-3B/\(file)"), to: lm.appendingPathComponent(file))
        }
        try fm.createSymbolicLink(
            at: pack.appendingPathComponent("model.safetensors"),
            withDestinationURL: source.appendingPathComponent(YuE2Pack.int4MixedHead.localDirectory + "/model.safetensors"))
        let session = try await ModelSession.load(
            modelsDir: dir, quant: .int4Mixed, quantizeHead: true, precision: .fp16, residency: [.ar])
        #expect(!session.tokenizer.encode("la la").isEmpty)
    }
}
