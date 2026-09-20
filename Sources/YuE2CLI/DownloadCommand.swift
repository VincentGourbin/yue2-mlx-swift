// DownloadCommand.swift - `yue2 download`
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

/// Downloads YuE2 checkpoints into `$YUE2_MODELS_DIR` (or `--models-dir`).
struct DownloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download YuE2 checkpoints (LM and/or VAE) from HuggingFace."
    )

    @Option(name: .long, help: "Directory to download checkpoints into (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Comma-separated models to download: lm, vae, vae-legacy.")
    var model: String = "lm,vae"

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let models = try Self.parseModels(model)
        let downloader = ModelDownloader(modelsDir: dir)
        let throttle = PercentThrottle()
        for target in models {
            print("\(target.rawValue) — \(target.license.name) (\(target.license.allowsCommercialUse ? "commercial use allowed" : "non-commercial only")): \(target.license.url)")
            try await downloader.download(target) { progress in
                guard progress.totalBytes > 0 else { return }
                let percent = Int(100 * Double(progress.writtenBytes) / Double(progress.totalBytes))
                guard percent % 5 == 0, throttle.shouldReport(file: progress.file, percent: percent) else { return }
                print("  [\(progress.fileIndex + 1)/\(progress.fileCount)] \(progress.file): \(percent)%")
            }
        }
        print("Download complete: \(dir.path)")
    }

    private static func parseModels(_ raw: String) throws -> [YuE2Model] {
        try raw.split(separator: ",").map { token in
            switch token.trimmingCharacters(in: .whitespaces) {
            case "lm": return .lm
            case "vae": return .vae
            case "vae-legacy": return .vaeLegacy
            case let unknown: throw ValidationError("Unknown model '\(unknown)': expected lm, vae, vae-legacy")
            }
        }
    }
}

/// Deduplicates progress printouts per file; a plain `var` can't be captured by the
/// `@Sendable` progress closure under Swift 6 strict concurrency.
private final class PercentThrottle: @unchecked Sendable {
    private let lock = NSLock()
    private var lastReported: [String: Int] = [:]

    func shouldReport(file: String, percent: Int) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard lastReported[file] != percent else { return false }
        lastReported[file] = percent
        return true
    }
}
