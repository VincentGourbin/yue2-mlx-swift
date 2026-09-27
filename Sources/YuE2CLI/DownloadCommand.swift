// DownloadCommand.swift - `yue2 download`
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

/// Downloads YuE2 checkpoints into `$YUE2_MODELS_DIR` (or `--models-dir`).
struct DownloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download YuE2 checkpoints (LM, VAE) and prequantized packs from Hugging Face."
    )

    @Option(name: .long, help: "Directory to download checkpoints into (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    @Option(name: .long, help: "Comma-separated targets: lm, vae, vae-legacy, packs (all three prequantized packs), or a pack name (int4-mixed-head, qint8-all-head, int4-head).")
    var model: String = "lm,vae"

    func run() async throws {
        let dir = try resolveModelsDir(modelsDir)
        let targets = try Self.parseTargets(model)
        let downloader = ModelDownloader(modelsDir: dir)
        let throttle = PercentThrottle()
        let report: @Sendable (DownloadProgress) -> Void = { progress in
            guard progress.totalBytes > 0 else { return }
            let percent = Int(100 * Double(progress.writtenBytes) / Double(progress.totalBytes))
            guard percent % 5 == 0, throttle.shouldReport(file: progress.file, percent: percent) else { return }
            print("  [\(progress.fileIndex + 1)/\(progress.fileCount)] \(progress.file): \(percent)%")
        }
        for target in targets {
            switch target {
            case .model(let model):
                print("\(model.rawValue) — \(model.license.name) (\(model.license.allowsCommercialUse ? "commercial use allowed" : "non-commercial only")): \(model.license.url)")
                try await downloader.download(model, progress: report)
            case .pack(let pack):
                print("\(YuE2Pack.repoID)/\(pack.rawValue) (\(pack.approximateBytes / 1_000_000_000) GB) — \(pack.license.name), derivative of m-a-p/YuE2-3B, non-commercial only: \(pack.license.url)")
                try await downloader.download(pack: pack, progress: report)
                print("  verified: SHA-256 matches \(pack.rawValue)/model.safetensors.sha256")
            }
        }
        print("Download complete: \(dir.path)")
    }

    enum Target { case model(YuE2Model), pack(YuE2Pack) }

    static func parseTargets(_ raw: String) throws -> [Target] {
        var targets: [Target] = []
        for token in raw.split(separator: ",") {
            switch token.trimmingCharacters(in: .whitespaces) {
            case "lm": targets.append(.model(.lm))
            case "vae": targets.append(.model(.vae))
            case "vae-legacy": targets.append(.model(.vaeLegacy))
            case "packs": targets.append(contentsOf: YuE2Pack.allCases.map(Target.pack))
            case let name:
                guard let pack = YuE2Pack(rawValue: name) else {
                    throw ValidationError("Unknown target '\(name)': expected lm, vae, vae-legacy, packs, or a pack name (\(YuE2Pack.allCases.map(\.rawValue).joined(separator: ", ")))")
                }
                targets.append(.pack(pack))
            }
        }
        return targets
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
