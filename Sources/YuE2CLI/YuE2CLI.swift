// YuE2CLI.swift - yue2 command-line entry point
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

@main
struct YuE2CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "yue2",
        abstract: "YuE2 music generation on Apple Silicon (MLX)",
        version: YuE2.version,
        subcommands: [
            InfoCommand.self, DownloadCommand.self, ParityCommand.self, DecodeCommand.self,
            PlanCommand.self, SemanticCommand.self, ProfileCommand.self,
            GenerateCommand.self, SynthesizeCommand.self, EncodeCommand.self, RemixCommand.self,
        ],
        defaultSubcommand: InfoCommand.self
    )
}

/// Resolves the checkpoints directory: `--models-dir` if given, else `$YUE2_MODELS_DIR`.
func resolveModelsDir(_ option: String?) throws -> URL {
    if let option, !option.isEmpty { return URL(fileURLWithPath: option) }
    if let env = ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"], !env.isEmpty {
        return URL(fileURLWithPath: env)
    }
    throw YuE2Error.invalidRequest("no --models-dir given and $YUE2_MODELS_DIR is not set")
}

/// Prints the port version and, when a models directory is known, per-file checkpoint status.
struct InfoCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Show version and checkpoint status."
    )

    @Option(name: .long, help: "Directory containing YuE2 checkpoints (defaults to $YUE2_MODELS_DIR).")
    var modelsDir: String?

    func run() async throws {
        print("YuE2Swift \(YuE2.version)")
        guard let dir = try? resolveModelsDir(modelsDir) else {
            print("YUE2_MODELS_DIR: non défini")
            return
        }
        print("YUE2_MODELS_DIR: \(dir.path)")
        let downloader = ModelDownloader(modelsDir: dir)
        for target in YuE2Model.allCases {
            print("\n\(target.rawValue) (\(target.repoID)):")
            let status = try await downloader.verify(target)
            for file in target.files {
                print("  \((status[file] ?? false) ? "✓" : "✗") \(file)")
            }
            if let shaOK = status["model.safetensors.sha256"] {
                print("  sha256 \(shaOK ? "✓" : "✗") model.safetensors")
            }
        }
    }
}
