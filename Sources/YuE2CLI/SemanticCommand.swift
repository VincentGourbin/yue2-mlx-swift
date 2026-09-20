// SemanticCommand.swift - `yue2 semantic`: semantic-phase generation from a saved plan (T-2.10)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

private struct SemanticMetadata: Codable {
    let timing: GenerationTiming
    let truncated: Bool
}

struct SemanticCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "semantic",
        abstract: "Generate semantic (codec) tokens from a saved plan."
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var samplingOverrides: SamplingOverrides

    @Option(name: .long, help: "Directory containing the saved plan (yue2 plan's --out).")
    var plan: String

    @Option(name: .long, help: "Output directory for semantic.npy and semantic.json.")
    var out: String

    func run() async throws {
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let savedPlan = try SymbolicPlan.load(from: URL(fileURLWithPath: plan))

        YuE2MemoryManager.configure(for: .ar)
        let session = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead)
        let semanticSampling = try samplingOverrides.semanticSampling(default: session.config.semantic)
        let generator = SemanticGenerator(model: session.model, tokenizer: session.tokenizer, config: session.config)

        var generated = 0
        let result = try generator.generateSemantic(plan: savedPlan, sampling: semanticSampling) { _ in
            generated += 1
            if generated % 100 == 0 {
                FileHandle.standardError.write(Data("semantic \(generated) tokens\u{2026}\n".utf8))
            }
        }

        let outURL = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: outURL, withIntermediateDirectories: true)
        try NumpyIO.writeInt32(result.tokens.map(Int32.init), url: outURL.appendingPathComponent("semantic.npy"))

        let metadata = SemanticMetadata(timing: result.timing, truncated: result.truncated)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        try encoder.encode(metadata).write(to: outURL.appendingPathComponent("semantic.json"))

        let summary =
            "semantic \(savedPlan.request.id): \(result.timing.contentTokens) tokens \u{00B7} "
            + "\(String(format: "%.1f", result.timing.outputTPS)) tok/s\n"
        FileHandle.standardError.write(Data(summary.utf8))
    }
}
