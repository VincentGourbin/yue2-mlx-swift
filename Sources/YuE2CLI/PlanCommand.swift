// PlanCommand.swift - `yue2 plan`: ABC-phase generation, saved to disk (plan §2.3/§8.3, T-2.10)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import YuE2Core

struct PlanCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "plan",
        abstract: "Generate an ABC score plan from a song request."
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var requestOptions: RequestOptions
    @OptionGroup var samplingOverrides: SamplingOverrides

    @Option(name: .long, help: "Output directory for the saved plan.")
    var out: String

    func run() async throws {
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let request = try requestOptions.makeRequest()

        YuE2MemoryManager.configure(for: .ar)
        let session = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead,
            precision: modelOptions.precision)
        let abcSampling = try samplingOverrides.abcSampling(default: session.config.abc)

        let planner = Planner(model: session.model, tokenizer: session.tokenizer, config: session.config)
        let plan = try planner.plan(request: request, abcSampling: abcSampling)
        try plan.save(to: URL(fileURLWithPath: out))

        let tokens = plan.timing?.contentTokens ?? 0
        let tps = plan.timing?.outputTPS ?? 0
        FileHandle.standardError.write(
            Data("planning score \(request.id): \(tokens) tokens \u{00B7} \(String(format: "%.1f", tps)) tok/s\n".utf8))
    }
}
