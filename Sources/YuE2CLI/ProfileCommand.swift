// ProfileCommand.swift - `yue2 profile plan|semantic`: LLM metrics + Chrome Trace (T-2.11)
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLXProfiler
import YuE2Core

struct ProfileCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "profile",
        abstract: "Profile the ABC or semantic generation phase (TTFT, prefill/decode tok/s, memory, Chrome Trace).",
        subcommands: [ProfilePlanCommand.self, ProfileSemanticCommand.self]
    )
}

/// Runs `session.finish()`, prints the phase report and `getLLMMetrics().summary`, and writes
/// `trace.json` — shared by both subcommands so the two reports stay identical in shape.
private func finishAndReport(session: ProfilingSession, out: String) throws {
    session.finish()
    print(session.generateReport())
    print(MLXProfiler.shared.getLLMMetrics().summary)

    let outDir = URL(fileURLWithPath: out)
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    try ChromeTraceExporter.export(session: session).write(to: outDir.appendingPathComponent("trace.json"))
}

struct ProfilePlanCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "plan",
        abstract: "Profile the ABC planning phase."
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var requestOptions: RequestOptions
    @OptionGroup var samplingOverrides: SamplingOverrides

    @Option(name: .long, help: "Output directory for trace.json.")
    var out: String

    func run() async throws {
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let request = try requestOptions.makeRequest()

        let session = ProfilingSession(config: .singleRun)
        session.title = "yue2 profile plan"
        session.metadata = ["model": "YuE2-3B", "cot": request.cot.rawValue, "seed": "\(request.seed)"]
        MLXProfiler.shared.enable()
        MLXProfiler.shared.activeSession = session

        YuE2MemoryManager.configure(for: .ar)
        session.beginPhase("1. Model Loading", category: .modelLoad)
        let modelSession = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead,
            precision: modelOptions.precision)
        session.endPhase("1. Model Loading", category: .modelLoad)

        let abcSampling = try samplingOverrides.abcSampling(default: modelSession.config.abc)
        let planner = Planner(model: modelSession.model, tokenizer: modelSession.tokenizer, config: modelSession.config)

        // Tokenization, prefill, and the sampling loop all happen inside `plan()`, whose public
        // API (T-2.8) has no seam to bracket them separately from here; "Prefill" and
        // "Generation" already appear as their own nested phases below, emitted directly by
        // `TokenGenerator.generate` (this fiche's profiler hooks), which is where that
        // boundary actually is.
        session.beginPhase("2. Plan Generation", category: .generation)
        _ = try planner.plan(request: request, abcSampling: abcSampling)
        session.endPhase("2. Plan Generation", category: .generation)

        try finishAndReport(session: session, out: out)
    }
}

struct ProfileSemanticCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "semantic",
        abstract: "Profile the semantic (codec) generation phase."
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var samplingOverrides: SamplingOverrides

    @Option(name: .long, help: "Directory containing the saved plan (yue2 plan's --out).")
    var plan: String

    @Option(name: .long, help: "Output directory for trace.json.")
    var out: String

    func run() async throws {
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let savedPlan = try SymbolicPlan.load(from: URL(fileURLWithPath: plan))

        let session = ProfilingSession(config: .singleRun)
        session.title = "yue2 profile semantic"
        session.metadata = [
            "model": "YuE2-3B", "cot": savedPlan.request.cot.rawValue, "seed": "\(savedPlan.request.seed)",
        ]
        MLXProfiler.shared.enable()
        MLXProfiler.shared.activeSession = session

        YuE2MemoryManager.configure(for: .ar)
        session.beginPhase("1. Model Loading", category: .modelLoad)
        let modelSession = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead,
            precision: modelOptions.precision)
        session.endPhase("1. Model Loading", category: .modelLoad)

        let semanticSampling = try samplingOverrides.semanticSampling(default: modelSession.config.semantic)
        let generator = SemanticGenerator(
            model: modelSession.model, tokenizer: modelSession.tokenizer, config: modelSession.config)

        session.beginPhase("2. Semantic Generation", category: .generation)
        _ = try generator.generateSemantic(plan: savedPlan, sampling: semanticSampling)
        session.endPhase("2. Semantic Generation", category: .generation)

        try finishAndReport(session: session, out: out)
    }
}
