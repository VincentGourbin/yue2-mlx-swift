// RemixCommand.swift - EXPERIMENTAL: SDEdit-style rewrite of a real latent (not part of the plan)
// Copyright 2026 Vincent Gourbin
//
// No fixture, no parity budget, no tolerance target: this mixes a real VAE-encoded latent with
// noise at an intermediate flow-matching timestep (`t_start = 1 - startStep/steps`, the exact
// linear interpolant `nar.py`'s own `solve` trains against: `x_t = t*noise + (1-t)*data`), then
// finishes the ODE with a *different* semantic conditioning (a fresh SongRequest's codec tokens).
// Whether this sounds like anything coherent is an open question, not a validated feature.

import ArgumentParser
import Foundation
import MLX
import YuE2Core

struct RemixCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "remix-experimental",
        abstract: "EXPERIMENTAL, unvalidated: rewrite a `yue2 encode`d latent under new style/lyrics conditioning."
    )

    @OptionGroup var modelOptions: ModelOptions
    @OptionGroup var requestOptions: RequestOptions

    @Option(name: .long, help: "latent.npy from `yue2 encode` — the source to remix.")
    var sourceLatent: String

    @Option(name: .long, help: "Frames to take from the start of --source-latent (1 frame = 40ms).")
    var frames: Int

    @Option(name: .long, help: "Flow-matching ODE steps.")
    var odeSteps: Int = 32

    @Option(name: .long, help: "Fraction of noise mixed in at the start, in (0,1). Higher = more transformed, less faithful to the source.")
    var noiseFraction: Double = 0.7

    @Option(name: .long, help: "Which VAE checkpoint to decode with.")
    var vae: DecodeCommand.VAEVariant = .standard

    @Option(name: .long, help: "Output directory.")
    var out: String

    @Option(name: .long, help: "WAV sample format.")
    var format: DecodeCommand.WAVFormat = .int16

    func run() async throws {
        guard frames >= 1 else {
            throw YuE2Error.invalidRequest("--frames must be positive")
        }
        guard noiseFraction > 0, noiseFraction < 1 else {
            throw YuE2Error.invalidRequest("--noise-fraction must be in (0,1)")
        }
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let request = try requestOptions.makeRequest()

        let (sourceShape, sourceData) = try NumpyIO.readFloat32(URL(fileURLWithPath: sourceLatent))
        guard sourceShape.count == 2, sourceShape[1] == 64, sourceShape[0] >= frames else {
            throw YuE2Error.invalidRequest(
                "--source-latent shape \(sourceShape) can't supply --frames \(frames) (need [>= frames, 64])")
        }
        let sourceSlice = MLXArray(sourceData).reshaped(sourceShape)[0..<frames, 0...]

        YuE2MemoryManager.configure(for: .ar)
        let session = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead,
            precision: modelOptions.precision)
        let vaeModel = try YuE2VAE.load(directory: modelsDir.appendingPathComponent(vae.model.directoryName))

        let planner = Planner(model: session.model, tokenizer: session.tokenizer, config: session.config)
        let plan = try planner.plan(request: request)

        let forcedSampling = try Sampling(
            temperature: session.config.semantic.temperature, topP: session.config.semantic.topP,
            topK: session.config.semantic.topK, repetitionPenalty: session.config.semantic.repetitionPenalty,
            penaltyWindow: session.config.semantic.penaltyWindow, minTokens: frames, maxTokens: frames)
        let semanticGen = SemanticGenerator(model: session.model, tokenizer: session.tokenizer, config: session.config)
        let semantic = try semanticGen.generateSemantic(plan: plan, sampling: forcedSampling)

        YuE2MemoryManager.configure(for: .nar)
        guard let chunk = try NARChunk.songChunks(prefix: plan.prefix, codec: semantic.tokens, seed: UInt64(request.seed)).first
        else {
            throw YuE2Error.invalidRequest("empty codec sequence")
        }
        let cachedNAR = CachedNAR(model: session.model, chunk: chunk)

        let stepsD = Double(odeSteps)
        // `t_start` (the ODE's own `t = 1 - step/steps`) should equal `noiseFraction` directly —
        // higher fraction = more noise = starting further from the source, hence `(1 - fraction)`
        // here (bug caught by hand: the naive `noiseFraction*steps` gives `t_start = 1-fraction`,
        // the opposite of what the flag promises).
        let startStep = min(odeSteps - 1, max(0, Int(((1 - noiseFraction) * stepsD).rounded())))
        let tStart = 1.0 - Double(startStep) / stepsD
        let mixed = chunk.noise * Float(tStart) + sourceSlice.asType(.float32) * Float(1 - tStart)

        let resultLatent = try cachedNAR.solve(steps: odeSteps, initialState: mixed, startStep: startStep)

        YuE2MemoryManager.releaseBetweenStages()
        YuE2MemoryManager.configure(for: .vae)
        let audio = try vaeModel.decodeTiled(resultLatent.expandedDimensions(axis: 0))
        let outDir = URL(fileURLWithPath: out)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        try AudioExporter.exportToWAV(
            audio: audio.transposed(0, 2, 1), url: outDir.appendingPathComponent("remix.wav"),
            sampleRate: vaeModel.config.sampleRate, format: format.sampleFormat)

        print(
            "remix: \(frames) frames, noiseFraction=\(noiseFraction) (startStep=\(startStep)/\(odeSteps),"
                + " t_start=\(String(format: "%.3f", tStart))) -> \(outDir.appendingPathComponent("remix.wav").path)"
        )
    }
}
