// GenerationViewModel.swift - drives YuE2Pipeline from the GUI (T-4.7, O10)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLXProfiler
import YuE2Core

/// One generation's live progress, mirrored from `PipelineEvent` (`YuE2Pipeline.swift`).
enum GenerationStage: Equatable {
    case idle
    case loadingModel
    case running(String)
    case done
    case failed(String)
}

/// Owns the form state, runs `YuE2Pipeline.generate` on a detached `Task` (never on the main
/// actor — a 60-90 s generation would freeze the UI), and republishes progress/results back to
/// this `@MainActor` view model. Mirrors `yue2 generate`'s CLI flow (`GenerateCommand.swift`)
/// but drives a GUI instead of stdout.
@Observable
@MainActor
final class GenerationViewModel {
    // Request fields (mirrors `RequestOptions`/`SongRequest`)
    /// Remembered across launches (`UserDefaults`, per-machine — never committed): a double-clicked
    /// `.app` doesn't inherit `$YUE2_MODELS_DIR` from the shell, so the env var only helps the first
    /// time this runs from a terminal that has it; after that, whatever the user last picked sticks.
    var modelsDir: String = ModelDiscovery.rememberedOrDefaultModelsDir() {
        didSet { UserDefaults.standard.set(modelsDir, forKey: ModelDiscovery.modelsDirDefaultsKey) }
    }
    var style = "English, warm piano pop, expressive female voice, acoustic piano, light drums, 88 BPM"
    var lyrics = "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain\n\n[Chorus]\nLet the day come into view\nEvery road begins with you"
    var cot: CoTMode = .full
    var seed = 831_001
    var cfgScaleText = ""
    var requestID = "gui_song"
    /// Empty = the checkpoint's own default (usually well under a minute); otherwise a target
    /// duration in seconds, converted to a semantic token count (`ModelDiscovery.framesPerSecond`)
    /// and applied as `Sampling.minTokens`/`maxTokens` for the semantic phase (plan §2.1: one
    /// semantic token is one 40 ms latent frame). The model still decides the exact ending — this
    /// is a floor/ceiling on length, not a precise clock.
    var targetDurationText = ""

    // Model/quant selection
    var availableVariants: [AvailableVariant] = []
    var selectedVariant: AvailableVariant?
    var availableVAEs: [VAEChoice] = []
    var selectedVAE: VAEChoice = .standard

    // Progress
    private(set) var stage: GenerationStage = .idle
    private(set) var abcTokenCount = 0
    private(set) var semanticTokenCount = 0
    private(set) var narProgress: (completed: Int, total: Int) = (0, 0)
    private(set) var vaeProgress: (completed: Int, total: Int) = (0, 0)
    var isRunning: Bool { if case .running = stage { true } else if stage == .loadingModel { true } else { false } }

    // Results
    private(set) var lastMetrics: RunMetrics?
    let audioPlayer = AudioPlayerModel()
    private(set) var history: [HistoryEntry] = []
    private var runningTask: Task<Void, Never>?

    func refreshVariants() {
        let dir = URL(fileURLWithPath: modelsDir)
        availableVariants = ModelDiscovery.availableVariants(modelsDir: dir)
        availableVAEs = ModelDiscovery.availableVAEs(modelsDir: dir)
        if selectedVariant == nil { selectedVariant = availableVariants.first }
        if !availableVAEs.contains(selectedVAE), let first = availableVAEs.first { selectedVAE = first }
    }

    func cancel() {
        runningTask?.cancel()
    }

    /// `SongRequest` requires `0 <= seed < 2**63` (`Int.max` on a 64-bit platform is `2**63 - 1`).
    func randomizeSeed() {
        seed = Int.random(in: 0...Int.max)
    }

    func generate() {
        guard !isRunning, let variant = selectedVariant else { return }
        stage = .loadingModel
        abcTokenCount = 0
        semanticTokenCount = 0
        narProgress = (0, 0)
        vaeProgress = (0, 0)

        let dir = URL(fileURLWithPath: modelsDir)
        let vaeChoice = selectedVAE
        let cotChoice = cot
        let targetSeconds = Double(targetDurationText.trimmingCharacters(in: .whitespaces))
        guard let request = try? SongRequest(
            style: style, lyrics: lyrics, cot: cotChoice, seed: seed,
            cfgScale: Double(cfgScaleText), id: requestID)
        else {
            stage = .failed("requête invalide (style/lyrics/id)")
            return
        }

        runningTask = Task.detached { [weak self] in
            do {
                let profilingSession = ProfilingSession(config: .singleRun)
                profilingSession.title = "yue2-bench-ui generate"
                MLXProfiler.shared.enable()
                MLXProfiler.shared.activeSession = profilingSession

                profilingSession.beginPhase("1. Model Loading", category: .modelLoad)
                let session = try await ModelSession.load(
                    modelsDir: dir, quant: variant.quant, quantizeHead: variant.quantizeHead)
                let vae = try YuE2VAE.load(directory: dir.appendingPathComponent(vaeChoice.directoryName))
                profilingSession.endPhase("1. Model Loading", category: .modelLoad)

                await MainActor.run { [weak self] in self?.stage = .running("plan") }
                let pipeline = YuE2Pipeline(session: session, vae: vae, config: session.config)
                // `YuE2Pipeline` is a plain (non-Sendable) class; `request` is already `Sendable`
                // (`SongRequest`) and needs no wrapper.
                nonisolated(unsafe) let unsafePipeline = pipeline

                var semanticSampling: Sampling?
                if let targetSeconds, targetSeconds > 0 {
                    let base = session.config.semantic
                    let frames = max(1, Int((targetSeconds * ModelDiscovery.semanticFramesPerSecond).rounded()))
                    semanticSampling = try? Sampling(
                        temperature: base.temperature, topP: base.topP, topK: base.topK,
                        repetitionPenalty: base.repetitionPenalty, penaltyWindow: base.penaltyWindow,
                        minTokens: frames, maxTokens: frames + 200)
                }

                let result = try await unsafePipeline.generate(
                    request: request, semanticSampling: semanticSampling, profiling: profilingSession,
                    onEvent: { event in
                        Task { @MainActor [weak self] in self?.apply(event) }
                    },
                    cancel: { Task.isCancelled }
                )

                profilingSession.finish()
                let ttsSummary = MLXProfiler.shared.getTTSMetrics().summary
                let phases = profilingSession.phaseSummaries()
                let peakProcess = phases.compactMap(\.peakProcessMB).max()
                let peakActive = phases.compactMap(\.peakMLXActiveMB).max()
                let report = profilingSession.generateReport()

                nonisolated(unsafe) let unsafeResult = result
                await MainActor.run { [weak self] in
                    self?.finish(
                        result: unsafeResult, requestID: request.id, variantLabel: variant.label, cot: cotChoice.rawValue,
                        ttsSummary: ttsSummary, peakProcessMB: peakProcess, peakMLXActiveMB: peakActive, report: report)
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in self?.stage = .idle }
            } catch {
                await MainActor.run { [weak self] in self?.stage = .failed("\(error)") }
            }
        }
    }

    private func apply(_ event: PipelineEvent) {
        switch event {
        case .stage(let name): stage = .running(name)
        case .abcToken(let count): abcTokenCount = count
        case .semanticToken(let count): semanticTokenCount = count
        case .narProgress(let c, let t): narProgress = (c, t)
        case .vaeProgress(let c, let t): vaeProgress = (c, t)
        }
    }

    private func finish(
        result: SongResult, requestID: String, variantLabel: String, cot: String,
        ttsSummary: String, peakProcessMB: Double?, peakMLXActiveMB: Double?, report: String
    ) {
        stage = .done
        lastMetrics = RunMetrics(
            ttsSummary: ttsSummary, peakProcessMB: peakProcessMB, peakMLXActiveMB: peakMLXActiveMB, phaseReport: report)

        let audioURL = exportAudio(result: result, requestID: requestID)
        if let audioURL { audioPlayer.load(url: audioURL) }

        let audioSeconds = Double(result.audio.dim(1)) / Double(result.sampleRate)
        history.append(
            HistoryEntry(
                date: Date(), requestID: requestID, quant: variantLabel, cot: cot,
                abcTokens: result.timing.abc?.contentTokens ?? 0, abcTPS: result.timing.abc?.outputTPS ?? 0,
                semanticTokens: result.timing.semantic.contentTokens, semanticTPS: result.timing.semantic.outputTPS,
                narSeconds: result.timing.narSeconds, vaeSeconds: result.timing.vaeSeconds,
                audioSeconds: audioSeconds, e2eSeconds: result.timing.e2eSeconds,
                peakProcessMB: peakProcessMB, peakMLXActiveMB: peakMLXActiveMB, audioURL: audioURL))
    }

    /// Writes every artifact (`SongResult.saveArtifacts`) under `~/Documents/YuE2Bench/<id>-<ts>/`
    /// — a fixed, predictable location instead of a save panel per run, so a batch of A/B
    /// comparisons doesn't interrupt the user with a dialog every time.
    private func exportAudio(result: SongResult, requestID: String) -> URL? {
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        guard let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
        let dir = documents.appendingPathComponent("YuE2Bench").appendingPathComponent("\(requestID)-\(stamp)")
        do {
            try result.saveArtifacts(to: dir)
            return dir.appendingPathComponent("audio.wav")
        } catch {
            return nil
        }
    }
}
