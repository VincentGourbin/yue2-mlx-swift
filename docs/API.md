# `YuE2Core` Swift API

Everything below is `public`, documented in the code (doc comments) and covered by tests. Package `YuE2Swift`, product `YuE2Core`; pinned dependencies (`mlx-swift` 0.31.6 exact, `swift-mlx-profiler` ≥ 1.5). macOS 15+ / iOS 27+, Swift 6, strict concurrency.

```swift
dependencies: [.package(url: "https://github.com/VincentGourbin/yue2-mlx-swift", exact: "1.0.0")]
```

## The shortest path

```swift
import YuE2Core

let profile = YuE2ReferenceProfile.named("4bit-lean")!
profile.applyGlobalPolicy()                       // memory profile, NAR compute, compiled decode

let session = try await ModelSession.load(
    modelsDir: modelsDir, quant: profile.quant, quantizeHead: profile.quantizeHead,
    precision: profile.precision,
    residency: profile.releaseWeightsBetweenStages ? [.ar] : .all)
let vae = try await loadVAEBackend(.mlx, directory: modelsDir.appendingPathComponent("YuE2-Vae"),
                                   precision: profile.vaePrecision)
let pipeline = YuE2Pipeline(
    session: session, vae: vae, config: session.config,
    vaeCoreFrames: profile.vaeCoreFrames,
    releaseWeightsBetweenStages: profile.releaseWeightsBetweenStages)

let song = try await pipeline.generate(request: request) { event in /* progress */ }
try song.saveArtifacts(to: outputDirectory)
```

`SongRequest` (style, lyrics, `cot`, seed, optional `abc`) is `Codable` and reads the CLI's JSON. `SongResult` carries the audio (`[1, samples, 2]` fp32), the latents, the tokens and the per-phase timings.

## Model and residency

```swift
public final class ModelSession {
    static func load(modelsDir:quant:quantizeHead:precision:residency:) async throws -> ModelSession
    var model: YuE2ForCausalLM; var tokenizer: YuE2Tokenizer; var config: GenerationConfig
    private(set) var resident: YuE2WeightResidency      // .ar, .nar, .all
    func loadWeights(of: YuE2WeightPath) throws           // no-op when already resident
    func releaseWeights(of: YuE2WeightPath) -> Int        // bytes released
    func releaseAllWeights() -> Int
}
```

A released branch is unusable until reloaded (it would compute on zeros): that is the contract of stage-scoped residency. Safetensors loading is lazy; a tensor that is never evaluated is never allocated. `applyPrecision(_:where:)` only casts resident branches (casting a parked branch materializes it: 1.44 GB, the iPhone peak of 0.2.1).

`YuE2ForCausalLM.dequantizeWeights(of: .nar)` replaces the 8-bit projections with bf16 `Linear`s (fast profiles); the branch can no longer be reloaded, so recreate the session for the next song.

## Pipeline and stages

```swift
public final class YuE2Pipeline {
    init(session:vae: any VAEDecoding, config:, vaeCoreFrames: Int? = nil, releaseWeightsBetweenStages: Bool? = nil)
    func generate(request:abcSampling:semanticSampling:profiling:onEvent:narResume:onNARStep:cancel:) async throws -> SongResult
}
```

The `nil` defaults come from the memory profile (Mac: 1024 / everything resident; mobile: 256 / stage-scoped residency). `PipelineEvent`: `.stage`, `.abcToken`, `.semanticToken`, `.narProgress`, `.vaeProgress`.

The stages exist separately, each resumable from its artifacts:

| Stage | Type | Input → output |
|---|---|---|
| 1 | `Planner.plan(request:abcSampling:onToken:cancel:)` | request → `SymbolicPlan` (score, prefix) |
| 2 | `SemanticGenerator.generateSemantic(plan:sampling:onToken:cancel:)` | plan → semantic tokens + KV cache |
| 3 | `Synthesizer.synthesize(prefix:codec:seed:steps:reuseCache:onBeforeChunk:resume:onStep:cancel:)` | tokens → latents `[T, 64]` |
| 4 | `decodeTiled(using: any VAEDecoding, _:coreFrames:haloFrames:)` | latents → audio |

`onBeforeChunk(needsARPath:)` is where residency releases the AR branch and loads the NAR one; `onStep` delivers a `NARCheckpoint` after every ODE step and `resume:` restarts from one (bit-exact resume, single-chunk songs).

```swift
public struct NARCheckpoint { chunkIndex, step, steps, state: MLXArray
    func save(to: URL) throws; static func load(from: URL) throws -> NARCheckpoint?; static func clear(in: URL) }
```

## Audio in and song edits

The model has no audio understanding at inference; its entries are its own artifacts. `YuE2Pipeline+Edit.swift` exposes the two edits an app needs, `MelodyTranscriber` turns a recording into the plan.

```swift
// hummed / sung melody → ABC score → song on that melody
let samples = try AudioImporter.loadAudio(url: recording)                     // [1, n, 2] fp32 at 48 kHz, or [Float]
var options = MelodyTranscriber.Options(); options.bpm = 110                   // tempo of the recording, section label, note floor
let melody = MelodyTranscriber.transcribe(audio: samples, sampleRate: 48_000, options: options)
melody.notes            // [MelodyNote(midi:startSixteenth:lengthSixteenths:)], melody.key, melody.voicedRatio
var request = SongRequest(style: style, lyrics: lyrics); request.abc = melody.abc   // the score is the plan, no planner run
let song = try await pipeline.generate(request: request)

// variation: same score, same tokens, SDEdit on the latents (strength 0 = unchanged, 1 = fresh)
let variation = try await pipeline.vary(song, strength: 0.4, seed: 2, onEvent: onEvent)
// "regenerate from here": keep up to a frame (40 ms), re-sample the tokens after it, solve against the kept part
let ending = try await pipeline.regenerate(song, fromFrame: 20 * 25, seed: 3, onEvent: onEvent)   // length: .keepSource by default, .free lets the AR decide

let saved = try SongResult.load(from: directory)                               // a generate --out directory, for later edits
```

Underneath: `SemanticGenerator.generateSemantic(plan:sampling:continuation:onToken:cancel:)` takes the kept tokens as the AR's past; `Synthesizer.synthesize(... edit: NAREdit?)` with `.variation(latents:strength:)` (start state `noise·t + latents·(1−t)`, `round((1−strength)·steps)` steps skipped) or `.keep(latents:frames:)` (kept frames re-imposed after every step, returned bit-exact); `CachedNAR.solve(... keep:)` is the inpainting mask. `MelodyTranscriber` is pure CPU (vDSP), deterministic, tested on synthetic tones; `NAREditTests` covers the two edits on the tiny model.

## Losing the GPU in the background (iOS)

iOS refuses every GPU submission from an app that is not in the foreground, and mlx-core raises the error inside the Metal completion handler, where it cannot be caught. The engine stops before that:

```swift
YuE2GPUGate.shared.suspend()   // scenePhase == .inactive
YuE2GPUGate.shared.resume()    // .active, and on every cancellation
```

The gate is awaited before every token, every NAR layer (when `YuE2ExecutionPolicy.evalPerLayer`, the mobile default: 28 units of 0.2-0.4 s instead of one 10-20 s graph) and every VAE tile; `wait(cancel:)` returns `false` when the cancellation lands during the wait. With per-step checkpoints, what remains costs one step.

## VAE backends

```swift
public protocol VAEDecoding {
    var config: VAEConfig { get }
    var enumeratedFrameCounts: [Int]? { get }        // nil = any length (MLX)
    func decode(_ z: MLXArray) async throws -> MLXArray
}
func loadVAEBackend(_ kind: VAEBackendKind, directory: URL, precision: VAEPrecision) async throws -> any VAEDecoding
func planVAETiles(frames:coreFrames:haloFrames:enumerated:) -> [VAETileWindow]
```

`.mlx` (always), `.coreaiGPU` (macOS/iOS 27, 54.7 dB parity thanks to `planVAETiles`' exact-shape windows), `.coreaiANE` (dead end with `coreai-torch` 0.4.2). `loadVAEBackend` throws `.backendUnavailable`, never substitutes silently: the MLX fallback belongs to the caller, logged. Never `.cpuOnly` for this model (transposed convolutions are wrong on CPU).

## Memory and execution policy

```swift
YuE2MemoryManager.profile                  // .mac | .mobile (#if os(iOS), YUE2_MEMORY_PROFILE)
YuE2MemoryManager.configure(for: .ar | .nar | .vae | .load)
YuE2MemoryManager.mobileLimitsMB()         // cache ≈ 1 GB, threshold = available − 1.25 GB
YuE2ExecutionPolicy.narCompute             // .packed | .dequantized
YuE2ExecutionPolicy.evalPerLayer           // mobile default
YuE2ExecutionPolicy.compiledDecode         // off, measured without gain
let sampler = FootprintSampler(); sampler.start(); …; sampler.stop(); sampler.peakMB; sampler.mlxActivePeakMB
```

`FootprintSampler` tracks `phys_footprint` (the number jetsam judges) on a dedicated thread, including what MLX does not see (Core AI, Metal).

## Reference profiles

```swift
public struct YuE2ReferenceProfile { id, bits, kind, quant, quantizeHead, precision, narCompute, compiledDecode,
    vaePrecision, vaeCoreFrames, releaseWeightsBetweenStages, memoryProfile, odeSteps, summary
    static let all: [YuE2ReferenceProfile]; static func named(_:) -> YuE2ReferenceProfile?; func applyGlobalPolicy() }
```

See [References.md](References.md) for the six and their measurements.

## Quantization

`YuE2Quantization`: `.none`, `.qint8`, `.int4` (AR path), `.qint8All`, `.int4All`, `.int4Mixed` (4-bit AR path, 8-bit NAR, 4-bit embeddings); `quantizeHead` adds `lm_head`. `LMWeightLoader.load` quantizes on the fly then exports `mlx-prequantized/<preset>[-head]/`; later loads read the export. Parities: AR int4 greedy 8/8, NAR 8-bit rel 0.049 over 32 steps (a 4-bit NAR fails: 0.15).

## Tests

`Scripts/run-tests.sh` (swift-testing, parallelization 1: a known deadlock in mlx-swift). Two tiers: weight-free (tiny fixtures under `parity/`, always green, 156 tests) and real weights (`YUE2_MODELS_DIR`, suites `Real*`, `Quantization`, `GenerationSmoke`). Every numerical change goes through `RealLMParityTests` (greedy 8/8 on both phases) and `RealNARParityTests`.
