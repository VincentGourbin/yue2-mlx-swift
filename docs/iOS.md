# iPhone: integrating, measuring, fitting in 8 GB

The engine runs on iOS 27 (Metal through MLX; Core AI optional for the VAE). The reference app, `YuE2Studio`, lives in a separate repository (`yue2-ios`) and pins this package at an exact version (`Scripts/pin-engine.sh v1.0.0` on its side). This document says what the app must do and what was measured on an iPhone 15 Pro Max (A17 Pro, 8 GB, iOS 27.0).

## Xcode project

- Deployment target iOS 27; dependency `YuE2Core` (and `MLXProfiler` for measurements); mlx-swift compiles its metallib for iOS inside the Xcode build, nothing to do.
- Capability **Increased Memory Limit** (`com.apple.developer.kernel.increased-memory-limit`): without it the jetsam limit falls below the 3.4 GB of `4bit-lean`.
- `UIFileSharingEnabled` to drop the pack through Finder and retrieve WAVs.
- Do not link `CoreAI` explicitly (absent from the simulator SDK; auto-linked where `import CoreAI` compiles).
- The Simulator is useless for inference: MLX crashes on the first tensor (software Metal); a physical device is required.

## The iPhone profile

`4bit-lean`: `int4-mixed-head` pack (2.5 GB, dropped into `Caches/models/YuE2-3B/mlx-prequantized/int4-mixed-head/`), fp16, stage-scoped residency, packed NAR, VAE fp16 tile 256, memory limits sized from `os_proc_available_memory` (cache ≈ 1 GB, threshold = available − 1.25 GB). `8bit-lean` (4.4 GB) also fits. bf16 does not.

```swift
let profile = YuE2ReferenceProfile.named("4bit-lean")!
profile.applyGlobalPolicy()
```

## The four stages, resumable

| Stage | Call | Resident | Artifact |
|---|---|---|---|
| Plan (ABC) | `Planner.plan(request:)` | AR path + embeddings + head | `plan.json`, `score.abc` |
| Semantic | `SemanticGenerator.generateSemantic(plan:)` | same | `semantic.npy` (+ KV cache in memory when stage 3 follows) |
| NAR | `Synthesizer.synthesize(prefix:codec:seed:reuseCache:onBeforeChunk:resume:onStep:)` | NAR path (AR released once the cache is ready) | `latent.npy`; `nar-checkpoint.json` + `nar-state.npy` every step |
| VAE | `decodeTiled(using: vae, latents, coreFrames: 256)` | nothing from the LM | `audio.wav` |

The simplest remains `YuE2Pipeline` with the mobile-profile defaults; persist `onNARStep` in the song's folder, resume with `narResume`, reload one `ModelSession` per song.

## Losing the GPU in the background

iOS cuts the GPU as soon as the app leaves the foreground (`kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`), with no grace period, and mlx-core raises the error inside the Metal handler: an uncatchable crash. Three measures in the engine, two gestures in the app:

1. `YuE2GPUGate.shared.suspend()` on `scenePhase == .inactive`, `resume()` on `.active` and on every cancellation.
2. Persist every `NARCheckpoint` received through `onNARStep`; on return, resume with `narResume`. An incident costs one ODE step plus the re-prefill (≈ 4 s), not the stage.
3. Tell users: "Stay in the app during synthesis: iOS cuts the GPU in the background."

What remains (switching during the multitasking animation, before the worker is parked) only closes with an mlx-core patch (record the error, rethrow on the next `eval`) — proposed upstream, not integrated.

## What was measured (22 September 2026, int4-mixed-head pack, before the v1.0.0 fixes)

| Point | Measurement |
|---|---|
| AR-only load | 5.6 s, 3.0 GB footprint (≈ 1.4 GB of it spurious, fixed in v1.0.0: fp16 cast of the parked branch) |
| AR, 512 tokens | 25 tok/s, TTFT 0.3 s (Mac bf16: 60-67 tok/s) |
| NAR, ms per evaluation at 256 / 512 / 1 024 frames | 665 / 1 294 / 2 776 cold; 3 700-4 050 at 1 024 once `fair` |
| Thermal | `nominal` → `fair` after ≈ 60 s of sustained GPU, NAR −31 to −46 %; `serious` on a 60 s song, the app paces (2 s rest between steps) |
| VAE, 1 024 frames | MLX fp16/256: 6.0 s, 2.2 GB peak; Core AI GPU: 5.4 s, 2.1 GB peak, 2 GB resident from load |
| 30 s song | 273 s, 3.6 GB peak (plan phase; expected ≈ 2.2-2.4 GB with v1.0.0), listening validated |
| 60 s song | ≈ 12.7 min excluding loads, 3.5 GB peak |
| NAR Core AI GPU | 4.4 s per evaluation at 256 frames (6.6× MLX), buckets ≤ 1 024 prefix tokens: not retained |

Measurement rule on the device: any NAR point past the first minute of a pass is a throttled point; for a cold value, one point per launch, phone rested. Record `SystemMetrics.processFootprint()` before / after / peak (`FootprintSampler`), `os_proc_available_memory()` at start, `ProcessInfo.thermalState` before / after, and the exact configuration. A point that "disappears" (jetsam, no Swift error) is a measurement: that is the limit.

## To re-measure with v1.0.0

Plan-phase peak (fp16 fix), effect of the adaptive limits on time, switching to Safari during each stage (gate + resume), Core AI GPU VAE on the device, 16 and 24 ODE steps by ear.

## Core AI

`Scripts/coreai/export_vae.py` and `export_nar_stack.py` (`coreai-torch` 0.4.2, `coreai-core` 1.0.0b2) produce the `.aimodel` assets; `xcrun coreai-build compile --platform iOS --preferred-compute gpu` compiles them. Never a CPU specialization for the VAE (transposed convolution is wrong on CPU for kernels ≥ 8). The Neural Engine is a dead end with this toolchain version (silent fallback of transposed convolutions, `dequantize` crash).
