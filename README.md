# yue2-mlx-swift

[![PocketAnthem on the App Store](https://img.shields.io/badge/App_Store-PocketAnthem-0D96F6?logo=apple&logoColor=white)](https://apps.apple.com/us/app/pocketanthem/id6815755463) [![Website](https://img.shields.io/badge/vinceforge.com-PocketAnthem-blue)](https://vinceforge.com) [![Featured in awesome-YuE](https://img.shields.io/badge/featured_in-awesome--YuE-orange)](https://github.com/RevolutionLA/awesome-YuE) [![Release](https://img.shields.io/github/v/release/VincentGourbin/yue2-mlx-swift)](https://github.com/VincentGourbin/yue2-mlx-swift/releases) [![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE) [![Buy Me a Coffee](https://img.shields.io/badge/Buy_me_a_coffee-fluxforgestudio-FFDD00?logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/fluxforgestudio)

Swift / MLX port of **YuE2** (m-a-p, September 2026) for Apple Silicon: lyrics and style → editable ABC score → semantic tokens → acoustic latents by flow matching → 48 kHz stereo song. The whole pipeline runs locally on a Mac or an iPhone 15 Pro Max, with no Python at inference time.

## PocketAnthem, on the App Store

[**PocketAnthem**](https://apps.apple.com/us/app/pocketanthem/id6815755463) is the iPhone app built on this engine: write lyrics, pick a style, and the song is generated **entirely on the phone** — no server, no upload, the 3-billion-parameter YuE2 model running through MLX on the A17 Pro with the `4bit-lean` configuration described below. Everything measured in this repository (stage-scoped residency, per-step checkpoints, the GPU gate, the published 4-bit pack) exists because that app had to fit in 8 GB and survive a trip to Safari mid-song. More on [vinceforge.com](https://vinceforge.com).

Listed in [awesome-YuE](https://github.com/RevolutionLA/awesome-YuE), the curated YuE / YuE2 ecosystem list.

**Version 1.2.0**: seven reference configurations, measured and reproducible, from 65.7 s for 70 s of audio (Mac, 4-bit) to a 2.6 GB peak (`4bit-tiny`), with their prequantized packs published on Hugging Face.

| Documentation | |
|---|---|
| [The reference configurations](docs/References.md) | which pack, which settings, for which machine |
| [Benchmarks: method, results, reproduction](docs/Benchmarks.md) | protocol, Mac and iPhone tables, Core AI, how to contribute |
| [Command line](docs/CLI.md) | every `yue2` command and option |
| [Swift API](docs/API.md) | `ModelSession`, `YuE2Pipeline`, residency, checkpoints, GPU gate, VAE backends, `SheetSage2Core` transcription |
| [iPhone](docs/iOS.md) | integration, losing the GPU in the background, on-device measurements |
| [Weights and packs](docs/Weights.md) | download, the published prequantized packs, licence |
| [Engineering knowledge base](docs/knowledge/index.md) | measurement log, decisions and verified pitfalls (OKF, in French, readable by humans and agents) |
| [Changelog](CHANGELOG.md) | 0.1 → 1.0.0 |

## Listen first

Ten examples are attached to the [v1.0.1 release](https://github.com/VincentGourbin/yue2-mlx-swift/releases/tag/v1.0.1) (AAC, 70 s each): the same techno-funk song about this port rendered by each of the six configurations (`swift-engine_technofunk_<id>.m4a`, same seed and lyrics), and the reference test song *City Lights* at 32 / 24 / 16 ODE steps.

## Getting started

Requirements: macOS 15+ (macOS 27 for Core AI), Xcode 26+ (Swift 6), Apple Silicon, about 8 GB of disk for the weights.

```bash
# build (xcodebuild, never `swift build` for a binary: MLX's metallib would not be found)
xcodebuild -scheme yue2 -configuration Release -destination 'platform=macOS' -derivedDataPath .xcodebuild build
BIN=.xcodebuild/Build/Products/Release/yue2

# weights (Hugging Face, CC-BY-NC-4.0) and tokenizer, plus the 4-bit reference pack (2.5 GB)
export YUE2_MODELS_DIR=$HOME/Library/Caches/models
$BIN download --models-dir $YUE2_MODELS_DIR --model lm,vae,int4-mixed-head
$BIN info

# a full song with a reference configuration
$BIN references
$BIN generate --request reference/yue/examples/song.json --reference 4bit-fast --out outputs/song --profile
```

The three reference packs are published at `VincentGOURBIN/yue2-mlx-packs` (`--model packs` fetches all of them, 10.4 GB); any other quantized preset is produced locally on first use (a few minutes, once). The request file follows the reference format: `style`, `lyrics` (with `[Verse]`, `[Chorus]`… tags), `cot`, `seed`, optional `abc`. The run's artifacts (`score.abc`, `semantic.npy`, `latent.npy`, `audio.wav`, `result.json`, `trace.json`) go to `--out`.

Stage by stage, each one resumable:

```bash
$BIN plan       --request song.json --out outputs/plan
$BIN semantic   --plan outputs/plan --out outputs/song
$BIN synthesize --plan outputs/plan --semantic outputs/song --out outputs/song --checkpoint-dir outputs/song
$BIN decode     --latents outputs/song/latent.npy --out outputs/song/audio.wav --precision fp16
```

From Swift:

```swift
let profile = YuE2ReferenceProfile.named("4bit-lean")!; profile.applyGlobalPolicy()
let session = try await ModelSession.load(modelsDir: dir, quant: profile.quant, quantizeHead: profile.quantizeHead,
                                          precision: profile.precision, residency: [.ar])
let vae = try await loadVAEBackend(.mlx, directory: dir.appendingPathComponent("YuE2-Vae"), precision: .fp16)
let song = try await YuE2Pipeline(session: session, vae: vae, config: session.config).generate(request: request)
```

## The reference configurations

70 s of audio, 32 ODE steps, MacBook Pro M3 Max, idle GPU, 120 s cool-down ([details](docs/References.md)).

| Id | Pack | Time | Peak memory | For |
|---|---|---|---|---|
| `4bit-fast` | int4-mixed-head, everything resident, NAR in bf16 | **65.7 s** | 8.9 GB | Mac: faster than real time |
| `4bit-lean` | int4-mixed-head, stage-scoped residency, fp16, 256-frame tile | 77.7 s | **3.4 GB** | iPhone 15 Pro Max, 8 GB Macs |
| `4bit-tiny` | as `4bit-lean`, 64-frame tile, 256 MB cache | 94.8 s | **2.6 GB** | the most constrained devices, same audio |
| `8bit-fast` | qint8-all-head, everything resident, NAR in bf16 | 76.5 s | 9.9 GB | Mac, 8-bit AR path |
| `8bit-lean` | qint8-all-head, residency, fp16, 256-frame tile | 79.3 s | 4.4 GB | iPhone with headroom, 8-16 GB Macs |
| `16bit-fast` | bf16, everything resident | 84.9 s | 12.0 GB | quality reference |
| `16bit-lean` | bf16, stage-scoped residency, 256-frame tile | 84.9 s | 6.9 GB | 16-24 GB Macs |

On the iPhone 15 Pro Max (`4bit-lean`): a 30 s song in 273 s, 60 s in ≈ 12.7 min under thermal throttling, first listening validated ([measurements](docs/iOS.md)).

## What the port does

1. **Reproduces YuE2 inference faithfully** in MLX Swift, module by module, with numerical proof at every step (PyTorch fixtures, tolerances fixed before porting: VAE fp32 max 2e-3, LM bf16 rel < 2 %, exact greedy tokens).
2. **Fits in a phone**: 4 / 8-bit quantization per branch, stage-scoped weight residency (the budget is the largest stage, not the sum), per-step ODE checkpoints and a GPU gate to survive iOS cutting the GPU in the background, memory limits sized from available memory.
3. **Measures before claiming**: every number comes from `swift-mlx-profiler`, with cool-down, A/B/B/A and a control; the log of measurements, decisions and pitfalls lives in `docs/knowledge/`.
4. **Takes audio in through the model's own entries**: a hummed melody becomes the ABC score the song follows (`yue2 melody`), a saved song gets variations (`yue2 vary`, SDEdit on its latents) or a new second half (`yue2 regenerate`, kept part re-imposed at every ODE step); a real song is transcribed into a score by `yue2 transcribe` (SheetSage2, ported to MLX Swift as `SheetSage2Core`) and the cover is generated on it (`cot: melody`, `--abc-file` or `--abc-prefix-file`). The model has no audio understanding at inference, so these are the honest paths; details in [docs/CLI.md](docs/CLI.md#audio-in-melody-vary-regenerate) and [covers](docs/CLI.md#covers-from-a-recording-transcribe-sheetsage2--abc--cot-melody).
5. **Times what it sings**: `timeline.json` gives every score bar, sung note and lyric word of a generated song in seconds and in bars/beats, read from the LM's own alignment heads (no audio analysis, no extra model; [docs/CLI.md](docs/CLI.md#timing-timelinejson---timeline-karaoke)); `--instrumental` renders a song without a voice.
6. **Serves as a test bed for plan execution by coding agents**: task cards in `tasks/`, the plan in `plan/`.

Out of scope: MERT2 embeddings as a feature (only SheetSage2's transcription is ported), `cot: off` (supported, not measured), bit-exact reproduction of PyTorch's random generator (parity on injected noise).

## Parity and tests

`Scripts/run-tests.sh` (swift-testing, parallelization 1: a known deadlock in mlx-swift). Two tiers: weight-free (tiny fixtures under `parity/`, 156 tests, always green) and real weights (`YUE2_MODELS_DIR`; `Real*`, `Quantization`, `GenerationSmoke`). `yue2 parity vae|lm|nar` against `Scripts/reference/real_fixtures.py`. Reference Python environment: `Scripts/setup-reference-env.sh`.

## Repository layout

```
Sources/YuE2Core/      Protocol, Tokenizer, Loading, Models/{LM,VAE}, Generation, Synthesis, Pipeline, Audio, Memory, Backends, CoreAI, Configuration
Sources/YuE2CLI/       yue2: info, download, generate, plan, semantic, synthesize, decode, encode, melody, vary, regenerate, references, parity, profile, bench-coreai-nar
Sources/YuE2BenchUI/   yue2-bench-ui: generation, profiler metrics, history, player
Tests/YuE2Tests/       swift-testing, two tiers
docs/                  guides (above) and the knowledge base
BENCHMARKS.md          every measurement, one line per run, single source = swift-mlx-profiler
Scripts/               build, tests, parity, Python fixtures, benchmarks, Core AI export
parity/                committed tiny fixtures
plan/, tasks/          the porting plan and the agent task cards (French)
```

## Licences

Code in this repository: **MIT**. Independent port, no copied code, of the architecture published by [m-a-p](https://huggingface.co/m-a-p) and its [reference implementation](https://github.com/multimodal-art-projection/YuE) (Apache-2.0).

Weights `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae` and tokenizer: **CC-BY-NC-4.0**, non-commercial use, downloaded separately from Hugging Face and never distributed here. `tokenizer.json` is derived from `Qwen/Qwen2.5-0.5B`'s (Apache-2.0), checked token by token against `qwen.tiktoken`. The audio examples attached to the release are outputs of those weights and carry the same non-commercial terms.
