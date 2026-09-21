# CLAUDE.md
## Project Overview
Swift MLX port of YuE2 (m-a-p/YuE2-3B + YuE2-Vae): lyrics + style → editable ABC score → semantic tokens → acoustic latents (flow matching) → 48 kHz stereo song. Text-only input; MERT2/SheetSage2 are not used at inference.
**PLAN.md is the porting spec** — read §2 (architecture facts) and §9 (pitfall checklist) before touching model code. Upstream reference: https://github.com/multimodal-art-projection/YuE (`src/yue2/*.py`).
## Build & Run
Build with `xcodebuild`, NOT `swift build` (SPM binaries cannot find MLX's `default.metallib`):
  xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build   # binary: .xcodebuild/Build/Products/Release/yue2
`swift build` is for compile checks only.
## Tests
Use `Scripts/run-tests.sh` (serial swift-testing: ABBA deadlock in mlx-swift between compiled functions and vjp; watchdog; relays every YUE2_* variable as TEST_RUNNER_YUE2_*). Two tiers: weight-free (always green) and checkpoint-backed (gated on YUE2_MODELS_DIR, in Real*Tests.swift and GenerationSmokeTests.swift). `xcodebuild test` relaunches crashed workers and still prints ✔ — trust `xcrun xctest`.
## Model weights
`$YUE2_MODELS_DIR/{YuE2-3B,YuE2-Vae,YuE2-Vae-legacy,parity}`. This repo is public: never commit a local absolute path; `.local-runs/` (gitignored) is for machine-specific scripts and logs. Weights are CC-BY-NC-4.0.
## Hardware constraints
M3 Max 96 GB: everything fits in bf16 (LM 7.3 GB + KV + VAE fp32 0.25 GB, peak ≈ 15 GB). `Memory.clearCache()` between stages. VAE stays float32.
## Architecture
Sources/YuE2Core/{Protocol,Tokenizer,Loading,Models/LM,Models/VAE,Generation,Synthesis,Pipeline,Audio,Memory}; Sources/YuE2CLI; Sources/YuE2BenchUI; Tests/YuE2Tests; parity/ (tiny fixtures); Scripts/reference (Python dumps).
## Key implementation facts (do not re-derive)
See PLAN.md §2.1–§2.8: token ids, prompt format, MoT dual paths, RoPE non-interleaved after q/k-norm, logits processor order, NAR positions (RoPE absolute, pe local), midpoint solver in bf16, VAE Oobleck topology, weight-norm merge before layout conversion, natural length 1920·T − 64.
## Performance work — read before measuring
GPU clock state moves results up to 10×: cool down before every point, A/B/B/A, control sample, judge against ±2 % spread. Profiler (`swift-mlx-profiler`) first, `print` never. Record durable findings in docs/knowledge/log.md.
## Engineering Knowledge Base
docs/knowledge/ is an OKF bundle (index.md, log.md, pitfalls/, benchmarks/, decisions/).

## iOS backend (Jalon 6)
`plan/13-ios-backend.md` (rev. 2) is the spec: MLX int4 for the AR path and orchestration, **Core AI** (iOS 27, `coreai-torch`, `xcrun coreai-build`) for the VAE and the whole NAR stack, GPU vs Neural Engine chosen by measurement, always with an MLX fallback. Never let a Core AI VAE asset specialize on CPU (open conv-transpose defect). Experiments E1–E5 run on the Mac first; device measurements gate everything (G-9). The iOS app lives in `Apps/YuE2Mobile` (Xcode project created by hand, §13.9; `Signing.xcconfig` gitignored).
