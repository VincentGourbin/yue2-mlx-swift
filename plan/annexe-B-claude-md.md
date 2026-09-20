## Annexe B — `CLAUDE.md` à créer en 1.0 (contenu de départ, à maintenir)

```
# CLAUDE.md
## Project Overview
Swift MLX port of YuE2 (m-a-p/YuE2-3B + YuE2-Vae): lyrics + style → editable ABC score → semantic tokens → acoustic latents (flow matching) → 48 kHz stereo song. Text-only input; MERT2/SheetSage2 are not used at inference.
**`plan/` is the porting spec** (one file per section; `PLAN.md` is only the index) — read `plan/02.*` (architecture facts) and `plan/09-pieges.md` (pitfall checklist) before touching model code. Upstream reference: https://github.com/multimodal-art-projection/YuE (`src/yue2/*.py`).
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
See `plan/02.1-vocabulaire.md` … `plan/02.8-dtypes.md`: token ids, prompt format, MoT dual paths, RoPE non-interleaved after q/k-norm, logits processor order, NAR positions (RoPE absolute, pe local), midpoint solver in bf16, VAE Oobleck topology, weight-norm merge before layout conversion, natural length 1920·T − 64.
## Performance work — read before measuring
GPU clock state moves results up to 10×: cool down before every point, A/B/B/A, control sample, judge against ±2 % spread. Profiler (`swift-mlx-profiler`) first, `print` never. Record durable findings in docs/knowledge/log.md.
## Engineering Knowledge Base
docs/knowledge/ is an OKF bundle (index.md, log.md, pitfalls/, benchmarks/, decisions/).
```
