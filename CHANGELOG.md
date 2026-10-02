# Changelog

Published versions are squashed commits on `main` (the working history stays local); each version has a tag and a GitHub release.

## 1.4.0 — 2026-10-02

- **Covers from a recording, the upstream way**: SheetSage2 (Python, runs on Apple silicon through MPS) transcribes a mixed song into an ABC score; the song is generated on it with `cot: melody`. Recipe, Mac workarounds and measurements in [docs/CLI.md](docs/CLI.md#covers-from-a-recording-sheetsage2--abc--cot-melody).
- **Forced beginning of score**: `SongRequest.abcPrefix` (`abc_prefix` in the request JSON) / `generate --abc-prefix-file`; the planner continues from the given header and sections in the same key and tempo — a transcribed chorus becomes the opening of a longer song.
- **Whistled melodies**: `yue2 melody --min-hz/--max-hz/--octave-shift`, automatic octave shift into the singing range, short glides absorbed into the note they lead to.
- Tried and dropped: keeping a real recording's VAE latents inside a generated song (inpainting, and anchoring it on the first ODE steps only). The model never hears the excerpt (the semantic audio tokenizer is not released), so it is a paste-in at best; not shipped.
- Build commands now pass `-destination 'platform=macOS'` (required by the current `xcodebuild` for a Swift package).

## 1.3.0 — 2026-10-01

- **Audio in, through the model's own entries** (issue #1 closed as a feature): `yue2 melody` transcribes a hummed or sung recording into an ABC score in the model's dialect (`MelodyTranscriber`, CPU: YIN pitch tracking, sixteenth-grid quantization at the given BPM, key and diatonic chords); `generate --abc-file` then follows that melody.
- **Song edits** on a `generate --out` directory (`SongResult.load(from:)`): `yue2 vary` / `YuE2Pipeline.vary(_:strength:seed:)` (SDEdit variation, same score and tokens, `round(strength × steps)` ODE steps) and `yue2 regenerate` / `YuE2Pipeline.regenerate(_:fromFrame:seed:)` (kept part bit-exact, tokens after it re-sampled with the kept ones as the AR's past, RePaint-style mask in the solve); the source's length is kept by default (`RegenerateLength`, `--free-length` lets the AR decide — re-conditioned on its first half it readily grows the song ×2.5).
- Engine: `NAREdit` (`.variation`, `.keep`) on `Synthesizer.synthesize`, `keep:` mask in `CachedNAR.solve`, `continuation:` on `SemanticGenerator.generateSemantic`.

## 1.2.0 — 2026-09-27

- **`4bit-tiny` reference profile** for the most constrained devices: `4bit-lean` plus a 64-frame VAE tile and a fixed 256 MB cache / 3 GB threshold (`YuE2ReferenceProfile.mobileLimitsMB`, `YuE2MemoryManager.mobileLimitsOverrideMB`): 2.6 GB peak instead of 3.4, +29 % time, same audio. Seven profiles.
- 2-bit and 3-bit AR presets (`int2-mixed`, `int3-mixed`) measured and documented, not adopted: the 8-bit NAR dominates the packs (−0.2 / −0.4 GB only), 2-bit is no longer music by ear, 3-bit at the limit; the three renders are attached to the release (`limit_{4,3,2}bit_*.m4a`).

## 1.1.0 — 2026-09-27

- **Published prequantized packs**: `int4-mixed-head` (2.5 GB), `qint8-all-head` (3.5 GB) and `int4-head` (4.4 GB) on Hugging Face (`VincentGOURBIN/yue2-mlx-packs`, CC-BY-NC-4.0 derivatives). `yue2 download --model <pack>|packs`, `ModelDownloader.download(pack:)` with SHA-256 verification, `YuE2Pack`, `YuE2ReferenceProfile.pack`; `yue2 references` prints each profile's download line. The first-use quantization (7.3 GB of bf16, minutes) is no longer needed for the six references.
- Pack sizes in the docs corrected to the measured ones.

## 1.0.1 — 2026-09-27

- Documentation and profile summaries in English; ten audio examples (AAC) attached to the release.

## 1.0.0 — 2026-09-27

First "turnkey" version: six reference configurations, measured and reproducible, with full documentation and audio examples attached to the release.

- **Six reference profiles** (`YuE2ReferenceProfile`, `yue2 references`, `generate --reference <id>`, `Scripts/bench-references.sh`): 4 / 8 / 16 bits × fast / lean. On the 70 s song (M3 Max): 65.7 s / 8.9 GB to 84.9 s / 12 GB for the fast ones, 77.7 s / 3.4 GB to 84.9 s / 6.9 GB for the lean ones.
- **Preallocated KV cache** (`KVCache`, in-place writes by steps of 512): no more whole-cache copy per token; AR phases −16 % in 4-bit, bit-exact.
- **Dequantized NAR for the solve** (`--nar-compute dequantized`, `YuE2ExecutionPolicy.narCompute`): the 8-bit matmul is slower than a bf16 GEMM on large activations; NAR −10 % for +1.4 GB.
- **Precision cast limited to resident branches** (`applyPrecision(_:where:)`): the fp16 cast used to materialize the parked NAR branch, +1.44 GB, the iPhone plan-phase peak.
- **Adaptive mobile memory limits** (`YuE2MemoryManager.mobileLimitsMB()`, `YUE2_CACHE_LIMIT_MB`, `YUE2_MEMORY_LIMIT_MB`): the fixed 256 MB / 3 GB cost +32 % (4-bit) to +73 % (bf16).
- **Compiled decode step** (`CompiledDecodeStep`, `--compiled-decode`): 8/8 parity, no measured gain; kept as a disabled option.
- **Benchmarks**: 16/8/4-bit × plain MLX / optimized on 70 s (A/B/B/A, idle GPU), ODE steps 32/24/20/16, CPU vs GPU per phase; measurement discipline (competing processes, pre-exported pack, `.noindex` outputs).
- **Documentation**: `docs/References.md`, `docs/Benchmarks.md`, `docs/CLI.md`, `docs/API.md`, `docs/iOS.md`, `docs/Weights.md`, rewritten README, this changelog.

## 0.2.1 — 2026-09-26

- Losing the GPU in the background on iOS (Q7): `NARCheckpoint` (bit-exact per-step resume), `YuE2GPUGate` (awaited before every token, NAR layer, VAE tile), `YuE2ExecutionPolicy.evalPerLayer`.
- README: status as of 26 September.

## 0.2.0 — 2026-09-24

- Quantized restricted head (`RestrictedHead` through `quantizedMatmul`): plan-phase peak −283 MB, ABC throughput 57.7 → 131 tok/s.
- First iPhone 15 Pro Max measurements; `int4-mixed-head` pack; G-9 / G-10; NAR Core AI GPU measured on the device (5-7× slower).

## 0.1.0-beta — 2026-09-21

- Stage-scoped weight residency; exact-shape windows for the Core AI VAE (parity 33.8 → 54.7 dB); `FootprintSampler`; the pipeline's peak located in the VAE decode.

## 0.1 — 2026-09-20

- Initial publication: full pipeline (ABC score, semantic tokens, flow matching, Oobleck VAE), parity against the PyTorch implementation, VAE encoder, 8-bit quantization of the AR path, bench GUI.
