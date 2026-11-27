# Changelog

Published versions are squashed commits on `main`, merged through a pull request (the working history stays local); each version has a tag and a GitHub release.

## 1.12.2 — 2026-11-26

- **CJK lyrics timed character by character** (#7, reported by Simon Law): a line written without spaces was one word, so `timeline.json` gave it one timing. Han characters and kana are now one word each (small kana and `ー` stay with the character before them, CJK punctuation separates), Hangul words count one syllable per block. Words spelled by one shared token (the tokenizer merges `今天`) share its attention segment and are placed by the score. On a 41 s Mandopop render, 13/8/8 timed characters per line; against Whisper, ≈ 0.3 s on the verses, 0.5-1 s on the chorus.
- **CLI: a flag written explicitly wins over `--reference`, and says so** (#5, reported by Simon Law): `--quant`, `--quant-head`, `--precision`, `--vae-precision`, `--vae-core-frames` and `--release-weights-between-stages` were silently overwritten by the profile (unlike `--ode-steps` and `--nar-compute`). One rule for `generate`, `vary` and `regenerate`: the profile fills what was not written, and each departure is printed after its summary (`overridden by flags (not the reference any more): …`). `--reference` alone is unchanged. The help lists the seven profiles (`4bit-tiny` was missing).
- **Inline fields inside a bar** (ASK Q19 c): `ABCScore` reads `z2[K:Ab]z6|` (key from there on, meter from the next bar; `[L:…]` refused as before); an unreadable key such as the planner's `[K:A,B]` is kept as a token and ignored. Two app covers failed `karaoke` and the timeline on it.
- **Docs: `8bit-lean` is the iPhone profile** (ASK Q19 b), with the app's AB2 measurements (3.3-3.4 GB peak on an iPhone 15 Pro Max).
- **The seven profiles re-measured under AB2 and as drafts** (ASK Q19 a; M3 Max, 70 s song): full 39.4-60.1 s instead of 65.7-94.8 s at midpoint, draft 21.1-39.7 s, peaks unchanged. References.md, Benchmarks.md and the README carry the new tables; `yue2 references` names `8bit-lean` as the iPhone profile. `Scripts/bench-references.sh` measures all seven, `QUALITY=draft` for drafts, and waits for Time Machine and busy processes before each point.

## 1.12.1 — 2026-11-25

- **No NaN audio from the fp16 VAE on M5, a WAV export that survives it** (PR #6 by Simon Law, @ramishi; fixes #3, addresses #4): on an M5 (MLX's NAX kernels), fp16 decode tiles of 704-1280 frames emitted NaN, and the int16 export trapped and lost the whole render. The exporter now replaces non-finite samples with silence; the `fast` profiles decode with 640-frame tiles instead of 1024; both tiled decode entry points re-decode at half the core size (floor 64) when a crop is non-finite. Output is bit-identical at every tile size (checked on an M3 Max, which never showed the NaN); the clean path pays one finiteness check per tile.

## 1.12.0 — 2026-11-25

- **An unsung last line is bad luck (ASK Q18)**: the same text sings its final line on 5 seeds of 6 (the sixth hits the 300 s cap); no planning lever. Another seed is the simple answer; the tools below repair a song worth keeping.

- **Timeline: word and line attention (ASK Q18)**: `LyricTimeline.Word.attention` / `Line.attention`, the word head's mean attention while the word is sung (attention clock only). A hint: ≥ 0.35 the line was sung; low inside the song it was not; low on the last line it proves nothing (sung final lines score 0–0.02 too). Optional fields: older `timeline.json` files still load.
- **Repair or lengthen a finished song**: `regenerate(_:fromFrame:seed:score:)` conditions the regenerated part on an edited score (notes written into a silent outro, bars added after the end); kept frames stay bit-exact, the score keeps its planned tokens up to the first changed line (`SymbolicPlan.replacingScore(_:tokenizer:)`). `RegenerateLength.extended(frames:)` (within the sampling's `maxTokens`; a length the single-chunk context cannot hold is refused before the AR runs). CLI: `regenerate --score`, `--extend-seconds`, `--draft`.

## 1.11.0 — 2026-11-22

- **AB2 NAR integrator on every reference profile**: Adams–Bashforth 2 (`ODEMethod.ab2`, `--ode-method ab2`) costs steps + 1 network calls instead of 2 × steps (33 instead of 64 at 32 steps), bootstrapped by one midpoint step; edits that re-impose frames (`.keep`, regenerate) fall back to midpoint. Judged identical by ear to midpoint in bf16 and in the fp16 packed iPhone profile. NAR −49 % on an M3 Max (52.6 → 27.0 s, bf16, 70 s song); on an iPhone 15 Pro Max 3.8 s of NAR per second of music instead of 7.5–8.6, pause/kill/resume unchanged. `GenerationConfig.odeMethod` (`"midpoint"` stays the checkpoint default), `ODEMethod.evaluations(steps:startStep:)` for time estimates.
- **Draft NAR**: `YuE2ReferenceProfile.generationConfig(_:quality: .draft)` (`generate --draft`, `synthesize --draft`) solves at AB2 8 steps (`draftODESteps`, 9 calls): 8–10× less NAR time than the full render at the same peak memory, same plan, semantic tokens and VAE; judged good by ear in 4, 8 and 16 bits on a 251 s song. Upgrading a draft means solving the same `semantic.npy` again with `.full`. See docs/iOS.md, « Draft NAR ».
- `YuE2ReferenceProfile.generationConfig(_:)`: the one place that applies a profile's NAR schedule (method, pinned steps); edits under `--reference` now use it too.
- **NAR checkpoint commits through its JSON**: each save writes `nar-state-<chunk>-<step>-<id>.npy` (+ `nar-velocity-…` under AB2) then `nar-checkpoint.json` atomically, which names them; a save killed halfway leaves the previous step loadable. The checkpoint records its integrator and steps, and a resume under another schedule throws instead of mixing them. Checkpoints written by 1.10.0 (fixed `nar-state.npy`) still load.
- mlx-swift stays pinned at 0.31.6: 0.33.0 measured neutral on speed and memory (NAR latents bit-identical), not adopted.

## 1.10.0 — 2026-10-29

- **Where a plan sings (ASK Q16)**: the prompt carries no duration, so the planner alone decides the instrumental lead-in (0 to 57 bars on one 128 BPM request, median 18) and the song's length (26 to 220 bars); a `% intro` never carries the voice and `[Intro]` lyrics are sung in the next section. `ABCScore.Layout` (`ABCScore.layout`, `ABCScore.layout(of:)`, also tolerant of a plan cut by the ABC token cap) gives the sections with their bars and sung notes, `firstSungBar` / `firstSungSeconds`, `lastSungBar` / `lastSungEndSeconds`, `bars` and nominal `seconds`; every `plan.json` now carries it as `layout` (`SymbolicPlan.layout`, recomputed for plans saved before; `pipeline.py` ignores the key).
- `ABCScore.trimming(_:leadIn:tail:)`: keeps at most N bars before the voice's first group and after its last one by dropping whole `V: Vocal`/`V: Ins` groups as written (no bar re-spelled, groups with inline fields or uneven voices never dropped); render the result as an imposed score.
- CLI: `yue2 score --abc <plan dir | file.abc> [--lead-in N] [--tail N] [--out short.abc]` (no weights).
- **One short phrase under several texts (ASK Q17 b)**: documented as outside the model's contract — an identical melody calls the same text, and re-spelling its notation changes nothing. Measured levers, safest first: one sung block, a different melody per text, plus lyrics-only guidance (`--cfg-scale 1.5 --negative-style` set to the request's own style; doubles the semantic phase). No code change.

## 1.9.1 — 2026-10-10

- **Timeline order guaranteed (ASK Q14)**: a word the score would place at or before the previous one (lyrics sung past the end of a short score, no word-head anchor) now follows it off the score instead of jumping back to its note; a syllable never ends after the next one starts (a shared melisma note made them overlap on 6 of 8 songs). `lines`, `words` and `syllables` are in lyric order with increasing starts. Word-start accuracy unchanged on the eight measured songs (indie 87 → 89 %).
- `LyricTimeline.wordsOnScore` (share of words sung on the score's notes; under 0.9 the render did not follow its score) and `LyricTimeline.fit(abc:lyrics:language:)` (sung notes per lyric syllable of an imposed score, checked before generating: the model's own plans give 0.9–1.2).

## 1.9.0 — 2026-10-08

- **Song timeline (`timeline.json`)**: the start of every score bar and beat, every sung note, every lyric line, word and syllable of a generated song, with seconds and musical position (`bar`, fractional `beat`), for a player to follow without recomputing anything. The times come from the LM itself: four AR heads attend to the score bar being played (layers/heads 6/8, 10/3, 18/7, 18/6, 0.33 s ahead) and one to the next lyric word (14/10); a teacher-forced pass over layers 0-18 reads them after the semantic phase, reusing the generation's KV cache (1-2 s on an M3 Max, identical in int4). No audio analysis, no extra model. Measured on eight songs (four generated on an iPhone in int4, EN/FR, 87-132 BPM) against an MMS forced alignment of the Demucs-separated voice: word starts 71-97 % within 300 ms (median error 59-80 ms), bars 60-160 ms from a SheetSage2 transcription; it holds where placing the score on a SheetSage2 beat grid fails (voice leaving the planned melody, 10-12 % there).
  - `AttentionTimeline.timeline(model:tokenizer:semantic:)` (fresh song), `reanalyze(song:model:tokenizer:)` (songs saved earlier, CLI or app layout — migration), `LyricTimeline.position(at:)` (bar, beat, note, line, word, syllable by binary search), `write(to:)` / `load(from:)`; `YuE2Pipeline.generate(…, timeline: true)` → `SongResult.timeline`. GPU gate and cancellation per layer.
  - Songs cut to length keep only what is sung (`duration`); words sung away from the score's notes are `onScore: false`; instrumentals get bars and notes.
  - CLI: `generate --timeline`, `karaoke --song … [--skip-existing] [--sheetsage]`.
- **Instrumental songs**: `generate --instrumental` — upstream recipe (Vocal melody moved to Ins, section tags, "no vocals" style) plus a negative CFG branch on voice tags (`SongRequest.negativeStyle`, experimental, `--negative-style`) and a SheetSage2 voice check that re-renders on the next seed (6 renders in 8 clean on the first seed). `Instrumental.renderRequest(from:plannedABC:)`.
- `ABCScore`: reader of the score dialect (voices, sections, multi-bar rests, ties, inline `M:`/`K:` changes). `transcribe` writes `events.json` as its help said. `SheetSage2Events` gain `beatGrid` / `melodyOnsets`.

## 1.8.1 — 2026-10-07

- **A pack downloaded alone is loadable (ASK Q13)**: `yue2 download --model <pack>` / `ModelDownloader.download(pack:)` also fetches the files `ModelSession` reads next to the weights — `config.json`, `yue2_generation_config.json` (m-a-p/YuE2-3B) and the converted Qwen2.5 tokenizer (`tokenizer.json`, `tokenizer_config.json`) — now published under `support/` in `VincentGOURBIN/yue2-mlx-packs`, SHA-256 verified, skipped when present (`downloadPackSupport()`). Checked end to end: empty directory → `download --model int4-mixed-head` → `plan` runs; `PackSupportTests` keeps it so.

## 1.8.0 — 2026-10-04

- **Resume an edited NAR solve (ASK Q8)**: `Synthesizer.synthesize(edit:resume:)` now accepts both — the `.keep` mask ("redo from here") is rebuilt from the seed's noise and the kept latents and re-imposed from the resumed step, `.variation`'s start state is superseded by the checkpoint; bit-identical to the uninterrupted edited solve (tested for both edits). An interrupted "redo from here" no longer restarts its stage.
- **Quiet hums (ASK Q9)**: `MelodyTranscriber`'s silence gate is relative to the take (`min(0.01, 0.1 × p90 frame RMS)`; `Options.minimumRMS` forces an absolute one), so a −38 dBFS phone take keeps its notes; tracker octave errors are folded by melodic continuity (`Options.octaveContinuity`, leaps over 8 semitones from the last notes' median).
- **Follow the style's key and tempo (ASK Q12)**: `generate|plan --follow-style-key` / `SongRequest.followingStyleKey()` turns "key of D major", "A minor", "120 BPM" in the style into a forced score header (`StyleKey`); the planner otherwise treats them as hints. Opt-in, checked for coherence, not yet by ear.

## 1.7.1 — 2026-10-04

- **Fix (ASK Q11): transcription no longer crashes an iOS app sent to the background.** `SheetSage2Transcriber.checkpoint` is called before every unit of GPU work — the STFT or each of its chunks, each ConvNeXt block or chunk, each Conformer block, the projection, each decoding step — so an app waits on `YuE2GPUGate` there (nothing is submitted between two calls) and throws to cancel; a cancelled `Task` stops at the same points (`CancellationError`). It receives `SheetSage2Progress` (stage, window, encoder fraction, tokens, `overallFraction`), which gives the app an encoder progress bar. `SheetSage2Model.encode(_:checkpoint:)` and `SheetSage2Generator.generate(…checkpoint:)` expose the same hook. Pausing changes no result (tested).

## 1.7.0 — 2026-10-03

- **Prequantized SheetSage2 packs** for the 8/4-bit transcription profiles (ASK Q10 from the app): [`VincentGOURBIN/sheetsage2-mlx-q8`](https://huggingface.co/VincentGOURBIN/sheetsage2-mlx-q8) (0.9 GB) and [`sheetsage2-mlx-q4`](https://huggingface.co/VincentGOURBIN/sheetsage2-mlx-q4) (0.6 GB), CC-BY-NC-4.0 derivatives; `yue2 download --model sheetsage2-q8,sheetsage2-q4`, `YuE2Model.sheetsage2Q8/.sheetsage2Q4`. `SheetSage2Model.load(directory:profile:)` loads a pack (`weights_format: mlx-quantized`) without requantizing: 0.16-0.19 s and 0.6-0.9 GB instead of 0.33 s and 1.6-1.9 GB; transcription peaks `8bit-lean` 1.7 GB, `4bit-lean` 1.4 GB. The packs are bit-identical to the profiles' on-the-fly (GPU) quantization.
- `yue2 sheetsage2-pack --bits 8|4` writes a pack; `yue2 transcribe --load-only` measures a load; `transcribe` picks the profile's pack by default.

## 1.6.0 — 2026-10-03

- **Transcription profiles** `16bit-fast|lean`, `8bit-fast|lean`, `4bit-fast|lean` (`SheetSage2Profile`, `yue2 transcribe --profile`), as for YuE2. *Lean*: STFT and ConvNeXt by chunks (the global GRN normalization in two passes), every window encoded then the encoder released before decoding, 128 MB MLX cache. `16bit-lean`: 2.2-2.4 GB peak instead of 3.3-3.6 GB for +5 % time, same scores. The quantized profiles (encoder only) are measured, not adopted: they change orchestral scores. Table in [docs/References.md](docs/References.md#transcription-profiles-yue2-transcribe---profile-sheetsage2profile).
- **Merged fp16 SheetSage2 pack** on Hugging Face, [`VincentGOURBIN/sheetsage2-mlx-fp16`](https://huggingface.co/VincentGOURBIN/sheetsage2-mlx-fp16) (CC-BY-NC-4.0 derivative, 1.4 GB, SHA-256 verified): `yue2 download --model sheetsage2-fp16`, default `transcribe` model when present; loads in 0.2 s / 1.4 GB, same tokens as the upstream release under the 16-bit profiles. `ModelDownloader` verifies a model's `model.safetensors.sha256` sidecar when it has one.
- Correction: `m-a-p/SheetSage2` and `m-a-p/MERT-v2-FullSong` are public, not gated (1.5.0 said otherwise); no token is needed.
- The Conformer's kernel-1 pointwise convolutions are `Linear` layers (same product, quantizable).

## 1.5.0 — 2026-10-03

- **SheetSage2 ported to MLX Swift**: new library product `SheetSage2Core` and `yue2 transcribe --audio song.m4a --out run/score` (recording → ABC score, melody only by default, `--chords` for the full lead sheet). No Python any more for covers: `yue2 download --model sheetsage2` fetches the gated upstream release and its MERT-v2-FullSong parent at pinned revisions, the LoRA adapters are merged at load.
- Parity with upstream: tokens and ABC byte-identical in fp32 on two excerpts (orchestral, pop) and on a 500 s song in four overlapping windows; `fp16` (default) identical on the excerpts, where `bf16` changes the score.
- M3 Max, fp16: 30 s transcribed in 3.0 s, 500 s in 21 s; 1.6 GB loaded, 3.3-3.5 GB peak (upstream PyTorch MPS: 7.9 s / 17 GB for 30 s, collapses on long songs).
- `ModelDownloader` fetches a pinned revision per model (`YuE2Model.revision`).

## 1.4.0 — 2026-10-02

- **Covers from a recording, the upstream way**: SheetSage2 (Python, runs on Apple silicon through MPS) transcribes a mixed song into an ABC score; the song is generated on it with `cot: melody`. Recipe, Mac workarounds and measurements in [docs/CLI.md](docs/CLI.md#covers-from-a-recording-transcribe-sheetsage2--abc--cot-melody).
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
