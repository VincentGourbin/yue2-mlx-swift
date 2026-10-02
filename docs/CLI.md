# `yue2` command line

Binary produced by `xcodebuild -scheme yue2 -configuration Release -destination 'platform=macOS' -derivedDataPath .xcodebuild build` (never `swift build` for a binary: MLX's `default.metallib` would not be found). `yue2 <command> --help` always prints the full option list.

## Commands

| Command | Role |
|---|---|
| `info` | version and checkpoint status |
| `download` | downloads the LM and/or VAE from Hugging Face |
| `generate` | full song: ABC score → semantic tokens → NAR → VAE |
| `plan` | the ABC score alone (stage 1, resumable) |
| `semantic` | semantic tokens from a saved plan (stage 2) |
| `synthesize` | NAR + VAE from saved plan and tokens (stages 3-4), with per-step checkpoints |
| `decode` | latents `.npy` → WAV (VAE alone, MLX or Core AI) |
| `encode` | audio → latents (VAE encoder), optional round trip |
| `references` | lists the reference configurations |
| `melody` | hummed or sung recording → ABC score the model takes as its plan |
| `vary` | variation of a saved song (same score and tokens, SDEdit on its latents) |
| `regenerate` | keeps a saved song up to a point in time, regenerates the rest |
| `parity vae|lm|nar` | numerical parity against the PyTorch dumps |
| `profile plan|semantic` | profile of one AR phase (TTFT, tok/s, memory, trace) |
| `bench-coreai-nar` | NAR MLX against Core AI GPU/ANE, time and footprint |
| `remix-experimental` | SDEdit rewrite of an encoded latent (unvalidated) |

Common environment variables: `YUE2_MODELS_DIR` (checkpoints), `YUE2_MEMORY_PROFILE=mac|mobile`, `YUE2_CACHE_LIMIT_MB`, `YUE2_MEMORY_LIMIT_MB`, `YUE2_NAR_COMPUTE=packed|dequantized`, `YUE2_EVAL_PER_LAYER=1|0`, `YUE2_COMPILED_DECODE=1`, `YUE2_DEBUG=1` (log on stderr), `YUE2_DISABLE_COMPILE=1` (uncompiled SnakeBeta), `YUE2_DISABLE_COREAI=1`.

## `generate`

```bash
yue2 generate --request song.json --reference 4bit-fast --out run/ [--profile]
```

The request file: `style`, `lyrics` (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]` tags), `cot` (`full`, `melody`, `off`), `seed`, `id`, optional `abc` (imposed score, the planner is skipped), optional `abc_prefix` (forced beginning of the score, the planner writes the rest in the same key and tempo), optional `cfg_scale`.

| Option | Default | Effect |
|---|---|---|
| `--reference <id>` | — | applies one of the six configurations ([References.md](References.md)); the options below still work to depart from it |
| `--quant` | `none` | `none`, `qint8`, `int4` (AR path only), `qint8-all`, `int4-all`, `int4-mixed` (both paths) |
| `--quant-head` | off | quantized `lm_head`, computed by `quantizedMatmul` on the phase's rows |
| `--precision` | `bf16` | `fp16` for the iPhone |
| `--ode-steps` | 32 | midpoint solver steps |
| `--vae-precision`, `--vae-core-frames` | `fp32`, 1024 (256 on the mobile profile) | fp16 decoder is inaudible and faster; the tile sets the transient peak |
| `--vae-backend` | `mlx` | `coreai-gpu` (54.7 dB parity, macOS/iOS 27), `coreai-ane` (dead end today); MLX fallback is logged |
| `--release-weights-between-stages` | off (on with the mobile profile) | stage-scoped residency |
| `--nar-compute` | `packed` | `dequantized`: NAR in bf16 for the solve, +1.4 GB, ≈ 10 % faster on Mac |
| `--compiled-decode` | off | compiled decode step (measured without gain, kept as an option) |
| `--abc-max-tokens`, `--abc-min-tokens`, `--semantic-max-tokens`, `--semantic-min-tokens` | checkpoint config | phase lengths; 25 semantic tokens = 1 s of audio (1 750 = 70 s) |
| `--temperature`, `--top-p`, `--top-k` | checkpoint config | sampling for both phases |
| `--style`, `--lyrics`, `--cot`, `--seed`, `--abc-file`, `--cfg-scale`, `--id` | — | override the request |
| `--abc-prefix-file` | — | beginning of a score (header with key and tempo, first sections) the planner continues from; overrides the request's `abc_prefix` |
| `--vae` | `standard` | `legacy` for the older VAE |
| `--format` | `int16` | `float32` |
| `--profile` | off | per-phase report, TTS metrics, `trace.json` in `--out` |

Artifacts in `--out`: `score.abc`, `abc_tokens.npy`, `prefix.npy`, `semantic.npy`, `latent.npy`, `audio.wav`, `plan.json`, `request.json`, `config.json`, `result.json` (times and hashes), `trace.json` with `--profile`.

## Stage by stage

```bash
yue2 plan       --request song.json --out run/plan
yue2 semantic   --plan run/plan --out run/song
yue2 synthesize --plan run/plan --semantic run/song --out run/song --checkpoint-dir run/song [--resume]
yue2 decode     --latents run/song/latent.npy --out run/song/audio.wav --precision fp16 --core-frames 256
```

`synthesize --checkpoint-dir` writes `nar-checkpoint.json` and `nar-state.npy` (192 KB) after every ODE step and clears them at the end; `--resume` restarts from the saved step, bit-exact with an uninterrupted solve (same plan, tokens, seed and steps). This is what makes losing the GPU (iOS in the background, a crash) harmless: one step lost, not the stage.

`decode` prints a line `DECODE backend=… tiles=… padded_tiles=… seconds=… footprint_before_load_mb=… footprint_after_load_mb=… footprint_decode_peak_mb=… mlx_decode_peak_mb=…`.

## `download`, `info`

```bash
yue2 download --models-dir $YUE2_MODELS_DIR            # lm,vae ; --model lm,vae,vae-legacy
yue2 download --model int4-mixed-head                  # a published prequantized pack (2.5 GB)
yue2 download --model packs                            # all three packs (10.4 GB)
yue2 info
```

Weights `m-a-p/YuE2-3B` and `m-a-p/YuE2-Vae` (CC-BY-NC-4.0), tokenizer converted from `Qwen/Qwen2.5-0.5B`. The reference packs (`int4-mixed-head`, `qint8-all-head`, `int4-head`) come from `VincentGOURBIN/yue2-mlx-packs`, verified by SHA-256, and land under `$YUE2_MODELS_DIR/YuE2-3B/mlx-prequantized/<pack>/`; any other preset is quantized locally on first use — see [Weights.md](Weights.md).

## `references`

```
$ yue2 references
4bit-fast   quant=int4-mixed+head precision=bf16 nar=dequantized compiled=false vae=fp16/1024 residency=all memory=mac ode=32
            int4-mixed-head pack (2.5 GB), everything resident, NAR dequantized to bf16 for the solve
…
```

## `parity`, `profile`, `bench-coreai-nar`

```bash
yue2 parity vae [--backend coreai-gpu --core-frames 256] [--precision fp16]
yue2 parity lm
yue2 parity nar [--backend coreai-gpu] [--sweep-family …]
yue2 profile semantic --plan run/plan --out prof/ --quant int4-mixed --quant-head
yue2 bench-coreai-nar --frames 256,1024 --backends mlx,coreai-gpu --quant int4-mixed [--iterations 6]
```

Real fixtures come from `Scripts/reference/real_fixtures.py {vae|lm|nar|song}` (Python environment from `Scripts/setup-reference-env.sh`). Verdicts: `PARITY OK …` / `PARITY FAILED …` on the last line.

## Audio in: `melody`, `vary`, `regenerate`

The model has no audio understanding at inference: the semantic audio tokenizer it was trained with is not released, so a recording cannot be turned into its tokens, and injecting a recording's VAE latents into the solve only pastes it in (tried and dropped). What it does accept is its own artifacts: an ABC score as the plan, semantic tokens as the AR past, latents as the start state of the acoustic solve. The three commands feed those entries from a recording or from a previous run.

```bash
# 1. hum or sing a melody (mono, any format AVFoundation reads), 100 BPM by default
yue2 melody --audio hum.m4a --bpm 110 --out run/melody        # writes melody.abc + melody.json, prints the score
# 2. the song follows that melody: the score is the plan, the AR only writes the tokens
yue2 generate --request song.json --abc-file run/melody/melody.abc --reference 4bit-fast --out run/song

# a variation of a saved song: same lyrics, same score, same tokens, new texture
yue2 vary --song run/song --strength 0.4 --seed 2 --reference 4bit-fast --out run/song-v2
# keep the first 20 s exactly, regenerate everything after (new tokens, acoustic solve against the kept part)
yue2 regenerate --song run/song --from-seconds 20 --seed 3 --reference 4bit-fast --out run/song-alt-ending
```

`melody` runs on the CPU (YIN pitch tracking, median smoothing, quantization on a sixteenth grid at the given tempo, Krumhansl key estimate, one diatonic chord per bar) and writes the score in the model's dialect: `L:1/16`, `V: Vocal` carrying the melody, `V: Ins` resting (`Z<n>|`), one `% verse` section. The `melody.json` next to it lists the MIDI notes, their start and length in sixteenths, the key, the voiced ratio (below 0.3 the command warns: the recording carried little pitch). A whistle sits two octaves above a voice and up to 2 kHz: pass `--max-hz 2500`; the melody is brought into the singing range by whole octaves automatically (`--octave-shift` to choose). Notes shorter than a sixteenth that sit within two semitones of a neighbour are read as slides into it, not as notes. Edit the `.abc` by hand before `generate` if a note is wrong; the model's planner is skipped entirely when a score is imposed.

`vary --strength` is the SDEdit ratio: 0 returns the source latents unchanged, 1 is a fresh synthesis; 0.3-0.5 keeps the arrangement and changes the texture, 0.6-0.8 drifts further. The solve runs `round(strength × steps)` steps instead of 32, so a 0.4 variation costs 40 % of a full NAR. `regenerate --from-seconds` is rounded to a 40 ms frame; the kept frames are re-imposed after every ODE step (RePaint-style mask) and returned bit-exact, the seam is continuous by construction. By default the regenerated song keeps the source's length (the AR is cut there if it has not ended the song); `--free-length` lets the AR decide, and a song re-conditioned on its first half readily grows to two or three times its length. Both read the `generate --out` directory (`plan.json`, `semantic.npy`, `latent.npy`, `config.json`) and write a full one. Single-chunk songs only.

## Covers from a recording: SheetSage2 → ABC → `cot melody`

For a mixed recording (a real song, not a hummed line), the upstream route goes through the score: [SheetSage2](https://huggingface.co/m-a-p/SheetSage2) (gated, CC-BY-NC-4.0, PyTorch, built on MERT-v2-FullSong) transcribes the audio into an ABC score in the model's dialect, and the song is generated on that score with `cot: melody` (the melody is held, the accompaniment and the style come from the prompt). SheetSage2 is not ported; it runs in its own Python environment, on Apple silicon through MPS.

```bash
# once: Python 3.10/3.11 environment, CPU/MPS torch (the upstream README targets CUDA)
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python "huggingface-hub==0.36.0"
.venv/bin/huggingface-cli download m-a-p/SheetSage2 --local-dir SheetSage2
uv pip install --python .venv/bin/python -r SheetSage2/requirements.txt
# transformers 4.45 copies only part of a local repo's modules: copy them all into its module cache
M=~/.cache/huggingface/modules/transformers_modules/SheetSage2; mkdir -p "$M" && cp SheetSage2/*.py "$M"/

# transcribe (melody only: vocal and instrumental lines, no chord symbols) → cover-score/score.abc
(cd SheetSage2 && ../.venv/bin/python infer.py ../excerpt.wav --output ../cover-score --device mps --melody-only)

# a song on that score
yue2 generate --style "…, A minor, 131 BPM" --lyrics "[Chorus]…" --cot melody --abc-file cover-score/score.abc --out run/cover
# or that score as the opening of a longer song: the planner continues it (verses, repeats) in the same key and tempo
yue2 generate --style "…" --lyrics "[Chorus]…[Verse]…[Chorus]…" --cot melody --abc-prefix-file cover-score/score.abc --out run/around
```

Measured on a 13 s excerpt of a mixed song (M3 Max): transcription in 43 s including the model load, 16.5 GB peak footprint; a clean eight-bar hook with key, tempo and section. Put the key and tempo SheetSage2 found (`K:`, `Q:`) in the style prompt. With `--abc-file` alone the AR may sing past the end of a short score; with `--abc-prefix-file` the planner tends to repeat the given section before writing new ones. Read the score before generating: it is the edit point (fix a note, change the tempo, add sections by hand).

## `encode`, `remix-experimental`

```bash
yue2 encode --audio input.mp3 --out run/enc --roundtrip
yue2 remix-experimental --request song.json --source-latent run/enc/latent.npy --frames 1000 --noise-fraction 0.7 --out run/remix
```

The encoder is validated numerically (parity, listened round trip); the remix is an SDEdit experiment outside the plan, unvalidated.
