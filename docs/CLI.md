# `yue2` command line

Binary produced by `xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build` (never `swift build` for a binary: MLX's `default.metallib` would not be found). `yue2 <command> --help` always prints the full option list.

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
| `references` | lists the six reference configurations |
| `parity vae|lm|nar` | numerical parity against the PyTorch dumps |
| `profile plan|semantic` | profile of one AR phase (TTFT, tok/s, memory, trace) |
| `bench-coreai-nar` | NAR MLX against Core AI GPU/ANE, time and footprint |
| `remix-experimental` | SDEdit rewrite of an encoded latent (unvalidated) |

Common environment variables: `YUE2_MODELS_DIR` (checkpoints), `YUE2_MEMORY_PROFILE=mac|mobile`, `YUE2_CACHE_LIMIT_MB`, `YUE2_MEMORY_LIMIT_MB`, `YUE2_NAR_COMPUTE=packed|dequantized`, `YUE2_EVAL_PER_LAYER=1|0`, `YUE2_COMPILED_DECODE=1`, `YUE2_DEBUG=1` (log on stderr), `YUE2_DISABLE_COMPILE=1` (uncompiled SnakeBeta), `YUE2_DISABLE_COREAI=1`.

## `generate`

```bash
yue2 generate --request song.json --reference 4bit-fast --out run/ [--profile]
```

The request file: `style`, `lyrics` (`[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]` tags), `cot` (`full`, `melody`, `off`), `seed`, `id`, optional `abc` (imposed score), optional `cfg_scale`.

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

## `encode`, `remix-experimental`

```bash
yue2 encode --audio input.mp3 --out run/enc --roundtrip
yue2 remix-experimental --request song.json --source-latent run/enc/latent.npy --frames 1000 --noise-fraction 0.7 --out run/remix
```

The encoder is validated numerically (parity, listened round trip); the remix is an SDEdit experiment outside the plan, unvalidated.
