# The six reference configurations

`yue2 references` lists them, `yue2 generate --reference <id>` applies one, `YuE2ReferenceProfile.all` exposes them to an app. Each one pins **every** setting that matters: weight pack, compute precision, how the NAR is computed, VAE decoder, weight residency, memory profile, ODE steps. Nothing is left to chance between two machines: a reference produces the same song for the same seed, and comparable time on comparable hardware.

Source: `Sources/YuE2Core/Configuration/ReferenceProfiles.swift`. Measurements: [Benchmarks.md](Benchmarks.md). Audio: the [v1.0.1 release](https://github.com/VincentGourbin/yue2-mlx-swift/releases/tag/v1.0.1) carries the same 70 s techno-funk song rendered by each configuration (`swift-engine_technofunk_<id>.m4a`).

## The table

70.0 s of audio (1 750 semantic tokens), 32 ODE steps, MacBook Pro M3 Max 96 GB, idle GPU, one pass after a 120 s cool-down.

| Id | Pack | Compute | Time | Peak process | Who it is for |
|---|---|---|---|---|---|
| `4bit-fast` | `int4-mixed-head` (2.5 GB) | everything resident, NAR dequantized to bf16, VAE fp16 tile 1024, Mac caches | **65.7 s** | 8.9 GB | Mac: the only profile faster than real time at 32 steps |
| `4bit-lean` | `int4-mixed-head` | stage-scoped residency, packed NAR, fp16, VAE fp16 tile 256, mobile limits | 77.7 s | **3.4 GB** | iPhone 15 Pro Max and any 8 GB Mac |
| `8bit-fast` | `qint8-all-head` (3.7 GB) | everything resident, NAR dequantized to bf16, VAE fp16/1024 | 76.5 s | 9.9 GB | Mac, when the AR path must stay 8-bit |
| `8bit-lean` | `qint8-all-head` | residency, packed NAR, fp16, VAE fp16/256, mobile limits | 79.3 s | 4.4 GB | iPhone 8 GB with headroom, 8-16 GB Macs |
| `16bit-fast` | bf16 (7.3 GB) | everything resident, VAE fp16/1024 | 84.9 s | 12.0 GB | quality reference |
| `16bit-lean` | bf16 | stage-scoped residency, VAE fp16/256, Mac caches | 84.9 s | 6.9 GB | 16-24 GB Macs |

Quick read: 4-bit is fastest because the autoregressive phases (score, then semantic tokens) are bound by memory bandwidth and dispatch; the NAR, 8-bit in both quantized packs, costs ≈ 50 s whatever the profile and is three quarters of the time. The lean profiles divide memory by two to three for no time cost (8-bit) to 18 % (4-bit, due to the packed NAR rather than to residency).

## What each setting does

| Setting | Values | Effect |
|---|---|---|
| Pack (`quant`, `quantizeHead`) | `none` bf16 · `qint8-all` + head · `int4-mixed` + head | Weights on disk and in memory. `int4-mixed` = AR path, embeddings and head in 4 bits, NAR path in 8 bits (a 4-bit NAR fails parity: the error compounds over 64 evaluations). `+head` = quantized `lm_head`, computed by `quantizedMatmul` on the rows the phase allows. |
| `precision` | `bf16` · `fp16` | Dtype of floating parameters and activations. fp16 for the iPhone (GPU and Neural Engine); bf16 on Mac. Both validated by listening. |
| `narCompute` | `packed` · `dequantized` | `packed` runs the NAR projections through `quantizedMatmul` (minimal memory). `dequantized` expands them once to bf16 before the solve: +1.4 GB for an 8-bit NAR, ≈ 10 % less NAR time on Mac (the quantized matmul is slower than a bf16 GEMM on large activations). Numerical gap 1.3 % (weights rounded to bf16), inside the parity budget. |
| `vaePrecision`, `vaeCoreFrames` | fp16 · fp32; 256 · 1024 | The Oobleck decoder in fp16 is inaudible against fp32 (SNR 53-57 dB) and faster. The tile sets the transient decode peak: 1024 ≈ 2.9 GB active in fp16, 256 ≈ 2.0 GB. |
| `releaseWeightsBetweenStages` | bool | Stage-scoped residency: AR path alone at load, AR → NAR swap once the prefix cache is ready, whole LM released before the VAE. The memory budget becomes the largest stage instead of the sum. Zero time cost (measured A/B/B/A). |
| `memoryProfile` | `mac` · `mobile` | MLX caches per stage (Mac: 2 / 4 / 1 GB) or limits sized from available memory (mobile: cache ≈ 1 GB, GC threshold = available − 1.25 GB). Mobile limits thrash a bf16 working set: `16bit-lean` stays on `mac`. |
| `odeSteps` | nil = 32 | Midpoint solver steps (2 evaluations each). 24 steps bring 8-bit under 70 s, 16 steps bring everything under 45 s; listening validation in progress. |

## Choosing

- **Mac, 32 GB and up**: `4bit-fast`. If you want the AR path in 8 bits: `8bit-fast`. For a quality reference: `16bit-fast`.
- **Mac, 16-24 GB**: `16bit-lean` (6.9 GB), or `8bit-lean` / `4bit-lean` when other apps are running.
- **8 GB Mac, iPhone**: `4bit-lean` (3.4 GB). `8bit-lean` also fits (4.4 GB) on an iPhone 15 Pro Max with the increased-memory entitlement.
- **Even faster, memory no object**: outside the six, `--quant int4 --quant-head` (4-bit AR path, bf16 NAR, `int4-head` pack, 3.5 GB) gives 66-69 s; it is the candidate for a seventh profile.

## Adding or changing a profile

1. Add the entry to `YuE2ReferenceProfile.all` (every field is an existing knob, nothing new to wire).
2. Measure with `Scripts/bench-references.sh <request.json> 1750 2` (two passes, protocol in [Benchmarks.md](Benchmarks.md)).
3. Listen to the WAV against `16bit-fast` on the same seed.
4. Add the row to `BENCHMARKS.md` and the decision to `docs/knowledge/decisions/reference-profiles.md`.

## Command-line equivalents

The profiles are shortcuts; every setting exists as an option:

```bash
# 4bit-fast
yue2 generate --request song.json --quant int4-mixed --quant-head --nar-compute dequantized --vae-precision fp16 --out run/
# 4bit-lean (YUE2_MEMORY_PROFILE=mobile turns on residency and the 256-frame tile by default)
YUE2_MEMORY_PROFILE=mobile yue2 generate --request song.json --quant int4-mixed --quant-head --precision fp16 --vae-precision fp16 --out run/
# 16bit-lean
yue2 generate --request song.json --release-weights-between-stages --vae-precision fp16 --vae-core-frames 256 --out run/
```
