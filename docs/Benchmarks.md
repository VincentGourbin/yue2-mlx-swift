# Benchmarks: method, results, reproduction

This is the reading guide. Raw rows live in [`BENCHMARKS.md`](../BENCHMARKS.md) (one line per run, never edited), dated analyses in [`docs/knowledge/benchmarks/`](knowledge/benchmarks/) and the journal in [`docs/knowledge/log.md`](knowledge/log.md) (both in French).

## 1. What is measured, and with what

- **Time** per phase (ABC score, semantic tokens, NAR, VAE) and total excluding load, as printed by `yue2 generate --profile` (`generated … in X s (abc / semantic / nar / vae)`).
- **Memory**: peak process footprint (`phys_footprint`, the number iOS jetsam judges) and peak MLX active memory, per phase, from the Chrome trace (`trace.json`) written by `swift-mlx-profiler`. Unit commands (`decode`, `bench-coreai-nar`) print the same figures on their `DECODE …` / `BENCH …` lines (`FootprintSampler`, 5 ms).
- **CPU / GPU utilization** per phase, in the same trace (`Utilization` counter).
- Never a hand-timed `print`.

## 2. The protocol (CLAUDE.md, "Performance work")

GPU clock state moves a result by up to 10×. A timing is only valid if:

1. **Idle GPU**: `pgrep -fl "serve|train|lora"` before every point. A third-party LoRA training tripled a whole day of measurements (26 September) without any single number looking absurd on its own.
2. **Cool-down**: 120 s before each point (60 s for a single phase).
3. **A/B/B/A with a control**: two runs of each configuration, order A B B A; a difference is only read if it exceeds the spread of the two A runs (≤ 3 % on an idle GPU).
4. **Prequantized pack already exported**: the first use of a quantized preset loads 7.3 GB of bf16, quantizes and exports (`mlx-prequantized/<preset>[-head]/`) — a run to discard.
5. **Outputs under `.local-runs/bench.noindex/`**: Spotlight indexes freshly written WAV and npy files (`corespotlightd`: 207 minutes of CPU on the reference machine) and steals the core the AR phases depend on; the `.noindex` suffix stops it. `mediaanalysisd`, Time Machine (`backupd`) and WindowServer remain occasional disturbers: the script logs the busiest process before every point; an out-of-band run is redone, never interpreted.

`Scripts/bench-references.sh` applies all of this to the six references; `Scripts/bench-song.sh` does the A/B/B/A on a short song.

## 3. Reference results (M3 Max 96 GB, macOS 27, mlx-swift 0.31.6)

### The six configurations, 70 s song, 32 steps

| Id | Total | abc / semantic / NAR / VAE | Peak process | Peak per phase (plan / sem. / NAR / VAE) |
|---|---|---|---|---|
| `4bit-fast` | **65.7 s** | 2.7 / 10.7 / 51.1 / 1.1 | 8.9 GB | 4.2 / 5.3 / 8.5 / 8.9 |
| `4bit-lean` | 77.7 s | 4.2 / 10.8 / 61.5 / 1.2 | **3.4 GB** | 2.8 / 3.4 / 2.8 / 3.4 |
| `4bit-tiny` | 94.8 s | 6.0 / 16.8 / 70.0 / 2.0 | **2.6 GB** | 2.2 / 2.6 / 2.4 / 1.9 |
| `8bit-fast` | 76.5 s | 5.8 / 15.5 / 54.0 / 1.1 | 9.9 GB | 5.3 / 6.4 / 9.6 / 9.9 |
| `8bit-lean` | 79.3 s | 5.6 / 15.3 / 57.1 / 1.1 | 4.4 GB | 4.0 / 4.4 / 3.8 / 3.4 |
| `16bit-fast` | 84.9 s | 8.7 / 23.8 / 51.3 / 1.1 | 12.0 GB | 8.8 / 9.7 / 10.3 / 12.0 |
| `16bit-lean` | 84.9 s | 8.4 / 23.1 / 52.2 / 1.2 | 6.9 GB | 6.0 / 6.9 / 6.9 / 3.5 |

One pass (traces `.local-runs/bench.noindex/ref-20260926-2300-*`). The A/B/B/A controls of the same evening on the same packs hold within ±3 %.

### Bits × mode, A/B/B/A on an idle GPU (same song, 32 steps)

| Pack | Plain MLX (2 runs) | Optimized, mobile (2 runs) | Peak plain → optimized |
|---|---|---|---|
| bf16 | 89.4 / 87.2 s | 98.2 / 134.7 s | 14.8 → 5.2 GB |
| 8-bit throughout | 83.5 / 86.2 s | 84.9 / 85.6 s | 11.5 → 4.3 GB |
| 4-bit `int4-mixed-head` | 76.6 / 79.0 s | 76.6 / 109.5 s | 10.4 → 3.3-3.5 GB |

"Optimized" = stage-scoped residency + VAE fp16/256 + adaptive mobile limits. In 8-bit and 4-bit, residency costs nothing; the two out-of-band runs (bf16 134.7 s: cache reclaim at the threshold, 4.3 GB working set; 4-bit 109.5 s: transient GPU state, trace without anomaly, next control back at 79 s) are documented in `docs/knowledge/benchmarks/mac-70s-quant-comparison-2026-09-26.md`.

### ODE steps (÷ 2 lever on the NAR, quality still to validate)

| Configuration | 32 steps | 24 steps | 20 steps | 16 steps |
|---|---|---|---|---|
| 4-bit AR + bf16 NAR (`--quant int4 --quant-head`) | 66.1-69.3 s | 55.3 s | — | 43.9 s |
| 8-bit, dequantized NAR | 75.4-76.5 s | 67.3 s | — | — |
| 8-bit, packed NAR | 83.5-86.2 s | — | 64.4 s | — |

### 2-bit and 3-bit AR paths (measured, not adopted)

`int3-mixed` / `int2-mixed` presets (AR path, embeddings and head at 3 / 2 bits, NAR kept at 8 bits), lean profile, 64-frame VAE tile, adaptive limits, 70 s song:

| Preset | Pack | Time | Peak process | LM parity vs bf16 (logits ABC / semantic rel) | ABC score |
|---|---|---|---|---|---|
| int4-mixed (shipped) | 2.5 GB | 73.5 s | 3 397 MB | 0.010 / 0.042 | 14 bars |
| int3-mixed | 2.3 GB | 73.6 s | 3 167 MB | 0.053 / 0.043 | 14 bars (same as 4-bit) |
| int2-mixed | 2.0 GB | 84.0 s | 2 969 MB | 0.072 / 0.068 | 60 bars: diverged on the same seed |
| int4-mixed, 256 MB cache, 3 GB threshold | 2.5 GB | 94.8 s | **2 580 MB** | as 4-bit | unchanged |

The NAR (8-bit, 1.43 GB) dominates the packs, so lower AR bits buy 0.2-0.4 GB at most; the VAE tile and the cache limit buy more (−0.8 GB for +29 % time) with no quality change — that is `4bit-tiny`. By ear (author, 27 September): 2-bit is "no longer music", 3-bit is at the limit and roughly equivalent to 4-bit. Hear the limit yourself: `limit_4bit`, `limit_3bit` and `limit_2bit` (same seed, same song) are attached to the [v1.2.0 release](https://github.com/VincentGourbin/yue2-mlx-swift/releases/tag/v1.2.0). No low-bit profile.

### AR phases: dispatch-bound, not compute-bound

Clean traces (21 September): plan phase at 54-62 % of one core and 68-78 % GPU, semantic phase at 96 % of one core and 72-79 % GPU, NAR at 2 % CPU and 79 % GPU. Semantic phase alone (8-bit, 600 tokens): 114 tok/s where memory bandwidth would allow ≈ 270. The preallocated KV cache (v1.0.0) brought CPU down to 2.2 s user + 0.9 s sys for 6 s of wall time; the compiled decode step brought nothing (103-113 tok/s against 114-115) and is enabled nowhere.

### iPhone 15 Pro Max (A17 Pro, 8 GB, iOS 27.0)

`int4-mixed-head` pack, fp16, stage-scoped residency, mobile profile (`docs/knowledge/benchmarks/iphone-15-pro-max-2026-09-22.md`):

| Measurement | Value |
|---|---|
| AR-only load | 5.6 s, 3.0 GB footprint (before v1.0.0's fp16 fix: ≈ 1.4 GB too much) |
| AR, 512 tokens | 25 tok/s, TTFT 0.3 s |
| NAR MLX, ms per evaluation (256 / 512 / 1 024 / 1 536 frames) | 665 / 1 294 / 2 776 / 6 516 (the last one throttled) |
| Throttling | `thermalState` reaches `fair` after ≈ 60 s of sustained GPU, NAR −31 to −46 %; `serious` on a 60 s song |
| VAE, 1 024 frames | MLX fp16/256: 6.0 s, 2.2 GB peak; Core AI GPU: 5.4 s, 2.1 GB peak |
| 30 s song | 273 s, 3.6 GB peak (plan phase; expected ≈ 2.2-2.4 GB with v1.0.0) |
| 60 s song (app) | ≈ 12.7 min, NAR 645 s under `serious` with pacing, 3.5 GB |
| NAR Core AI GPU | 5-7× slower than MLX (4.4 s per evaluation at 256 frames) |

### Core AI (macOS 27, `coreai-torch` 0.4.2)

| Component | Verdict | Numbers |
|---|---|---|
| VAE decoder, GPU | **valid** | tiled parity 54.7 dB (exact-shape windows), 1.04-1.07 s for 64 s, 2.0-2.1 GB footprint whatever the tile |
| VAE decoder, Neural Engine | dead end | 2-17 dB, silent fallback of the transposed convolutions |
| NAR stack, GPU | "memory-constrained" scenario only | 5-7× slower than MLX, ≈ 0.9 GB less on the NAR stage, buckets ≤ 1 536 frames |
| NAR stack, Neural Engine | crash | `dequantize` unsupported |

## 4. Reproducing

```bash
export YUE2_MODELS_DIR=$HOME/Library/Caches/models
xcodebuild -scheme yue2 -configuration Release -destination 'platform=macOS' -derivedDataPath .xcodebuild build

# the six references, two passes, 70 s song
Scripts/bench-references.sh reference/yue/examples/song.json 1750 2

# one configuration by hand, profiled
.xcodebuild/Build/Products/Release/yue2 generate --request reference/yue/examples/song.json \
  --semantic-min-tokens 1750 --semantic-max-tokens 1750 --reference 4bit-fast --profile \
  --out .local-runs/bench.noindex/my-run

# the NAR alone, MLX against Core AI
.xcodebuild/Build/Products/Release/yue2 bench-coreai-nar --frames 256,1024 --backends mlx,coreai-gpu --quant int4-mixed

# the VAE alone, with footprint
.xcodebuild/Build/Products/Release/yue2 decode --latents run/latent.npy --out out.wav --precision fp16 --core-frames 256
```

Reading a trace: `trace.json` opens in Perfetto or `chrome://tracing`; the `Memory` counters (MLX active, cache, process) and `Utilization` (process CPU, system GPU) are sampled every ≈ 20 ms, phases are `B`/`E` events. The Python script that produced the tables above is twenty lines: aggregate the counters between each phase's bounds.

## 5. Contributing a row

Add a line to `BENCHMARKS.md` (date, commit, configuration, figures copied verbatim, trace path), never edit an existing one. A number without cool-down, without a control, or with a competing GPU process is marked "in session" and is not a reference.
