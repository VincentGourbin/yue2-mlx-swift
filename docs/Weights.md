# Weights, packs and licence

## What the engine loads

| Directory (`$YUE2_MODELS_DIR/…`) | Source | Size | Licence |
|---|---|---|---|
| `YuE2-3B/` | `m-a-p/YuE2-3B` (bf16, 628 tensors) + `tokenizer.json` converted from `Qwen/Qwen2.5-0.5B` | 7.3 GB | CC-BY-NC-4.0 (tokenizer: Apache-2.0) |
| `YuE2-Vae/` | `m-a-p/YuE2-Vae` (Oobleck decoder and encoder, fp32) | 0.5 GB | CC-BY-NC-4.0 |
| `YuE2-Vae-legacy/` | older VAE, optional | 0.5 GB | CC-BY-NC-4.0 |
| `YuE2-3B/mlx-prequantized/<preset>[-head]/model.safetensors` | **produced locally** on the first use of a preset | 2.4-5.5 GB | derivative, same licence |
| `YuE2-Vae/coreai/*.aimodel`, `YuE2-3B/coreai/*.aimodel` | `Scripts/coreai/export_*.py` | 0.13 / 1.3 GB | derivative |
| `SheetSage2/` | `m-a-p/SheetSage2` @ `398b228` (LoRA adapters + BART decoder, fp32), for `yue2 transcribe` | 0.23 GB | CC-BY-NC-4.0 |
| `MERT-v2-FullSong/` | `m-a-p/MERT-v2-FullSong` @ `d8ba1c7` (the parent revision SheetSage2 pins, not `main`) | 2.5 GB | CC-BY-NC-4.0 |

`yue2 download` fetches the first two; `yue2 download --model sheetsage2` the last two (public repositories, no token needed), merged at load. The quicker route for transcription is the merged fp16 pack below.

| `SheetSage2-fp16/` | [`VincentGOURBIN/sheetsage2-mlx-fp16`](https://huggingface.co/VincentGOURBIN/sheetsage2-mlx-fp16): the two above merged and cast to fp16 (ConvNeXt front and mel buffers kept fp32), SHA-256 verified — `yue2 download --model sheetsage2-fp16` | 1.4 GB | CC-BY-NC-4.0 derivative (`Scripts/publish-sheetsage2-pack.sh`) |

With the pack, `yue2 transcribe` loads in 0.2 s and 1.4 GB, and gives the same tokens and scores as the upstream release under the `16bit-*` profiles (`SheetSage2FP16PackTests`); it is the default `--model` when present. The adapters are merged into MERT at load; no merged copy is published. The code in this repository is MIT; the weights are never distributed here.

## The prequantized packs

| Pack | Content | Size | Profiles |
|---|---|---|---|
| `int4-mixed-head` | AR path, embeddings, head in 4 bits (group 64, affine), NAR path in 8 bits, `latent_pos_embed.pe` recomputed | 2.5 GB | `4bit-fast`, `4bit-lean` |
| `qint8-all-head` | everything in 8 bits (group 64) | 3.5 GB | `8bit-fast`, `8bit-lean` |
| `int4-head` | AR path, embeddings, head in 4 bits, NAR bf16 | 4.4 GB | the fastest (66-69 s), candidate for a seventh profile |
| `qint8`, `qint8-head`, `int4`, `int4-all-head` | measurement variants | 2.4-5.5 GB | — |

The first use of a preset loads the whole bf16 checkpoint (7.3 GB), quantizes, writes the export and carries on: several minutes and an 8-14 GB peak, once per machine. Later loads read the export directly (`LMWeightLoader.load`), including branch by branch for stage-scoped residency.

Pack parities (real fixtures, `yue2 parity lm|nar`): AR int4 greedy 8/8 on both phases, rel 0.0075; NAR 8-bit rel 0.049 over a 32-step solve (budget 0.10), validated by listening; NAR 4-bit rel 0.15, rejected.

## The published packs (Hugging Face)

The three reference packs are published as derivatives at [`VincentGOURBIN/yue2-mlx-packs`](https://huggingface.co/VincentGOURBIN/yue2-mlx-packs), under the same CC-BY-NC-4.0 licence with attribution to m-a-p, non-commercial use: one folder per pack, the exported `model.safetensors` as is (metadata `format`, `quantization`, `quantize_head`, `bits`, `group_size`, `pe`) and its SHA-256 sidecar.

```bash
yue2 download --model int4-mixed-head        # one pack (2.5 GB) — what 4bit-fast / 4bit-lean load
yue2 download --model packs                  # all three (10.4 GB)
yue2 references                              # prints the download line for each profile
```

`ModelDownloader.download(pack:)` fetches the sidecar, then the weights (resumable), and refuses a file whose SHA-256 does not match; a pack already present and verified is skipped. Files land exactly where `LMWeightLoader.load` looks for the preset, so the first-use quantization never runs. `Scripts/publish-packs.sh` is what produced the staging tree and checksums (it uploads nothing by itself; `hf upload` did).

In Swift: `YuE2ReferenceProfile.named("4bit-lean")!.pack` → `.int4MixedHead`; `try await ModelDownloader(modelsDir: dir).download(pack: .int4MixedHead)`. The iOS app can download the profile's pack on first launch instead of shipping it through Finder.

## Model card (draft)

```
# YuE2 MLX packs — prequantized weights for yue2-mlx-swift

Derived from m-a-p/YuE2-3B (CC-BY-NC-4.0) for the Swift/MLX port
https://github.com/VincentGourbin/yue2-mlx-swift (MIT). Non-commercial use, attribution m-a-p.

| Pack | Bits | Size | Profiles | Parity |
| int4-mixed-head | AR/embed/head 4, NAR 8 | 2.5 GB | 4bit-fast, 4bit-lean | AR greedy 8/8; NAR rel 0.049 (32 steps) |
| qint8-all-head | 8 throughout | 3.5 GB | 8bit-fast, 8bit-lean | AR rel 0.005; NAR rel 0.027 |
| int4-head | AR/embed/head 4, NAR bf16 | 4.4 GB | (fast) | AR greedy 8/8 |

Format: MLX safetensors (packed uint32 weights + scales + biases, group 64, affine), metadata
`format=yue2-prequantized-v1`. Loading: `yue2 generate --reference <profile>` or
`LMWeightLoader.load(directory:config:quantization:quantizeHead:)`.
```
