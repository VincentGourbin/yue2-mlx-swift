# Weights, packs and licence

## What the engine loads

| Directory (`$YUE2_MODELS_DIR/…`) | Source | Size | Licence |
|---|---|---|---|
| `YuE2-3B/` | `m-a-p/YuE2-3B` (bf16, 628 tensors) + `tokenizer.json` converted from `Qwen/Qwen2.5-0.5B` | 7.3 GB | CC-BY-NC-4.0 (tokenizer: Apache-2.0) |
| `YuE2-Vae/` | `m-a-p/YuE2-Vae` (Oobleck decoder and encoder, fp32) | 0.5 GB | CC-BY-NC-4.0 |
| `YuE2-Vae-legacy/` | older VAE, optional | 0.5 GB | CC-BY-NC-4.0 |
| `YuE2-3B/mlx-prequantized/<preset>[-head]/model.safetensors` | **produced locally** on the first use of a preset | 2.4-5.5 GB | derivative, same licence |
| `YuE2-Vae/coreai/*.aimodel`, `YuE2-3B/coreai/*.aimodel` | `Scripts/coreai/export_*.py` | 0.13 / 1.3 GB | derivative |

`yue2 download` fetches the first two. The code in this repository is MIT; the weights are never distributed here.

## The prequantized packs

| Pack | Content | Size | Profiles |
|---|---|---|---|
| `int4-mixed-head` | AR path, embeddings, head in 4 bits (group 64, affine), NAR path in 8 bits, `latent_pos_embed.pe` recomputed | 2.5 GB | `4bit-fast`, `4bit-lean` |
| `qint8-all-head` | everything in 8 bits (group 64) | 3.7 GB | `8bit-fast`, `8bit-lean` |
| `int4-head` | AR path, embeddings, head in 4 bits, NAR bf16 | 3.5 GB | the fastest (66-69 s), candidate for a seventh profile |
| `qint8`, `qint8-head`, `int4`, `int4-all-head` | measurement variants | 2.4-5.5 GB | — |

The first use of a preset loads the whole bf16 checkpoint (7.3 GB), quantizes, writes the export and carries on: several minutes and an 8-14 GB peak, once per machine. Later loads read the export directly (`LMWeightLoader.load`), including branch by branch for stage-scoped residency.

Pack parities (real fixtures, `yue2 parity lm|nar`): AR int4 greedy 8/8 on both phases, rel 0.0075; NAR 8-bit rel 0.049 over a 32-step solve (budget 0.10), validated by listening; NAR 4-bit rel 0.15, rejected.

## Republishing reference packs on Hugging Face

To avoid local quantization (and 7.3 GB of bf16 on an iPhone), the reference packs can be published as derivatives, under the same CC-BY-NC-4.0 licence with attribution to m-a-p, non-commercial use. This repository does not do it: it is the author's decision and account.

Proposal: a `VincentGourbin/yue2-mlx-packs` repository with one folder per pack (`int4-mixed-head/`, `qint8-all-head/`, `int4-head/`), the exported `model.safetensors` as is (metadata `format`, `quantization`, `quantize_head`, `bits`, `group_size`, `pe`), a model-card `README.md`, and each file's SHA-256. `Scripts/publish-packs.sh` prepares the tree and prints the `hf upload` commands; it uploads nothing by itself.

Once published, `yue2 download --model packs` (to be written) would place them under `mlx-prequantized/`; the iOS app would download them on first launch (2.5 GB for `4bit-lean`).

## Model card (draft)

```
# YuE2 MLX packs — prequantized weights for yue2-mlx-swift

Derived from m-a-p/YuE2-3B (CC-BY-NC-4.0) for the Swift/MLX port
https://github.com/VincentGourbin/yue2-mlx-swift (MIT). Non-commercial use, attribution m-a-p.

| Pack | Bits | Size | Profiles | Parity |
| int4-mixed-head | AR/embed/head 4, NAR 8 | 2.5 GB | 4bit-fast, 4bit-lean | AR greedy 8/8; NAR rel 0.049 (32 steps) |
| qint8-all-head | 8 throughout | 3.7 GB | 8bit-fast, 8bit-lean | AR rel 0.005; NAR rel 0.027 |
| int4-head | AR/embed/head 4, NAR bf16 | 3.5 GB | (fast) | AR greedy 8/8 |

Format: MLX safetensors (packed uint32 weights + scales + biases, group 64, affine), metadata
`format=yue2-prequantized-v1`. Loading: `yue2 generate --reference <profile>` or
`LMWeightLoader.load(directory:config:quantization:quantizeHead:)`.
```
