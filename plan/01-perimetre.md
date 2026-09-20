## 1. Objectif et périmètre

Porter **YuE2** (m-a-p, sept. 2026 — génération de chansons complètes 48 kHz stéréo avec plan symbolique ABC éditable) en **MLX Swift** pour Apple Silicon, dans la lignée de `gemma-4-swift-mlx`, `h3-swift-mlx`, `flux-2-swift-mlx`, `convertvoxtral`, en réutilisant les briques et optimisations déjà validées.

| Modèle | Rôle | Taille | Licence | Verdict M3 Max 96 Go |
|---|---|---|---|---|
| `m-a-p/YuE2-3B` | LM AR–NAR Mixture-of-Transformers (3,63 G params, 28 couches) | 7,26 Go bf16 (1 fichier `model.safetensors`) | CC-BY-NC-4.0 | ✅ **Jalons 2-3** — bf16 tient largement |
| `m-a-p/YuE2-Vae` | Décodeur audio Oobleck (latents 64-d @ 25 Hz → stéréo 48 kHz) | 530 Mo fp32 (décodeur seul : 253 Mo) | CC-BY-NC-4.0 | ✅ **Jalon 1** |
| `m-a-p/YuE2-Vae-legacy` | Même topologie, poids du protocole benchmark | 530 Mo | CC-BY-NC-4.0 | ✅ option `--vae legacy`, même code |
| `m-a-p/MERT-v2-30s`, `MERT-v2-FullSong` | Encodeur audio (632 M) | 2,5 Go | — | ❌ **Hors périmètre** : non utilisé à l'inférence (voir §2.9) |
| `m-a-p/SheetSage2` | Transcription audio → ABC (pour les covers) | — | — | ❌ Hors périmètre v1 (l'utilisateur fournit l'ABC) |

**Fait structurant** : la génération YuE2 est **texte → tokens → latents → audio**, sans aucune entrée audio. Le pipeline Python est `plan()` → `generate_semantic()` → `synthesize()` → `decode()` ; les quatre étapes tournent sur le **même** checkpoint LM (voies AR et NAR séparées dans chaque couche) plus le VAE. Le « tokenizer sémantique » audio n'est pas publié et n'est pas nécessaire.

**Périmètre v1 (Jalons 1-4)** : les trois modes `cot` (`full`, `melody`, `off`), ABC fourni (`abc=`), CFG (`cfg_scale`), seed, sauvegarde des artefacts (`score.abc`, `semantic.npy`, `latent.npy`, `audio.wav`, `result.json`), CLI, GUI de bench, profiler, quantification optionnelle de la voie AR.

**Hors périmètre v1** : encodeur VAE (encodage audio), serveur HTTP, batching multi-requêtes, iOS, vLLM/FP8/CUDA-graphs (spécifiques NVIDIA), reproduction bit-à-bit du RNG PyTorch (impossible : on injecte le bruit par fixture pour la parité, cf. §8).

