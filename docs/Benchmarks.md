# Benchmarks : méthode, résultats, reproduction

Ce document est le guide de lecture. Les lignes brutes vivent dans [`BENCHMARKS.md`](../BENCHMARKS.md) (une ligne par run, jamais modifiée), les analyses datées dans [`docs/knowledge/benchmarks/`](knowledge/benchmarks/) et le journal dans [`docs/knowledge/log.md`](knowledge/log.md).

## 1. Ce qu'on mesure, et avec quoi

- **Temps** par phase (partition ABC, tokens sémantiques, NAR, VAE) et total hors chargement, tels que `yue2 generate --profile` les imprime (`generated … in X s (abc / semantic / nar / vae)`).
- **Mémoire** : pic de footprint process (`phys_footprint`, le chiffre que juge jetsam sur iOS) et pic de mémoire MLX active, par phase, depuis la trace Chrome (`trace.json`) écrite par `swift-mlx-profiler`. Sur les commandes unitaires (`decode`, `bench-coreai-nar`), les lignes `DECODE …` / `BENCH …` portent les mêmes chiffres (`FootprintSampler`, 5 ms).
- **Occupation CPU / GPU** par phase, dans la même trace (compteur `Utilization`).
- Jamais de `print` chronométré à la main.

## 2. Le protocole (CLAUDE.md, « Performance work »)

L'état d'horloge du GPU déplace un résultat jusqu'à 10×. Une mesure de temps n'est valable que si :

1. **GPU libre** : `pgrep -fl "serve|train|lora"` avant chaque point. Un entraînement LoRA tiers a multiplié par trois toute une journée de mesures (26 septembre) sans qu'aucun chiffre ne paraisse absurde isolément.
2. **Refroidissement** : 120 s avant chaque point (60 s pour une phase seule).
3. **A/B/B/A** avec contrôle : deux runs de chaque configuration, ordre A B B A ; on ne lit un écart que s'il dépasse la dispersion des deux A (≤ 3 % sur GPU libre).
4. **Pack pré-quantifié déjà exporté** : la première utilisation d'un preset quantifié charge 7,3 Go de bf16, quantifie et exporte (`mlx-prequantized/<preset>[-head]/`) — un run à jeter.
5. **Sorties dans `.local-runs/bench.noindex/`** : Spotlight indexe les WAV et npy fraîchement écrits (`corespotlightd` : 207 min de CPU cumulé sur la machine de référence) et vole le cœur dont dépendent les phases AR ; le suffixe `.noindex` l'arrête. `mediaanalysisd`, Time Machine (`backupd`) et WindowServer restent des perturbateurs occasionnels : le script journalise le processus le plus gourmand avant chaque point, un run hors bande est refait, jamais interprété.

`Scripts/bench-references.sh` applique tout cela aux six références ; `Scripts/bench-song.sh` fait l'A/B/B/A sur une chanson courte.

## 3. Résultats de référence (M3 Max 96 Go, macOS 27, mlx-swift 0.31.6)

### Les six configurations, chanson de 70 s, 32 pas

| Id | Total | abc / sémantique / NAR / VAE | Pic process | Pic par phase (plan / sém. / NAR / VAE) |
|---|---|---|---|---|
| `4bit-fast` | **65,7 s** | 2,7 / 10,7 / 51,1 / 1,1 | 8,9 Go | 4,2 / 5,3 / 8,5 / 8,9 |
| `4bit-lean` | 77,7 s | 4,2 / 10,8 / 61,5 / 1,2 | **3,4 Go** | 2,8 / 3,4 / 2,8 / 3,4 |
| `8bit-fast` | 76,5 s | 5,8 / 15,5 / 54,0 / 1,1 | 9,9 Go | 5,3 / 6,4 / 9,6 / 9,9 |
| `8bit-lean` | 79,3 s | 5,6 / 15,3 / 57,1 / 1,1 | 4,4 Go | 4,0 / 4,4 / 3,8 / 3,4 |
| `16bit-fast` | 84,9 s | 8,7 / 23,8 / 51,3 / 1,1 | 12,0 Go | 8,8 / 9,7 / 10,3 / 12,0 |
| `16bit-lean` | 84,9 s | 8,4 / 23,1 / 52,2 / 1,2 | 6,9 Go | 6,0 / 6,9 / 6,9 / 3,5 |

Un passage (traces `.local-runs/bench.noindex/ref-20260926-2300-*`). Les contrôles A/B/B/A de la même soirée sur les mêmes packs tiennent à ±3 %.

### Bits × mode, A/B/B/A sur GPU libre (même chanson, 32 pas)

| Pack | MLX pur (2 runs) | Optimisé mobile (2 runs) | Pic pur → optimisé |
|---|---|---|---|
| bf16 | 89,4 / 87,2 s | 98,2 / 134,7 s | 14,8 → 5,2 Go |
| 8 bits intégral | 83,5 / 86,2 s | 84,9 / 85,6 s | 11,5 → 4,3 Go |
| 4 bits `int4-mixed-head` | 76,6 / 79,0 s | 76,6 / 109,5 s | 10,4 → 3,3-3,5 Go |

« Optimisé » = résidence par étape + VAE fp16/256 + limites mobiles adaptatives. En 8 et 4 bits la résidence ne coûte rien ; les deux runs hors bande (bf16 134,7 s : récupération du cache au seuil, working set 4,3 Go ; 4 bits 109,5 s : état GPU ponctuel, trace sans anomalie, contrôle suivant à 79 s) sont documentés dans `docs/knowledge/benchmarks/mac-70s-quant-comparison-2026-09-26.md`.

### Pas d'ODE (levier ÷ 2 sur le NAR, qualité à valider)

| Configuration | 32 pas | 24 pas | 20 pas | 16 pas |
|---|---|---|---|---|
| AR 4 bits + NAR bf16 (`--quant int4 --quant-head`) | 66,1-69,3 s | 55,3 s | — | 43,9 s |
| 8 bits, NAR dé-quantifié | 75,4-76,5 s | 67,3 s | — | — |
| 8 bits, NAR packé | 83,5-86,2 s | — | 64,4 s | — |

### Phases AR : bornées par le dispatch, pas par le calcul

Traces propres (21 septembre) : phase plan à 54-62 % d'un cœur et 68-78 % de GPU, phase sémantique à 96 % d'un cœur et 72-79 % de GPU, NAR à 2 % de CPU et 79 % de GPU. Phase sémantique seule (8 bits, 600 tokens) : 114 tok/s là où la bande passante permettrait ≈ 270. Le cache KV préalloué (v1.0.0) a ramené le CPU à 2,2 s user + 0,9 s sys pour 6 s réels ; le pas de décodage compilé n'a rien apporté (103-113 tok/s contre 114-115) et n'est activé nulle part.

### iPhone 15 Pro Max (A17 Pro, 8 Go, iOS 27.0)

Pack `int4-mixed-head`, fp16, résidence par étape, profil mobile (`docs/knowledge/benchmarks/iphone-15-pro-max-2026-09-22.md`) :

| Mesure | Valeur |
|---|---|
| Chargement AR seul | 5,6 s, 3,0 Go de footprint (avant le correctif fp16 de v1.0.0 : ≈ 1,4 Go de trop) |
| AR, 512 tokens | 25 tok/s, TTFT 0,3 s |
| NAR MLX, ms par évaluation (256 / 512 / 1 024 / 1 536 frames) | 665 / 1 294 / 2 776 / 6 516 (le dernier throttlé) |
| Throttling | `thermalState` passe à `fair` après ≈ 60 s de GPU soutenu, NAR −31 à −46 % ; `serious` sur une chanson de 60 s |
| VAE 1 024 frames | MLX fp16/256 : 6,0 s, pic 2,2 Go ; Core AI GPU : 5,4 s, pic 2,1 Go |
| Chanson de 30 s | 273 s, pic 3,6 Go (phase plan ; attendu ≈ 2,2-2,4 Go avec v1.0.0) |
| Chanson de 60 s (app) | ≈ 12,7 min, NAR 645 s en `serious` avec pacing, 3,5 Go |
| NAR Core AI GPU | 5-7× plus lent que MLX (4,4 s par évaluation à 256 frames) |

### Core AI (macOS 27, `coreai-torch` 0.4.2)

| Composant | Verdict | Chiffres |
|---|---|---|
| Décodeur VAE, GPU | **valide** | parité 54,7 dB tuilé (fenêtres à forme exacte), 1,04-1,07 s pour 64 s, footprint 2,0-2,1 Go quelle que soit la tuile |
| Décodeur VAE, Neural Engine | impasse | 2-17 dB, repli silencieux des convolutions transposées |
| Pile NAR, GPU | scénario « mémoire contrainte » seulement | 5-7× plus lent que MLX, ≈ 0,9 Go de moins sur l'étape NAR, paliers ≤ 1 536 frames |
| Pile NAR, Neural Engine | crash | `dequantize` non supporté |

## 4. Reproduire

```bash
export YUE2_MODELS_DIR=$HOME/Library/Caches/models
xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build

# les six références, deux passes, chanson de 70 s
Scripts/bench-references.sh reference/yue/examples/song.json 1750 2

# une configuration à la main, profilée
.xcodebuild/Build/Products/Release/yue2 generate --request reference/yue/examples/song.json \
  --semantic-min-tokens 1750 --semantic-max-tokens 1750 --reference 4bit-fast --profile \
  --out .local-runs/bench.noindex/my-run

# le NAR seul, MLX contre Core AI
.xcodebuild/Build/Products/Release/yue2 bench-coreai-nar --frames 256,1024 --backends mlx,coreai-gpu --quant int4-mixed

# le VAE seul, avec footprint
.xcodebuild/Build/Products/Release/yue2 decode --latents run/latent.npy --out out.wav --precision fp16 --core-frames 256
```

Lire une trace : `trace.json` s'ouvre dans Perfetto ou `chrome://tracing` ; les compteurs `Memory` (MLX actif, cache, process) et `Utilization` (CPU du processus, GPU système) sont échantillonnés toutes les ≈ 20 ms, les phases sont des événements `B`/`E`. Le script Python qui a produit les tableaux ci-dessus tient en vingt lignes : sommer les compteurs entre les bornes de chaque phase.

## 5. Contribuer une ligne

Ajouter une ligne à `BENCHMARKS.md` (date, commit, configuration, chiffres copiés tels quels, chemin de la trace), jamais modifier une ligne existante. Un chiffre sans refroidissement, sans contrôle ou avec un processus GPU concurrent se marque « en session » et ne sert pas de référence.
