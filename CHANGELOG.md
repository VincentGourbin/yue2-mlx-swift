# Changelog

Les versions publiées sont des commits squashés sur `main` (l'historique de travail reste local) ; chaque version a un tag et une release GitHub.

## 1.0.0 — 2026-09-27

Première version « clé en main » : six configurations de référence mesurées et reproductibles, documentation complète.

- **Six profils de référence** (`YuE2ReferenceProfile`, `yue2 references`, `generate --reference <id>`, `Scripts/bench-references.sh`) : 4 / 8 / 16 bits × rapide / économe. Sur la chanson de 70 s (M3 Max) : 65,7 s / 8,9 Go à 84,9 s / 12 Go pour les rapides, 77,7 s / 3,4 Go à 84,9 s / 6,9 Go pour les économes.
- **Cache KV préalloué** (`KVCache`, écriture en place par pas de 512) : plus de copie du cache entier à chaque token ; phases AR −16 % en 4 bits, bit-exact.
- **NAR dé-quantifié pour le solve** (`--nar-compute dequantized`, `YuE2ExecutionPolicy.narCompute`) : le matmul 8 bits est plus lent qu'un GEMM bf16 sur les grandes activations ; NAR −10 % pour +1,4 Go.
- **Conversion de précision limitée aux voies résidentes** (`applyPrecision(_:where:)`) : la conversion fp16 matérialisait la voie NAR parquée, +1,44 Go, le pic de la phase plan sur l'iPhone.
- **Limites mémoire mobiles adaptatives** (`YuE2MemoryManager.mobileLimitsMB()`, `YUE2_CACHE_LIMIT_MB`, `YUE2_MEMORY_LIMIT_MB`) : les 256 Mo / 3 Go fixes coûtaient +32 % (4 bits) à +73 % (bf16).
- **Pas de décodage compilé** (`CompiledDecodeStep`, `--compiled-decode`) : parité 8/8, aucun gain mesuré ; laissé en option désactivée.
- **Bancs** : matrice 16/8/4 bits × MLX pur / optimisé sur 70 s (A/B/B/A, GPU libre), pas d'ODE 32/24/20/16, CPU contre GPU par phase ; discipline de mesure (processus concurrents, pack pré-exporté, sorties `.noindex`).
- **Documentation** : `docs/References.md`, `docs/Benchmarks.md`, `docs/CLI.md`, `docs/API.md`, `docs/iOS.md`, `docs/Weights.md`, README réécrit, ce changelog.

## 0.2.1 — 2026-09-26

- Perte du GPU en arrière-plan sur iOS (Q7) : `NARCheckpoint` (reprise bit-exacte au pas près), `YuE2GPUGate` (attendue avant chaque token, couche NAR, tuile VAE), `YuE2ExecutionPolicy.evalPerLayer`.
- README : état du 26 septembre.

## 0.2.0 — 2026-09-24

- Head restreint quantifié (`RestrictedHead` par `quantizedMatmul`) : pic de la phase plan −283 Mo, débit ABC 57,7 → 131 tok/s.
- Premières mesures sur iPhone 15 Pro Max ; pack `int4-mixed-head` ; G-9 / G-10 ; NAR Core AI GPU mesuré sur l'appareil (5-7× plus lent).

## 0.1.0-beta — 2026-09-21

- Résidence des poids par étape ; fenêtres à forme exacte pour le VAE Core AI (parité 33,8 → 54,7 dB) ; `FootprintSampler` ; le pic du pipeline identifié dans le décodage VAE.

## 0.1 — 2026-09-20

- Publication initiale : pipeline complet (partition ABC, tokens sémantiques, flow matching, VAE Oobleck), parité contre l'implémentation PyTorch, encodeur VAE, quantification 8 bits de la voie AR, GUI de bench.
