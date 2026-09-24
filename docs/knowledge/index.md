---
okf_version: "0.1"
---

# Base de connaissances — YuE2Swift

Journal d'ingénierie du portage (bundle OKF). `log.md` porte l'historique horodaté ;
les sous-répertoires accumulent les conclusions durables.

## Benchmarks
Mesures reproductibles (état des horloges GPU, écart ±2 %).
- [iPhone 15 Pro Max, premier balayage (E5)](benchmarks/iphone-15-pro-max-2026-09-22.md) — AR 25 tok/s, NAR 2,6 ms/frame/éval à froid puis ×1,6 dès que `thermalState` passe à fair (≈ 60 s de charge), VAE 6 s / 41 s d'audio ; clip de 30 s réel en 273 s, pic 3,6 Go.

## Decisions
Choix d'architecture motivés (pourquoi telle voie, tel dtype, tel cache).
- [Résidence des poids par étape (iPhone)](decisions/stage-scoped-weight-residency.md) — le budget mémoire compte le max des étapes, pas leur somme ; implémentation MLX pure.

## Pitfalls
Pièges rencontrés et leur correctif, numérotés comme ceux de PLAN.md §9.
- [Core AI, formes énumérées : zéro-padding des latents](pitfalls/coreai-enumerated-shape-zero-padding.md) — la fin de tuile complétée par des zéros perd 34 dB ; fenêtres à forme exacte (`planVAETiles`).

## Investigations
Enquêtes en cours ou closes (deadlocks, dérives numériques, écarts de parité). Rien pour l'instant.
