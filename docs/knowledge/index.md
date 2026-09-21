---
okf_version: "0.1"
---

# Base de connaissances — YuE2Swift

Journal d'ingénierie du portage (bundle OKF). `log.md` porte l'historique horodaté ;
les sous-répertoires accumulent les conclusions durables.

## Benchmarks
Mesures reproductibles (état des horloges GPU, écart ±2 %). Rien pour l'instant.

## Decisions
Choix d'architecture motivés (pourquoi telle voie, tel dtype, tel cache).
- [Résidence des poids par étape (iPhone)](decisions/stage-scoped-weight-residency.md) — le budget mémoire compte le max des étapes, pas leur somme ; implémentation MLX pure.

## Pitfalls
Pièges rencontrés et leur correctif, numérotés comme ceux de PLAN.md §9.
- [Core AI, formes énumérées : zéro-padding des latents](pitfalls/coreai-enumerated-shape-zero-padding.md) — la fin de tuile complétée par des zéros perd 34 dB ; fenêtres à forme exacte (`planVAETiles`).

## Investigations
Enquêtes en cours ou closes (deadlocks, dérives numériques, écarts de parité). Rien pour l'instant.
