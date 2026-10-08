---
okf_version: "0.1"
---

# Base de connaissances — YuE2Swift

Journal d'ingénierie du portage (bundle OKF). `log.md` porte l'historique horodaté ;
les sous-répertoires accumulent les conclusions durables.

## Benchmarks
Mesures reproductibles (état des horloges GPU, écart ±2 %).
- [iPhone 15 Pro Max, premier balayage (E5)](benchmarks/iphone-15-pro-max-2026-09-22.md) — AR 25 tok/s, NAR 2,6 ms/frame/éval à froid puis ×1,6 dès que `thermalState` passe à fair (≈ 60 s de charge), VAE 6 s / 41 s d'audio ; clip de 30 s réel en 273 s, pic 3,6 Go.
- [SheetSage2 sur iPhone : mesures Mac et chiffrage](benchmarks/sheetsage2-port-estimate-2026-10-03.md) — encodeur toujours sur 300 s (4,4 s Mac), décodeur 5,5 ms/pas ; iPhone estimé ≈ 30 s pour un extrait, 1-1,5 min pour 3 min, pic ≈ 1,7 Go fp16 ; 8-11 jours, le symbolique pèse plus que le réseau.
- [Instrumental et karaoké](benchmarks/instrumental-and-karaoke-2026-10-07.md) — transfert Vocal→Ins indispensable (0/1 sans, 6/8 avec branche négative), contrôle de fuite SheetSage2 + relance ; paroles horodatées depuis la partition et la grille SheetSage2 (81-96 % des mots à ≤ 300 ms, 98-100 % avec ancres MMS) ; puis (2026-10-08) **têtes d'alignement du LM** : mesures et mots minutés sans analyse audio (71-97 % des mots à ≤ 300 ms sur 8 chansons, robuste quand la voix quitte la partition), `generate --timeline`.
- [Passage de relais : minutage pour l'app](handover-timeline-2026-10-08.md) — appel `AttentionTimeline.timeline` après la phase sémantique, format de `timeline.json`, ce que l'app n'a plus à faire.

## Decisions
Choix d'architecture motivés (pourquoi telle voie, tel dtype, tel cache).
- [Six configurations de référence](decisions/reference-profiles.md) — 4/8/16 bits × rapide/économe, temps et pics sur 70 s, choix motivés, points ouverts.
- [Entrée audio par les entrées natives du modèle](decisions/audio-input-via-model-entries.md) — mélodie fredonnée → ABC imposé, variation SDEdit, régénération avec masque ; mesures et pièges (YIN, longueur de l'AR).
- [Résidence des poids par étape (iPhone)](decisions/stage-scoped-weight-residency.md) — le budget mémoire compte le max des étapes, pas leur somme ; implémentation MLX pure.

## Pitfalls
Pièges rencontrés et leur correctif, numérotés comme ceux de PLAN.md §9.
- [Conversion de précision et voie non résidente](pitfalls/precision-cast-materializes-parked-branch.md) — `applyPrecision(.fp16)` matérialisait la voie NAR parquée (+1,44 Go, le pic iPhone) ; conversion restreinte aux clés résidentes.
- [Core AI, formes énumérées : zéro-padding des latents](pitfalls/coreai-enumerated-shape-zero-padding.md) — la fin de tuile complétée par des zéros perd 34 dB ; fenêtres à forme exacte (`planVAETiles`).

## Transmissions et mémoire
- [Mémoire d'agent](agent-memory.md) — conventions de Vincent, état du projet, pièges durables : à lire en début de session.
- [Transmission entrée audio (2026-10-02)](handover-audio-input-2026-10-02.md) — inpaint/ancrage d'un extrait réel réfutés à l'oreille ; la voie est symbolique (SheetSage2 → ABC → `cot melody`) ; état des commits locaux.

## Investigations
Enquêtes en cours ou closes (deadlocks, dérives numériques, écarts de parité). Rien pour l'instant.
