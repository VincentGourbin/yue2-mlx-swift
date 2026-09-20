# YuE2-mlx-swift — Plan d'implémentation

> **Statut : rév. 1 du 2026-09-16 — plan initial, rédigé après lecture intégrale de la référence Python (`src/yue2/*.py`, 3 785 lignes) et inventaire des briques réutilisables dans les dépôts frères.** Document de référence **local**. Le journal d'exécution (REQUEST / RÉPONSE / Étapes) sera ajouté à la suite du §12.
> Roadmap : **Jalon 1 (fondations + VAE) → Jalon 2 (backbone AR : plan ABC + tokens sémantiques) → Jalon 3 (NAR flow matching + chanson complète) → Jalon 4 (optimisations + GUI de bench)**. À chaque jalon, Vincent teste lui-même.
> Cible matérielle : MacBook Pro M3 Max, 96 Go RAM unifiée, macOS 27, Xcode 27 (Swift 6.4). Poids stockés sous `$YUE2_MODELS_DIR` (jamais de chemin absolu dans le dépôt).
> Référence Python : https://github.com/multimodal-art-projection/YuE (commit `ef1936f`, 2026-09-16), paquet `yue2-infer 0.1.6`. Poids : `m-a-p/YuE2-3B` (7,26 Go bf16), `m-a-p/YuE2-Vae` (530 Mo fp32).

---

---

## Sommaire — le plan vit dans `plan/`, un fichier par section

**Ce fichier est un sommaire (40 lignes). Ne pas chercher le plan ici : ouvrir le fichier de la section demandée, un seul à la fois.** Les fiches (`tasks/T-x.y.md`) citent ces fichiers par nom.

| Fichier | Section | Mots |
|---|---|---:|
| `plan/00-mode-execution.md` | 0. Mode d'exécution — ce plan sera exécuté par un agent (modèle plus petit) | 685 |
| `plan/00.2-lecture.md` | 0.2 Comment lire ce plan (protocole de lecture — obligatoire pour l'exécutant) | 334 |
| `plan/00.3-validation.md` | 0.3 Comment valider une étape (protocole de validation) | 628 |
| `plan/00.4-executant-local.md` | 0.4 Exécution par un modèle local (Qwen3.8 Flash-Next via `qwen38 serve` + harnais **pi**) | 645 |
| `plan/01-perimetre.md` | 1. Objectif et périmètre | 325 |
| `plan/02.0-architecture-intro.md` | 2. Rappel d'architecture (ce que l'exécutant doit connaître — ne pas re-dériver) | 45 |
| `plan/02.1-vocabulaire.md` | 2.1 Vocabulaire et protocole de prompt (`protocol.py`) | 387 |
| `plan/02.2-backbone-mot.md` | 2.2 Backbone MoT (`modeling_yue2.py`) | 265 |
| `plan/02.3-etape-ar.md` | 2.3 Étape AR — phases ABC et sémantique (`sampling.py`) | 406 |
| `plan/02.4-etape-nar.md` | 2.4 Étape NAR — flow matching (`nar.py`) | 479 |
| `plan/02.5-vae.md` | 2.5 VAE — décodeur Oobleck (`modeling_vae.py`, dérivé de stable-audio-tools) | 509 |
| `plan/02.6-tenseurs.md` | 2.6 Noms de tenseurs (contrat du loader — 628 tenseurs LM, 435 VAE) | 185 |
| `plan/02.7-tokenizer.md` | 2.7 Tokenizer (`tokenization_yue2.py`) | 139 |
| `plan/02.8-dtypes.md` | 2.8 Ordre des dtypes (à respecter pour la parité) | 92 |
| `plan/02.9-hors-perimetre.md` | 2.9 Ce qui n'est PAS à porter | 78 |
| `plan/03-budget.md` | 3. Budget mémoire et performance attendue (M3 Max 96 Go) | 445 |
| `plan/04-jalon1.md` | 4. Jalon 1 — Fondations et VAE (≈ 2-3 jours) | 1046 |
| `plan/05-jalon2.md` | 5. Jalon 2 — Backbone AR : plan ABC et tokens sémantiques (≈ 4-6 jours) | 933 |
| `plan/06-jalon3.md` | 6. Jalon 3 — NAR flow matching et chanson complète (≈ 4-6 jours) | 569 |
| `plan/07-jalon4.md` | 7. Jalon 4 — Optimisations et GUI de bench (≈ 5-8 jours, ⛔ GATE G-5 en entrée) | 582 |
| `plan/08-tests.md` | 8. Stratégie de tests et fixtures de parité | 960 |
| `plan/09-pieges.md` | 9. Checklist de pièges (contrat de revue — cocher les 16 à chaque tâche) | 387 |
| `plan/10-ecosysteme.md` | 10. Écosystème — brique → dépôt → fichier | 376 |
| `plan/11-risques.md` | 11. Risques | 265 |
| `plan/12-decisions.md` | 12. Décisions actées et questions restantes | 144 |
| `plan/annexe-A-fixtures.md` | Annexe A — Squelette de `Scripts/reference/tiny_fixtures.py` (à compléter en 1.3) | 374 |
| `plan/annexe-B-claude-md.md` | Annexe B — `CLAUDE.md` à créer en 1.0 (contenu de départ, à maintenir) | 329 |
| `plan/annexe-C-commandes.md` | Annexe C — Commandes de référence | 95 |

Pour lire le plan d'un bloc (humain seulement) : `Scripts/plan-concat.sh | less`.
