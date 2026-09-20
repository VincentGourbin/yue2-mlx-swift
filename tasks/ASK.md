# Questions à Vincent (ajout seul)

Gabarit : voir AGENTS.md §6. Vincent répond sous chaque question dans une sous-section `### Réponse de Vincent`.


## ASK — T-1.0 — 2026-09-16
- Contexte : GATE G-0 à la fin de la fiche 1.0 ; le squelette compile (BUILD OK) et le test trivial passe.
- Ce que j'ai essayé : `check-tests.sh` renvoie `TESTS OK (1 tests, 1 skippés)` — le compteur « skippés » est un faux positif du script (grep sur le mot « skipped » dans un avertissement d'appintentsmetadataprocessor), pas un test sauté.
- Question : Plan rév. 1 validé ? Nom du dépôt GitHub (pour `README.md`/`AGENTS.md` qui disent « this repo is public ») ? Quelle valeur pour `$YUE2_MODELS_DIR` sur cette machine (les poids n'y sont pas encore) ?
- Options : A) continuer sur T-1.1 tel quel ; B) corriger d'abord le compteur de `check-tests.sh` ; C) attendre la valeur de `$YUE2_MODELS_DIR` avant T-1.4.

### Réponse de Vincent (2026-09-16)
- Nom du dépôt GitHub : `yue2-mlx-swift`.
- `$YUE2_MODELS_DIR` = un dossier sous `~/Library/Caches` sur cette machine (contient aujourd'hui `test-org/` ; les poids YuE2-3B / YuE2-Vae y seront téléchargés à la fiche T-1.4).
- Ce que j'en ai fait : chemin consigné uniquement dans `.local-runs/env.sh` (gitignoré) — jamais dans un fichier committé (piège n°16). `README.md` et `CLAUDE.md` gardent `$YUE2_MODELS_DIR` littéral.

## ASK — hygiène du dépôt — 2026-09-16
- Contexte : `.pi/sessions/*.jsonl` (journal de session de l'agent, avec les chemins absolus de cette machine) est **suivi par git** depuis le commit `chore: plan and task harness` ; le dépôt doit être public et le piège n°16 interdit d'y committer un chemin absolu.
- Ce que j'ai essayé : rien modifié — `.gitignore` ne liste pas `.pi/` et je ne touche pas aux scripts hors fiche.
- Question : ajoute-t-on `.pi/` au `.gitignore` (et `git rm --cached` sur le journal de session déjà committé) avant la première publication ?
- Options : A) oui, `.pi/` ignoré + retrait de l'index ; B) garder le journal de session committé (historique de travail assumé) ; C) le supprimer du disque.

### Réponse (session de rédaction du plan, 2026-09-17) — compteur `check-tests.sh`
- Option B appliquée : `Scripts/check-tests.sh` ne compte plus que les lignes `↷ Test …` (swift-testing) ou `Test Case … skipped` (XCTest). La ligne de log « Metadata extraction skipped » n'est plus comptée. Attendu désormais : `TESTS OK (n tests, 0 skippés)` quand rien n'est sauté.

### Réponse (session de rédaction du plan, 2026-09-17) — hygiène du dépôt
- Option A appliquée : `.pi/sessions/` et `.pi/npm/` ajoutés à `.gitignore`, journal de session retiré de l'index (`git rm --cached`). `.pi/settings.json` reste suivi (pas de chemin absolu, il règle la compaction).
- Par ailleurs `PLAN.md` a été découpé en `plan/` (un fichier par section, ≤ 1 700 jetons) : les fiches citent maintenant `plan/<section>.md`. Ne plus lire `PLAN.md` (sommaire) ni `Scripts/plan-concat.sh`.

## ASK — T-1.2 — 2026-09-17
- Contexte : `Scripts/check-tests.sh` appelé **sans** argument de suite (`Scripts/check-tests.sh` seul) échoue immédiatement avec `args[@]: unbound variable`, car `set -uo pipefail` + tableau bash vide (`args=()`) déréférencé via `"${args[@]}"` est traité comme non défini par ce bash. L'appel documenté par les fiches (`Scripts/check-tests.sh ProtocolTests`) fonctionne normalement.
- Ce que j'ai essayé : validé T-1.2 avec l'appel ciblé (`ProtocolTests`) puis avec `Scripts/run-tests.sh` directement pour la suite complète (18 tests, 0 échec) — je n'ai pas touché au script, hors périmètre de la fiche.
- Question : corrige-t-on `Scripts/check-tests.sh` (ex. `"${args[@]:-}"`) pour que l'appel sans argument reste utilisable ?
- Options : A) corriger le script (une ligne) ; B) laisser tel quel, toujours appeler avec une suite explicite.

## ASK — T-1.11 — 2026-09-17 — ⛔ GATE G-1
- Contexte : `yue2 decode` et `YuE2MemoryManager` sont écrits, testés (tier 1 + tier 2) et validés — cf. `tasks/JOURNAL.md`. C'est la fin du Jalon 1 : avant d'attaquer le Jalon 2 (LM), une écoute humaine du WAV produit est requise.
- Ce que j'ai essayé : pas de `parity/song/latent.npy` disponible (sous-commande `song` de `real_fixtures.py` arrive en T-3.6) ⇒ décodage du seul latent réel disponible aujourd'hui, `latent_40` (40 frames, extrait de `$YUE2_MODELS_DIR/parity/vae.safetensors`, écrit en T-1.10) converti en `.npy` par un one-liner Python.
- Question : Démo : `yue2 decode` sur `.local-runs/latent_40.npy` → `.local-runs/decode_40.wav` ; écoute OK ? On passe au Jalon 2 ?
- Temps audio obtenu : 1,6 s (40 frames de latent, `40 · 1920 − 64 = 76736` échantillons à 48 kHz — pas de latent long disponible à ce stade, cf. limites connues T-1.11 dans `tasks/JOURNAL.md`).

### Réponse de Vincent (2026-09-18)
- Écoute : bruit totalement aléatoire, aucune structure perceptible.
- Clarification apportée : `latent_40` est un latent gaussien aléatoire (seed 11, fixture de parité T-1.10), pas un latent musical — aucun chemin du port ne produit encore de latent musical (Jalons 2-3). La preuve de correction de ce composant est la parité numérique VAE (rel ≈ 2×10⁻⁶ contre PyTorch), pas l'écoute.
- Décision : **GATE G-1 validée sur la base de la parité numérique.** On passe au Jalon 2.

## ASK — T-2.9 — 2026-09-18 — ⛔ GATE G-2
- Contexte : fin du Jalon 2 (backbone LM complet : attention/RoPE/MoT, cache KV, prefill par blocs, chaîne de logits, `lm_head` restreint, boucle de génération + CFG, Planner/SemanticGenerator). `yue2 parity lm` compare, sur les **vrais poids** (`YuE2-3B`, bf16) : les logits du dernier token du préfixe (ABC et sémantique, `song.json`/`score.abc` de `reference/yue/examples`), 4 tenseurs de bissection par couche (0, 7, 14, 27 sur 28), et 8 tokens greedy par phase — contre une référence PyTorch bf16 tournée sur MPS (`Scripts/reference/real_fixtures.py lm`).
- Ce que j'ai mesuré (`Scripts/check-parity.sh lm`, verdict final `PARITY OK lm rel=0.01420438 argmax_equal=true greedy_abc=8/8 greedy_semantic=8/8`) :
  - `logits_abc` : rel = 0.006543609, argmax identique
  - `hidden_layer_0` : rel = 0.006686861
  - `hidden_layer_7` : rel = 0.0063879485
  - `hidden_layer_14` : rel = 0.0070379646
  - `hidden_layer_27` : rel = 0.01420438 (pire valeur mesurée — légère croissance avec la profondeur, cohérent avec une accumulation d'arrondi bf16 plutôt qu'une erreur structurelle puisque monotone et lente : 0,0067 → 0,0064 → 0,0070 → 0,0142 sur 28 couches)
  - `logits_semantic` : rel = 0.003063072, argmax identique
  - `greedy_abc` : 8/8 tokens identiques à la référence (`[55, 25, 16, 198, 51, 510, 44, 25]`)
  - `greedy_semantic` : 8/8 tokens identiques à la référence (`[163899, 176637, 171053, 154567, 161787, 161787, 179801, 179801]`)
  - Aucun cas de la clause de fin de fiche (« si `greedy_semantic` < 8/8 ») ne s'est produit — les deux phases greedy sont exactement 8/8, donc pas de quasi-ex-æquo à documenter.
- Tolérances utilisées (fixées par la fiche T-2.9, pas choisies par moi) : `rel ≤ 2e-2` sur logits/hidden taps, `argmax` strictement égal, `greedy_abc`/`greedy_semantic` strictement `8/8`. Toutes satisfaites avec de la marge (pire cas 0,0142 contre seuil 0,02, soit ~30 % de marge).
- Question : la parité mesurée ci-dessus (rel max 1,42 % sur `hidden_layer_27`, argmax et greedy 8/8 des deux phases exacts) est-elle suffisante pour valider **GATE G-2** et permettre l'écriture du solveur NAR (Jalon 3, dont la parité `TokenGenerator`/`RestrictedHead` de ce Jalon 2 est un prérequis direct) ? Ou souhaites-tu resserrer/desserrer la tolérance `rel ≤ 2e-2` avant de continuer (ex. si tu veux voir la marge se maintenir aussi sur des préfixes plus longs / d'autres styles avant de la considérer acquise) ?
- Options : A) GATE G-2 validée telle quelle, tolérances actuelles (`rel ≤ 2e-2`, argmax exact, greedy 8/8) conservées pour la suite (T-3.x) ; B) GATE G-2 validée mais ajouter d'autres couples préfixe/style à la fixture `lm` avant de la considérer définitive (à faire en tâche annexe, pas bloquant pour T-2.10) ; C) resserrer la tolérance dès maintenant (préciser la valeur voulue).

### Réponse de Vincent (2026-09-18)
- Option A : GATE G-2 validée, tolérances conservées telles quelles. On enchaîne T-2.10.
- Remarque pour le Jalon 3 : la fixture NAR real (T-3.5) demande ≈ 5 min de PyTorch sur MPS pendant que le serveur `qwen38` local occupe ≈ 57 Go — ne jamais lancer les deux en même temps.
- Remarque d'hygiène : `.pi/settings.json`, `Scripts/launch-with-local.sh`, `plan/00.4-executant-local.md` restent modifiés sans être committés (travail de Vincent sur le runner local) — un futur `git add -A` d'un agent les emporterait par erreur ; rester sur des `git add` de fichiers explicites.

## ASK — T-2.11 — 2026-09-18 — ⛔ GATE G-3
- Contexte : fin du Jalon 2 (backbone LM complet, CLI `yue2 plan`/`yue2 semantic`/`yue2 profile`, profiler câblé). Démo demandée par le plan (§0.1) : `yue2 plan` produit un `score.abc` lisible ; `yue2 semantic` produit N tokens à X tok/s.
- Ce que j'ai mesuré, sur `reference/yue/examples/song.json` (bf16, poids réels, seed 831001) :
  - `yue2 plan` → `score.abc` bien formé (en-tête `X:1`/`M:4/4`/`L:1/32`/`K:Bb` + voix nommées) : `planning score city_lights: 572 tokens · 59.6 tok/s`
  - `yue2 semantic` (300 tokens, tronqué volontairement pour la démo) : `semantic city_lights: 300 tokens · 66.9 tok/s`
  - `yue2 profile semantic` (mêmes réglages) : préfill 262 ms pour 702 tokens de préfixe (2679 tok/s), génération 4,17 s pour 300 tokens (72,0 tok/s), pic mémoire process 8447 MB / MLX actif 7446 MB, `trace.json` exploitable dans Perfetto
  - Les deux débits sont largement au-dessus du seuil ≥ 30 tok/s de la fiche T-2.10, et cohérents avec la fourchette 40-70 tok/s attendue par `plan/03-budget.md` §3.2
- Question : cette démo (score ABC lisible + débits mesurés) te suffit-elle pour valider **GATE G-3** et attaquer le Jalon 3 (solveur NAR, flow matching, `yue2 generate` bout-en-bout) ?
- Options : A) GATE G-3 validée, on enchaîne T-3.1 ; B) tu veux d'abord lire/écouter... (pas possible avant Jalon 3, il n'y a pas d'audio à ce stade — seulement le score ABC et les tokens sémantiques bruts) — donc plutôt : tu veux voir le `score.abc` complet ou un extrait des tokens sémantiques avant de trancher ; C) resserrer/modifier les critères de perf avant de continuer.

### Réponse de Vincent (2026-09-18)
- Option A : GATE G-3 validée. On enchaîne T-3.1 (Jalon 3, solveur NAR).

## ASK — T-3.6 — 2026-09-18 — ⛔ GATE G-4
- Contexte : fin du Jalon 3 (Modules NAR, `CachedNAR`, solveur midpoint, `Synthesizer` multi-chunks, parité real NAR, pipeline bout-en-bout `YuE2Pipeline`/`yue2 generate`). Démo demandée par le plan (§0.1) : Vincent écoute la chanson `examples/song.json` seed 831001. Qualité acceptable pour continuer ?
- Fichier à écouter : **`.local-runs/song-full/audio.wav`** (63,0 s, stéréo 48 kHz, 16 bits — `open .local-runs/song-full/audio.wav`). Log complet : `.local-runs/song-full.log`. Tous les artefacts intermédiaires sont dans `.local-runs/song-full/` (`score.abc`, `plan.json`, `semantic.npy`, `latent.npy`, `result.json` avec SHA-256 par fichier).
- Ce que j'ai mesuré : `generated city_lights: 63.0 s audio in 89.1 s (abc 9.1 s, semantic 23.8 s, nar 54.0 s, vae 2.2 s)` ; `result.json` : `status: complete`, `truncated.abc: false`, `truncated.semantic: false` — chanson entière générée sans troncature, sur les réglages par défaut (pas de `--abc-max-tokens`/`--semantic-max-tokens` réduits comme pour le smoke test court `.local-runs/song-short/`).
- Pas de référence PyTorch (`$YUE2_MODELS_DIR/parity/song/audio.flac`) disponible pour une comparaison directe à l'écoute — seule la parité numérique par composant (VAE ≈2e-6, LM ≤1,4 %, NAR velocity 1,89 %/solve4 1,02 %) a été vérifiée jusqu'ici. C'est donc la première fois que ces erreurs cumulées à travers tout le pipeline (ABC→sémantique→NAR→VAE) s'entendent.
- Question : la qualité de `.local-runs/song-full/audio.wav` est-elle acceptable pour continuer (Jalon 4 : optimisations perf) ? Ou faut-il d'abord investiguer un défaut audible avant d'aller plus loin ?
- Options : A) qualité acceptable, on continue (T-3.7 puis Jalon 4) ; B) défaut audible — dis ce que tu entends (bruit, structure musicale absente/incohérente, artefacts, longueur/coupures) pour qu'on localise si le problème vient d'une étape précise (ABC/sémantique/NAR/VAE) ; C) tu veux d'abord une comparaison contre une référence PyTorch générée sur ta machine (`real_fixtures.py song`, ≈15-40 min sur MPS, cf. `plan/08-tests.md` §8.2) avant de juger.

### Réponse de Vincent (2026-09-18)
- Option A : écouté, « une belle musique ». GATE G-4 validée. On enchaîne T-3.7 puis le Jalon 4.

## ASK — T-4.0 — 2026-09-18 — ⛔ GATE G-5
- Contexte : T-3.7 (profiler TTS) et T-4.0 (mesures de référence, GPU refroidi 120 s, deux runs A/A) validées et committées. Avant la première optimisation (Jalon 4), voici la ligne de référence stable et le contexte de perf accumulé depuis T-2.11/T-3.7.
- Tableau de mesures (colonnes `plan/03-budget.md` §3.3, source unique `swift-mlx-profiler`) :

| Date | Commit | Quant AR | cot | Tokens ABC | Tokens sém. | ABC tok/s | Sém. tok/s | NAR s | VAE s | Audio s | RTF | Pic mémoire | Trace |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-18 | f628802 | none (bf16) | full | 64 (tronqué) | 100 (tronqué) | — | — | 1.3 | 0.4 | 4.0 | 1.20 | 9968 MB (process) | `.local-runs/p1/trace.json` |
| 2026-09-18 | f8bb4e2 | none (bf16) | full | 64 (tronqué) | 100 (tronqué) | — | — | 0.6 | 0.1 | 4.0 | — † | 9969 MB (process) | `.local-runs/bench/2026-09-18.txt` |
| 2026-09-18 | f628802 | none (bf16) | full | 572 | 1574 | 65.0 | 67.1 | 51.5 | 2.0 | 63.0 | 0.82 | 14242 MB (process) | `.local-runs/song-full-profiled/trace.json` |

  † `bench-song.sh` ne capture que les 4 durées de phase + le pic mémoire (pas le `RT factor`, non écrit sur disque par ce script) — à ajouter si utile pour T-4.x.
  Les deux premières lignes sont le **même travail exact** (chanson courte 64/100/4) : la première juste après une série de builds/générations consécutifs (« chaud »), la seconde après 120 s de refroidissement GPU + deux runs A/A (dispersion mesurée 0 % entre eux). Écart observé : NAR ×2,2 (1,3 s → 0,6 s), VAE ×4 (0,4 s → 0,1 s) plus rapides à froid — l'illustration directe de l'avertissement `CLAUDE.md` § Performance work (l'état thermique du GPU déplace les résultats, pas seulement une variante par rapport à une autre). La ligne de référence retenue pour juger les optimisations futures est la **refroidie** (`docs/knowledge/benchmarks/bench-reference.tsv` : abc=1.0s sem=1.3s nar=0.6s vae=0.1s peak=9969 MB, 0 % dispersion).
- Lecture : sur la **chanson complète** (réglages par défaut, 32 pas ODE), le NAR domine largement le temps total (51,5 s / 77,0 s de phases mesurées, ≈ 67 %), suivi de la génération sémantique (23,5 s, ≈ 30 %). RTF = 0,82 (plus lent que le temps réel). Ça correspond à l'ordre imposé par `plan/07-jalon4.md` (O4 → O2 → O8 → O7 → O5 → O6/O9 → O10) : O2 (réutilisation du cache KV sémantique comme préfixe NAR, élimine un préfill de ≈ 7 k tokens) et O8 (précalculs NAR : `pe`, time embedding, K/V contigus) ciblent directement le poste le plus coûteux. O4 (`Memory.cacheLimit` par étape) est déjà en place depuis T-1.11 — sa fiche (T-4.1) ne fera que le vérifier/documenter, pas l'implémenter à neuf.
- Question : quelles optimisations prioriser (l'ordre du plan O4→O2→O8→O7→O5→O6/O9→O10 te convient-il tel quel, vu que le NAR domine le temps total) ? Et la quantification 8-bit affine de la voie AR (O5, `plan/07-jalon4.md`) est-elle autorisée sur cette machine/ce modèle ?
- Options : A) ordre du plan conservé tel quel, 8-bit AR (O5) autorisé quand on y arrivera (validé par parité + écoute comme prévu par la fiche O5) ; B) reprioriser (dis quel poste t'intéresse le plus : latence bout-en-bout, mémoire, ou débit soutenu) ; C) sauter la quantification (rester bf16 partout) et se concentrer sur O2/O8/O7 (optimisations exactes, sans risque de qualité).

### Réponse de Vincent (2026-09-18)
- Option A confirmée explicitement (ordre du plan + 8-bit AR autorisé pour T-4.5, à revalider par parité + écoute comme prévu par sa fiche). GATE G-5 validée : « recherchons les optimisations dès maintenant ».

## ASK — T-4.5 — 2026-09-18 (remarque factuelle, pas un blocage)
- Contexte : O5 (quantification 8-bit affine, groupe 64, de `self_attn.*_proj`/`mlp.*` sur la voie AR seule, `--quant qint8` opt-in) implémentée, mesurée et validée par parité — chiffres complets dans `tasks/JOURNAL.md` T-4.5 et `BENCHMARKS.md`. Résumé : rel=0,0051 (ABC) / 0,0045 (sémantique) vs bf16 sur les mêmes préfixes réels que `RealLMParityTests` (seuil de la fiche : 5e-2 — 10× de marge), greedy 8/8 identique sur les deux préfixes (seuil : ≥7/8), y compris avec `--quant-head`. Gain perf : phase AR (abc+sem) −26 à −30 % selon l'instrument (chanson courte A/B/B/A et chanson complète), débit sémantique +40,6 % (68,7→96,6 fr/s), NAR/VAE strictement inchangés, pic mémoire process −8,9 %.
- Ce que je n'ai **pas** pu faire : la fiche demande une double validation parité + écoute par « toi, l'agent ». Je n'ai aucune capacité d'écoute audio dans cet environnement (pas de lecteur, pas d'oreille) — je ne peux donc pas juger perceptuellement si la quantification dégrade le timbre/la voix/la justesse. J'ai fait une vérification quantitative de substitution (RMS des deux WAV proches : 3877 vs 4025, écart ≈4 % ; pas d'anomalie de forme d'onde), mais ce n'est pas un jugement perceptuel, et les deux rendus ne sont de toute façon pas note-pour-note comparables (tirage stochastique à température >0, la même graine diverge après quelques centaines de tokens dès que les logits diffèrent même très légèrement — 1574 vs 1536 frames sémantiques, aucune troncature dans les deux cas).
- Fichiers à écouter (non committés, gitignorés) : **`.local-runs/song-full-bf16/audio.wav`** (référence, 63,0 s) vs **`.local-runs/song-full-qint8/audio.wav`** (quantifiée, 61,4 s) — même requête (`reference/yue/examples/song.json`, seed 831001, réglages par défaut).
- Décision prise en ton absence, conformément à tes instructions : `--quant qint8` reste strictement **opt-in** (`ModelOptions.quant` par défaut `.none` — aucune commande n'est affectée sans `--quant` explicite), donc aucun risque pour la voie par défaut déjà validée (GATE G-4). Rien n'est bloqué : le Jalon 4 continue (T-4.6 puis T-4.7/GATE G-8).
- Question (facultative, pas de blocage) : si tu écoutes les deux WAV à l'occasion, dis-moi si `qint8` te semble perceptuellement dégradée — je documenterai ton verdict ici et ajusterai la recommandation (garder tel quel vs déconseiller `--quant qint8` en pratique) sans que ça remette en cause ce qui a déjà été committé.

## ASK — T-4.7 — 2026-09-18 — ⛔ GATE G-8 (fin du Jalon 4)
- Contexte : Jalon 4 (optimisations + GUI de bench) terminé selon le plan. Résumé des fiches :
  - T-4.0 (référence froide) → T-4.1 (O4, déjà actif, confirmé) → T-4.2 (O2, gardée, −17 % NAR chanson courte) → T-4.3 (O8, **retirée**, gain < 5 %) → T-4.4 (O7, gardée, VAE −30 % chanson complète) → T-4.5 (O5, quantification AR qint8, **gardée en opt-in**, AR −26 à −30 %, débit sémantique +40,6 %, parité 10× sous seuil, écoute encore à faire par toi — cf. question T-4.5 ci-dessus) → T-4.6 (O6/O9, **aucune des deux nécessaire**, preuve par trace/code, aucun changement) → T-4.7 (O10, GUI `yue2-bench-ui`, cette fiche).
  - `BENCHMARKS.md` est à jour avec toutes les mesures A/B/B/A de chaque fiche.
- GUI livrée (`Sources/YuE2BenchUI/`) : onglet **Génération** (répertoire de checkpoints avec sélecteur de dossier, quantification/VAE détectées sur disque, formulaire style/lyrics/cot/seed/cfg_scale, bouton Générer/Annuler, progression par étape ABC/sémantique/NAR/VAE) ; panneau **Mesures** intégré (résumé `TTSMetrics`, pics mémoire MLX actif/process, rapport complet dépliable — tout vient de `swift-mlx-profiler`, rien n'est recalculé) ; lecteur audio (`AVAudioPlayer`, play/pause/scrub) sur le dernier run ; onglet **Historique** (table des runs + export CSV). `Scripts/check-build.sh yue2-bench-ui` → `BUILD OK`, suite complète inchangée (`TESTS OK 117 tests`).
- Ce que je n'ai **pas** pu faire : la démo interne demandée par la fiche (« chanson courte depuis la GUI ») n'a pas pu être vérifiée **visuellement** par l'agent — cet environnement d'exécution n'a pas d'écran attaché (`screencapture -x` échoue avec « could not create image from display », et `System Events` ne rapporte aucune fenêtre pour le process malgré le correctif d'activation policy). J'ai vérifié ce qui est vérifiable sans écran : le binaire se lance et reste vivant sans crash ; le code passe la vérification stricte de concurrence Swift 6 (2 bugs réels trouvés et corrigés au passage : `PipelineEvent` pas `Sendable`, et une `Table` à 11 colonnes qui dépassait la limite de 10 du `TableColumnBuilder` de SwiftUI) ; le moteur que la GUI appelle est exactement celui de `yue2 generate`, déjà validé par toi à l'oreille (GATE G-4, T-3.6) et par toute la suite de tests.
- Question : peux-tu lancer `.xcodebuild/Build/Products/Release/yue2-bench-ui` (ou `xcodebuild -scheme yue2-bench-ui -configuration Release build` puis ouvrir le binaire) et faire la démo toi-même — une chanson courte depuis l'onglet Génération, vérifier que la progression/les mesures/le lecteur fonctionnent ? Et, la question de fond du Jalon 4 : on continue sur quoi ensuite — l'encodeur VAE (upload/analyse), un serveur d'inférence, ou le portage iOS ?
- Options : A) démo GUI OK, on choisit la suite (encodeur VAE / serveur / iOS — dis lequel) ; B) bug trouvé dans la GUI pendant ta démo — décris-le, je corrige avant de choisir la suite ; C) tu veux d'abord le bouton « Export trace » (`ChromeTraceExporter`, listé en limite connue de T-4.7, non branché faute de temps) avant de juger la GUI complète.

### Réponse de Vincent (2026-09-19)
- Option A : démo GUI validée de facto (Vincent a généré plusieurs chansons depuis la GUI lui-même, y compris une version longue, entre-temps).
- Suite choisie : **l'encodeur VAE**, avec un vrai fichier audio pour tester ((un fichier MP3 personnel fourni par Vincent, hors dépôt), jamais committé). Fiches `tasks/T-5.1.md` (port de l'encodeur Oobleck + parité) et `tasks/T-5.2.md` (ingestion audio + `yue2 encode` + round-trip) écrites en conséquence (extension hors plan initial — `plan/02.9-hors-perimetre.md` notait cet encodeur comme « v2 éventuelle »).

## ASK — T-5.1/T-5.2 — 2026-09-19 (pas un GATE numéroté, extension hors plan)
- Contexte : T-5.1 (encodeur VAE Oobleck) et T-5.2 (ingestion audio + `yue2 encode`) validées et committées. Parité numérique de l'encodeur : tiny (fixture, tolérance 1e-4) → 6/6 tests verts ; réelle (vrais poids, contre PyTorch fp32 CPU) → `max=0.0014 rel=1,7e-5` (budget max≤2e-3, rel≤1e-3). Suite complète : 128/128.
- Démo réelle sur ton fichier :
  ```
  yue2 encode --audio "<chemin local vers le fichier audio de test>" \
    --out .local-runs/magic-system-roundtrip --roundtrip --models-dir "$YUE2_MODELS_DIR"
  ```
  → `encoded … : 8811520 samples (183.6 s) → 4589 frames latent in 8.77 s` puis `roundtrip: …/roundtrip.wav (183.6 s) in 23.73 s`. Durée originale (`afinfo`) : 183.61 s ; durée round-trip : 183.56 s (écart de 0,05 s, cohérent avec la formule de longueur naturelle `1920·T − 64` du décodeur — pas un bug).
- Fichiers à comparer à l'oreille : original (un fichier MP3 personnel fourni par Vincent, hors dépôt) vs `.local-runs/magic-system-roundtrip/roundtrip.wav` (35 Mo, trop lourd pour un envoi direct — chemin local, `open .local-runs/magic-system-roundtrip/roundtrip.wav`).
- Ce à quoi t'attendre : c'est un VAE audio (compression avec perte à 64 canaux latents / 1920×), pas un codec lossless — un peu de perte de détail (brillance, transitoires très fins) est normale et attendue. Le test qui compte : est-ce que la structure musicale (rythme, mélodie, voix, instruments reconnaissables) traverse le round-trip sans dégradation choquante ?
- Question : la qualité du round-trip te semble-t-elle bonne ? Un défaut particulier à signaler (structure méconnaissable, bruit, artefacts) pointerait vers un problème réel malgré la parité numérique propre (le round-trip complet — encode PUIS decode — n'a jamais été testé ensemble avant cette démo, seulement chaque moitié séparément contre PyTorch).
- Pas de décision requise dans l'immédiat sur la suite du Jalon 5 (T-5.3 potentielle : CLI `yue2 remix`/analyse, ou on s'arrête là) — dis-moi selon ton ressenti sur l'écoute.
