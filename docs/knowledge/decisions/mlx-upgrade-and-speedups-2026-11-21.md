# Analyse — montée mlx-swift 0.31.6 → 0.33.0 et pistes d'accélération (2026-11-21)

Analyse documentaire et compilation seulement : **aucune mesure de performance n'a été faite**. Les chiffres marqués [estimé] sont des ordres de grandeur, pas des mesures.

## 1. Les versions

| mlx-swift | Date | Cœur mlx embarqué | Note |
|---|---|---|---|
| 0.31.6 (épinglée) | 2026-07-02 | v0.31.1 | |
| 0.32.2 | 2026-09-28 | v0.32.2 | **à éviter** : plantage au lancement sur OS < 26.4 (#491) |
| 0.32.3 | 2026-09-30 | v0.32.2 | gemma-4 et haru y sont depuis le 2026-10-03 |
| 0.33.0 (dernière) | 2026-10-27 | v0.32.4 | cible proposée |

Les outils et les OS minimaux sont inchangés (swift-tools 6.3, macOS 14, iOS 17).

## 2. Ce que la montée apporte à YuE2 (vérifié à la source)

- **Erreur GPU sans `std::terminate`** (mlx #3523, cœur v0.32.0+). Dans `device.cpp` v0.32.4, le gestionnaire de complétion ne lève plus d'exception. Il range l'erreur dans `error_` et empoisonne les événements. `error_.check()` la relève ensuite sur le fil appelant (`synchronize`, encodeur suivant).
  - **Ce n'est pas encore la fermeture de Q7/Q11.** Le gestionnaire par défaut de mlx-swift finit en `fatalError`, et `Sources/` n'appelle `withError` nulle part. Il faudra aussi :
    - envelopper les `eval` des étapes dans `withError` ;
    - accepter que l'erreur sorte avec une évaluation de retard, et reprendre au checkpoint NAR ;
    - attendre #516 (`quantized()` qui trappe sous `withError`), corrigé sur `main` après la 0.33.0.
  - À tester sur l'iPhone (bascule en arrière-plan pendant le NAR) avant de retirer quoi que ce soit de la porte GPU.
- **Deadlock `compile` × `vjp` corrigé** (mlx-swift #461, 0.32.2). C'est la raison de la largeur 1 de `Scripts/run-tests.sh`. À revérifier avant d'élargir.
- **Noyaux** (titres des PR, pas de mesure de notre côté) :
  - VAE : conv transposée par tuiles (#3845), conv 2D et petits canaux (#3904, #4617) ;
  - qmv et split-K quantifiés (#3120, #3764, #4516, #4641) ;
  - SDPA vecteur GQA (#4077, #4380, #4596) ;
  - GEMM fins (#4640, #4654).
  - Rien pour les M5 (accélérateurs NAX), qui ne concernent ni le M3 Max ni l'A17 Pro.
- **Mémoire : peu de gain attendu.**
  - Le jeu de résidence Metal est scindé (#4211), ce qui corrige un recâblage continu sous macOS 27.
  - Les pages sont dé-câblées après environ 2 s d'inactivité GPU sous macOS 27 (#4633, réglable par `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`). C'est un **piège de mesure** avec nos refroidissements de 120 s : le premier pas après refroidissement repaie le câblage.
  - `Memory.bufferSize(_:)` est nouveau ; `cacheLimit`, `memoryLimit` et la limite câblée sont inchangés.
  - Le pic reste fait des poids (le NAR 8 bits pèse 1,43 Go dans les profils lean).

## 3. Risques

- **mlx-swift #521 (ouverte)** : avec les streams en pool, des opérations internes sans stream (casts dans `quantized_matmul`) passent par une deuxième file. Le décodage gpt-oss y perd environ 30 % sur M3 Ultra.
  - Le correctif (mlx #4746, fusionné le 2026-10-25) n'est embarqué dans aucune release.
  - Nos chemins exposés : l'AR quantifié, `RestrictedHead` (`quantizedMatmul`) et le NAR packé des profils lean. **C'est donc d'abord le profil de l'app qui est concerné.**
  - Critère : A/B tok/s AR et temps par évaluation NAR en `4bit-lean` et `8bit-lean`, protocole de CLAUDE.md. Si la régression se confirme, attendre la release qui embarque #4746 (pas de `branch:`).
- **Numérique** :
  - la dé-quantification passe en fp32 (#4241) et les partiels split-K aussi (#4641) ;
  - les noyaux bf16 sont recompilés (piège haru `metal-kernels-differ-between-builds`).
  - Conséquence : à graine égale, les chansons ne seront plus les mêmes.
  - La porte qualité doit inclure les parités (top-1, tolérances) et le banc des reprises en 4 bits (vers perdu après un silence, 2026-10-20), pas seulement les tests bit-exacts Swift↔Swift.
  - Un checkpoint NAR sauvé en 0.31.6 et repris en 0.33.0 reste valide, mais n'est plus bit-exact.
- **Ruptures de compilation** (worktree jetable épinglé en 0.33.0, `swift build --build-tests`, 2026-11-21) :
  - **une seule erreur** : `Generation/CompiledDecodeStep.swift` l. 36, 38, 46 et 47, captures de `DecoderLayer` et `YuE2Attention` non-`Sendable` dans les closures `compile`, devenues `@Sendable` (#469) ;
  - ce chemin est désactivé partout (aucun gain mesuré le 2026-09-26) : le retirer ou passer les poids en entrées ;
  - `SnakeBeta` (`compile(shapeless:)` sur une méthode) n'a pas été signalé ;
  - les tests ne sont pas encore compilés, puisque le module s'arrête à cette erreur.

## 4. Pistes d'accélération venues des implémentations voisines

Ports examinés :
- **audio.cpp** (0xShug0, ggml, environ 4,4 k étoiles) : YuE2 par défaut à **8 pas midpoint**, branche `preview/yue2-optimizations` avec une démo iOS (Q4_0, KV Q8_0, VAE f16, tuiles réduites) ;
- yue2.cpp (ServeurpersoCom) ;
- YuE2-Turbo (vLLM) ;
- mlx-Yue d'ianiv (branche `perf`) ;
- YuE2Mac, OpenSuno, YuE2-Studio ;
- Riff (sur le moteur Metal d'audio.cpp).

Ordre de gain attendu :

1. **Moins d'évaluations NAR.** Le NAR représente environ 70 % du temps, et ses noyaux tournent déjà vers environ 60 % du pic bf16 [estimé : ≈ 6,6 TFLOP par évaluation à 1 752 frames, 64 évaluations ≈ 420 TFLOP en 50 s]. Les noyaux ne donneront donc pas ×2 ; le levier est le nombre d'évaluations.
   - **Adams–Bashforth 2** (Riff, YuE2-Studio en option) :
     - fonctionnement : midpoint au premier pas, puis une évaluation par pas qui réutilise la vitesse précédente, soit S + 1 évaluations au lieu de 2S ;
     - Riff (M4, 16 pas) : 32 → 17 évaluations, graphe acoustique 37,3 → 19,7 s ; AB2 à 32 pas = 33 évaluations, 50,0 s ;
     - écoute aveugle par un LLM, deux études, « pas de différence nette » : faible preuve.
   - **Ce n'est pas un gain de vitesse nouveau.** AB2-32 coûte autant que midpoint-16 (sondé le 2026-09-26 : NAR 24,8 s). La vraie question est la qualité à coût égal, AB2-32 contre midpoint-16, à l'écoute et par transcription (Voxtral/Whisper).
   - Coûts d'intégration :
     - `NARCheckpoint` doit porter la vitesse précédente, ce qui change le format persisté par l'app et le contrat de reprise au bit près ;
     - `.keep` réimpose des trames à chaque pas, ce qui casse l'extrapolation (garder midpoint pour les éditions) ;
     - tester la stabilité en bf16 sur les derniers pas en `logit(t)` ;
     - ajouter un test d'EDO analytique (cf. `check_native_solver.py` de Riff).
   - **Brouillon puis finition** (YuE2Mac ; ianiv « Fast » 8 pas, NAR d'une chanson de 3 min 38,5 → 11,1 s sur M5 Max) : même graine, mêmes jetons sémantiques, rendu définitif à 32 pas. Cela correspond à ce que l'app sauvegarde déjà.
2. **Décalage du calendrier en t** (ACE-Step : shift 3), plus de pas du côté bruité. À balayer une fois le solveur choisi.
3. **Saut de pas par cache** (TeaCache, First-Block Cache, AudioCache) : 1,5–2× sur les DiT vidéo. Notre temps est un embedding additif, sans adaLN, donc l'indicateur est à recalibrer. Le gain recoupe en partie celui d'AB2.
4. **AR** (13–16 s, environ 40 % du total une fois le NAR réduit) :
   - ianiv passe de 122 à 175 tok/s (M5 Max) avec un décodage pipeliné, la fenêtre de pénalité gardée sur l'appareil, RMSNorm/RoPE/SwiGLU fusionnés aux mêmes arrondis et des tables cos/sin partagées. Nous avons déjà le pipeline (`asyncEval`) : reste à vérifier le travail CPU par jeton.
   - Le lookahead de prompt (mesures ABC répétées) ne joue que sur la phase ABC, environ 3,5 s.
   - Le CFG en lot ne vaut que pour `cot=off` et l'instrumental.
5. **Mémoire iPhone** :
   - KV en Q8 : 28 couches × 2 × 8 têtes × 128 × ≈ 4 030 positions × 2 octets ≈ 0,46 Go en bf16, soit environ −0,23 Go en Q8. C'est utile pour `4bit-tiny`, sans gain de temps.
   - VAE : SnakeBeta fusionné (déjà `compile`) ; conv transposée en GEMM + col2im, +1,7 à 6,6 % chez audio.cpp.
6. **Hors de portée sans entraînement** : distillation de pas (ACE-Step 1.5 turbo, Presto).
   - SageAttention n'a pas de portage Metal, et l'attention ne pèse qu'environ 25 % des FLOP du NAR.

## 5. Plan proposé, mesures séparées

- **M-1** : 0.33.0 dans une branche.
  - Retirer ou réécrire `CompiledDecodeStep`.
  - Faire passer les deux paliers de tests, avec les parités jugées au top-1 et aux tolérances.
- **M-2** : A/B 0.31.6 / 0.33.0 sur les sept profils de référence.
  - Machine calme : `pgrep`, refroidissement, A/B/B/A.
  - Mesurer ensemble le temps par étape et le pic.
  - Regarder en particulier les tok/s AR et le NAR packé en lean (#521).
  - Porte : aucune régression au-delà de ±2 % ; sinon attendre la release qui embarque #4746.
- **M-3** : `withError` autour des `eval` d'étape, plus une reprise au checkpoint ; test de bascule en arrière-plan sur l'iPhone (Q7).
- **M-4** (indépendant de la version) : AB2 dans `CachedNAR.solve`, en option, midpoint par défaut ; écoute AB2-32 / midpoint-16 / midpoint-32, plus transcription.
- **M-5** : décalage du calendrier, puis cache de pas.

## 6. Mesures (2026-11-21, après-midi, M3 Max sur secteur, machine calme)

### M-2 : 0.31.6 (A) contre 0.33.0 (B)
Chanson `reference/yue/examples/song.json` (city_lights, graine 831001), 1 750 jetons sémantiques imposés,
A/B/B/A par profil, 120 s de repos avant chaque run, pic = `peak memory footprint` de `/usr/bin/time -l`.
Données : `.local-runs/bench.noindex/mlx033-ab/results.tsv` (local).

| profil | sémantique tok/s A → B | NAR s A → B (banc complet) | pic Go A → B |
|---|---|---|---|
| 4bit-fast | 162,9 / 150,7 → 155,0 (B2 écarté, WindowServer 45 %) | 51,8 / 53,4 → 57,6 | 8,96 → 8,88 |
| 4bit-lean | 153,0 / 159,9 → 148,7 / 158,8 | 61,8 / 62,6 → 65,4 / 62,2 | 3,40 → 3,45 |
| 8bit-fast | 111,5 / 111,1 → 109,6 / 110,7 | 56,0 / 54,9 → 56,3 / 56,4 | 9,96 → 10,05 |
| 8bit-lean | 112,3 / 112,4 → 110,7 / 112,1 | 63,7 / 58,1 → 58,8 / 58,7 | 4,38 → 4,29 |
| 16bit-fast | 75,6 / 75,6 → 74,2 / 74,2 | 52,3 / 55,1 → 55,3 / 56,9 | 11,99 → 12,07 |

- **Décodage AR** : −0,5 à −2 %, à la limite de la porte ±2 %. Le −30 % de #521 ne se voit pas sur nos profils.
- **Mémoire** : inchangée (±1 %).
- **Le banc complet ne compare pas le NAR** : à graine égale, la 0.33 écrit une autre partition ABC
  (4bit-fast 422 → 613 jetons, 8bit-fast 588 → 787, 16bit-fast 573 → 938 ; 4bit-lean inchangé) et le NAR attend
  sur ce préfixe. Le surcoût NAR apparent suit l'allongement du préfixe.
- **NAR à entrée identique** (même plan, mêmes jetons sémantiques, `synthesize` bf16, midpoint 32, A/B/B/A/A/B,
  `.local-runs/bench.noindex/mlx033-nar-fixed/`) : A 51,7 / 53,0 / 52,5 s, B 52,3 / 52,4 / 52,6 s → **52,4 s des
  deux côtés** ; pic 14,6 Go des deux côtés. **Latents bit à bit identiques** entre les deux versions : la voie
  NAR bf16 + VAE n'a pas bougé numériquement ; ce qui change, c'est la voie AR (plan ABC différent à graine égale).

**Verdict M-2** : la 0.33.0 est neutre en vitesse et en mémoire sur nos profils ; elle ne gagne rien en soi.
Son intérêt reste la robustesse (#3523, #461) et la suite (#4746 quand une version le portera). Les chansons à
graine fixe changent (plan ABC) : à repasser sur le banc des reprises 4 bits avant toute publication.

### M-4 : AB2 contre midpoint (même plan, mêmes jetons, bf16, binaire `nar-ab2`)
Référence : midpoint 64 pas (128 évaluations). Écart relatif RMS des latents, cosinus, et écart moyen du
log-spectre (fenêtre 4 096, en dB) de l'audio. `.local-runs/bench.noindex/ab2-listen-city/`.

| variante | évaluations | NAR s | écart latent | cosinus | log-spectre dB |
|---|---|---|---|---|---|
| midpoint-64 (réf.) | 128 | 110,0 | — | — | — |
| midpoint-32 (défaut) | 64 | 52,6 | 0,034 | 0,9994 | 2,02 |
| **ab2-32** | 33 | **27,0** | 0,044 | 0,9991 | 2,06 |
| midpoint-16 | 32 | 29,0 | 0,049 | 0,9988 | 2,10 |
| ab2-24 | 25 | 24,2 | 0,062 | 0,9981 | 2,18 |
| ab2-16 | 17 | 17,0 | 0,092 | 0,9958 | 2,35 |
| ab2-12 | 13 | 13,8 | 0,116 | 0,9933 | 2,57 |

- À coût égal, **ab2-32 est plus proche de la référence que midpoint-16** et tombe entre midpoint-16 et
  midpoint-32 : NAR −49 % contre le défaut pour un écart latent 0,044 au lieu de 0,034.
- Ces chiffres mesurent la distance à une solution fine, pas l'oreille. Écoute à faire (Vincent) :
  midpoint-32 / ab2-32 / midpoint-16, puis ab2-16 pour situer la limite.
