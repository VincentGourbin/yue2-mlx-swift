# iPhone 15 Pro Max — premier balayage réel (E5 / T-6.5 / A-0.1) — 2026-09-22

Appareil : iPhone16,2 (A17 Pro, 8 Go), iOS 27.0 (24A437), app `YuE2Studio` (yue2-ios `50117b7`) sur `YuE2Core` `a0fb9d2`. Pack `int4-mixed-head` (2,54 Go), précision fp16, profil mémoire `mobile`, `os_proc_available_memory` au départ **6 125 Mo**. Un seul passage, à froid, 2 min 16 s au total, thermique **nominale** à chaque point. JSON brut : `.local-runs/iphone/bench-2026-09-22-052823.json` (non versionné). Footprint = `FootprintSampler` (process, `phys_footprint`) ; « actifs » = `Memory.activeMemory` MLX.

## Chargement (résidence par étape)

| Point | Temps | Footprint après | Pic | Note |
|---|---|---|---|---|
| AR seul (`residency: [.ar]`) | 5,6 s | 2 962 Mo | 3 027 Mo | 2 600 Mo MLX actifs ; pic transitoire 2 921 Mo actifs pendant le chargement |
| NAR ajouté (`loadWeights(of: .nar)`) | 1,5 s | 2 951 Mo | 2 962 Mo | actifs inchangés (chargement paresseux, non évalué : mesure non concluante) ; `releaseWeights(of: .nar)` rend **1 437 Mo** |

## Voie AR (greedy, préfixe 128 tokens)

| Tokens | Prefill | TTFT | Débit | Pic footprint |
|---|---|---|---|---|
| 64 (échauffement) | 0,71 s | 0,80 s | 19,0 tok/s | 2 280 Mo |
| 512 | 0,28 s | 0,29 s | **25,1 tok/s** | 2 226 Mo |

Mac M3 Max, même préfixe : 60-67 tok/s bf16, ≈ 97 fr/s qint8 (T-4.5). L'iPhone est **≈ 2,7× plus lent** que le Mac bf16, et **2× au-dessus** de l'estimation du plan (10-15 tok/s).

## NAR (MLX qint8, une évaluation `velocity`, moyenne de 4, après préfixe réel)

| L (frames) | Audio | Préfixe (prefill, voie AR) | Pic préfixe | ms / évaluation | ms / frame | Solve 32 pas (64 évals) | Pic velocity |
|---|---|---|---|---|---|---|---|
| 256 | 10 s | 385 tokens, 1,0 s | 1 876 Mo | 665 | 2,6 | ≈ 43 s | 2 042 Mo |
| 512 | 20 s | 641 tokens, 1,5 s | 3 197 Mo | 1 294 | 2,5 | ≈ 83 s | 2 250 Mo |
| 1 024 | 41 s | 1 153 tokens, 2,7 s | 3 207 Mo | 2 776 | 2,7 | ≈ 178 s | 2 297 Mo |
| 1 536 | 61 s | 1 665 tokens, 3,9 s | **3 184 Mo** | **6 516** | **4,2** | ≈ 417 s | 2 330 Mo |

Mac (int4-mixed, profil mobile, T-6.4b) : 140 ms à L=256, 489-680 ms à L=1024 → iPhone **≈ 4,1-5,7× plus lent**. Linéaire jusqu'à 1 024 frames, puis **super-linéaire à 1 536** (+57 % par frame) — à isoler (attention quadratique, pression mémoire ou horloge GPU ; un seul passage, 4 évaluations).

Le pic du pipeline sur le téléphone est le **préfixe NAR avec la voie AR encore résidente** (3,2 Go), pas le solve (2,3 Go) ni le VAE.

## VAE (1 024 frames = 41 s d'audio, latent aléatoire, fp16, halo 16)

| Backend | Tuile | Chargement | Footprint après chargement | Décodage | Pic |
|---|---|---|---|---|---|
| MLX | 256 | 0,6 s | 702 Mo | 6,0 s | 2 186 Mo |
| MLX | 128 | — | — | 6,5 s | 1 907 Mo |
| MLX | 64 | — | — | 7,8 s | 1 454 Mo |
| Core AI GPU (`.aimodel`, spécialisé sur l'appareil) | 256 | **1,7 s** | 2 019 Mo | **5,4 s** | 2 125 Mo |

Core AI GPU : −10 % de temps, pic identique à MLX/256, 2 Go résidents dès le chargement (plan mémoire statique, comme sur le Mac) ; MLX/64 offre le pic le plus bas pour +30 % de temps. La spécialisation sur l'appareil du `.aimodel` brut a pris 1,7 s (question Q1 de l'ASK : pas bloquant). Parité non mesurée sur l'appareil (latent aléatoire, pas de référence) — 54,7 dB sur le Mac.

## Projection chanson complète (hors chargement, thermique nominale supposée)

| Étape | 30 s (750 frames) | 60 s (1 500 frames) |
|---|---|---|
| ABC (≈ 570 tokens à 25 tok/s) | ≈ 23 s | ≈ 23 s |
| Sémantique (25 tokens / s d'audio) | ≈ 30 s | ≈ 60 s |
| Préfixe NAR + échange AR → NAR | ≈ 5 s | ≈ 6 s |
| NAR 32 pas (interpolé) | ≈ 127 s | ≈ 400 s |
| VAE MLX fp16/256 | ≈ 4,5 s | ≈ 9 s |
| **Total** | **≈ 3,2 min** | **≈ 8,3 min** (≈ 5 min à 16 pas) |

Cible « nord » du plan (§13.1), d'après ce premier passage : 30 s < 4 min ✅, 60 s < 8 min ≈ (limite), pic < 3,5 Go ✅ (3,2 Go). **Corrigé par les deux passages suivants ci-dessous** : la thermique change tout.

## Deuxième et troisième passages (05:40 et 05:51, mêmes points) — thermique

| Point | Passage 1 (05:28, nominal) | Passage 2 (05:40) | Passage 3 (05:51) |
|---|---|---|---|
| AR 512 tokens | 25,1 tok/s | 25,3 | 23,9 |
| NAR L=256 | 665 ms | 658 | 657 |
| NAR L=512 | 1 294 ms | 1 337 | 1 350 |
| NAR L=1 024 | 2 776 ms (nominal) | **4 046 ms (fair)** | **3 703 ms (nominal → fair)** |
| NAR L=1 536 | 6 516 ms | 6 418 (fair) | 6 322 (fair) |
| VAE MLX/256 | 6,0 s | 5,9 | 6,0 |
| VAE Core AI/256 | 5,4 s | 5,5 | 5,0 |

`ProcessInfo.thermalState` passe à **fair** après ≈ 60-65 s de charge GPU soutenue (au point NAR L=1 024), à chaque passage. À ce moment le NAR L=1 024 perd **31-46 %** (2,8 → 3,7-4,0 s) ; le point L=1 536 (6,3-6,5 s, 4,2 ms/frame) est identique aux trois passages, donc déjà mesuré throttlé dès le premier : la « super-linéarité » du passage 1 était **thermique, pas algorithmique**. Extrapolation à froid : L=1 536 ≈ 2,6 ms/frame × 1 536 ≈ 4,0 s ; throttlé ≈ 6,4 s (**×1,6**). L'AR (borné mémoire) et le VAE (court) ne bougent pas.

## Chanson complète de 30 s (passage 2, 05:42 → 05:47, `YuE2Pipeline`, résidence par étape)

Requête `song.json` (City Lights), `cot: full`, seed 831001, sémantique plafonnée à 750 tokens, ODE 32 pas, VAE MLX fp16/256, thermique **fair** du début à la fin (enchaînée après le balayage). Résultat : **30,0 s d'audio en 273 s** (RTF 9,1), **pic process 3 601 Mo / 3 486 Mo MLX actifs**, footprint final 690 Mo. Les chronos par étape n'ont pas été consignés dans le JSON (correctif A-0.2 : une sonde par étape) ; répartition estimée d'après le balayage : AR ≈ 55 s, préfixe + échange ≈ 8 s, NAR ≈ 200 s throttlé (64 × ≈ 3 s à 750 frames), VAE ≈ 5 s. Le pic à 3,6 Go dépasse les 3,2 Go du balayage et la cible de 3,5 Go : étape non identifiée (probablement l'échange AR → NAR ou la phase sémantique avec son cache KV), à localiser avec les sondes par étape. Clip : `.local-runs/iphone/clip-30s-mlx-32pas.wav` (écoute G-10 à faire).

## Cible « nord » révisée

| Critère | Cible | Mesuré / projeté |
|---|---|---|
| 30 s | < 4 min | **4,55 min** throttlé (≈ 3,2 min si le téléphone restait nominal — il ne reste pas) ❌ |
| 60 s à 32 pas | < 8 min | ≈ 10-11 min throttlé ❌ ; ≈ 6 min à 16 pas |
| Pic mémoire | < 3,5 Go | **3,6 Go** sur le clip, 3,2 sur le balayage ❌ (de peu, étape à localiser) |
| Arrêt thermique | aucun | aucun (`fair`, jamais `serious`) ✅ |

Leviers, par ordre d'effet : pas d'ODE (16 → NAR ÷ 2), durée par défaut (30 s), thermique (le NAR tourne throttlé dès la deuxième minute quoi qu'on fasse ; à mesurer : `critical`/`serious` sur 60 s), mémoire (localiser le pic de 3,6 Go).

## Quatrième passage (06:06 → 06:15) — sondes par étape, NAR Core AI GPU

**Chanson de 30 s, par étape** (même requête, 32 pas, VAE MLX/256, thermique fair dès le début, total 273,5 s — reproductible à 0,2 % avec les 273,1 s du passage 2) :

| Étape | Durée | Footprint début → fin | Pic footprint | Pic MLX actifs |
|---|---|---|---|---|
| plan (ABC, ≈ 570 tokens) | 41,4 s | 3 143 → 3 522 Mo | **3 644 Mo** | **3 486 Mo** |
| semantic (750 tokens) | 31,2 s | 3 522 → 3 191 Mo | 3 522 Mo | 3 077 Mo |
| nar (750 frames, 64 évaluations) | 193,3 s | 3 191 → 778 Mo | 3 191 Mo | 2 945 Mo |
| vae (30 s d'audio) | 3,9 s | 776 → 854 Mo | 2 364 Mo | 1 959 Mo |

Le NAR pèse 71 % du temps (3,0 s par évaluation throttlée à 750 frames), l'AR (plan + sémantique) 27 %. **Le pic du pipeline est la phase plan** : +0,5 Go au-dessus des poids AR. Cause, lue dans `RestrictedHead.swift` : avec `--quant-head`, chaque phase rassemble les lignes de `lm_head` qu'elle peut émettre puis les **dé-quantifie en fp16** ; la phase ABC autorise `0..<eod`, soit presque tout le vocabulaire (≈ 151 k × 2 048 × 2 octets ≈ 620 Mo), donc elle reconstitue un head fp16 entier — le head int4 n'économise rien sur l'étape qui porte le pic (la sémantique, 32 k lignes, ne coûte que ≈ 135 Mo). Demande côté moteur : ASK Q6.

**NAR Core AI GPU sur l'appareil** (asset int8 `YuE2NarStack.aimodel` 1,33 Go, spécialisé sur l'appareil, cache K/V MLX converti à chaque point ; poids NAR MLX restés résidents pendant la mesure — ce n'est donc pas le scénario « mémoire contrainte », seulement la vitesse) :

| L | MLX (ms / éval) | Core AI GPU (ms / éval) | Rapport | Core AI : chargement + 4 évals | Pic footprint Core AI |
|---|---|---|---|---|---|
| 256 | 660 (nominal) | **4 352** (nominal) | 6,6× | 59,2 s | 2 438 Mo (MLX : 2 042) |
| 512 | 1 937 (fair) | **10 072** (fair) | 5,2× | 54,2 s | 2 622 Mo (MLX : 2 240) |
| 1 024 / 1 536 | 3 978 / 6 293 | refusé : `CoreAI NAR stack only supports up to 1024 AR prefix tokens (got 1153 / 1665)` | | | |

Même rapport que sur le Mac (5-7×). Un clip de 30 s passerait de ≈ 3,2 min de NAR à ≈ 16 min. Le palier de préfixe AR (1 024 tokens) exclut d'ailleurs toute chanson dont plan + sémantique dépassent 1 024 tokens, c'est-à-dire toutes au-delà de ≈ 18 s. **T-6.10 reste clos, cette fois sur mesure appareil.**

Thermique : `fair` atteint à 06:07:54, après ≈ 105 s (dont 59 s de Core AI) ; MLX L=512 passe de 1 294-1 350 (nominal) à 1 937 ms (fair), **+45 %**.

## Chanson de 60 s dans l'app (A-1, 14:05 → 14:59, reprise après crash)

| Étape | Durée | Pic footprint | Thermique |
|---|---|---|---|
| plan (ABC) | 39 s | 3 492 Mo | nominal |
| sémantique (1 500 tokens) | 60 s | 3 480 Mo | fair |
| NAR (1 500 frames, 64 évaluations, reprise : préfixe re-préfillé ≈ 2 200 tokens) | **645 s** | 3 238 Mo | serious (2 s de repos entre pas) |
| VAE (60 s d'audio) | 16 s | 2 169 Mo | serious |

Total ≈ 12,7 min hors chargements ; 60,0 s d'audio. La première tentative de C avait crashé (erreur de command buffer Metal, message inconnu) ; la reprise a abouti.
