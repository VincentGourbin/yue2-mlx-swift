# Mac M3 Max — chanson de 70 s : bf16 / 8 bits / 4 bits × MLX pur / optimisé — 2026-09-26

Même requête (`city_lights`, seed 831001), sémantique forcée à 1 750 tokens (70,0 s d'audio), 32 pas d'ODE, `--profile`, 120 s de refroidissement avant chaque run. **Un entraînement LoRA tiers (`gemma4-cli`, autre projet) occupait le GPU pendant toute la série** : les temps sont des ordres de grandeur (dispersion observée jusqu'à ±30 %), les mémoires sont exactes (footprint process, `phys_footprint`).

Modes : **MLX pur** = profil Mac (caches 2/4/1 Go), tout résident, VAE fp32 tuile 1024. **Optimisé, profil mobile** = résidence par étape, VAE fp16 tuile 256, cache 256 Mo, seuil 3 Go (réglages iPhone de la 0.2.1). **Optimisé, profil Mac** = mêmes optimisations sans les limites mobiles (isole le coût de la résidence).

| Pack | MLX pur | Optimisé, profil mobile | Optimisé, profil Mac |
|---|---|---|---|
| bf16 | 13 656 Mo · 269 s | 5 146 Mo · 471 s | 7 277 Mo · 273 s |
| 8 bits (`qint8-all --quant-head`) | 11 712 Mo · 256 s | 3 558 Mo · 324 s | 5 351 Mo · 323 s |
| 4 bits `int4-mixed-head` (NAR 8 bits) | 10 542 Mo · 227 s | **2 820 Mo** · 294 s | 4 392 Mo · 223 s |
| 4 bits `int4-all --quant-head` (non validé à l'écoute) | 10 005 / 9 987 Mo · 240 / 245 s | 2 870 Mo · 272 s | — |

Temps par phase (abc / sémantique / NAR / VAE) : bf16 pur 36/106/119/7 ; bf16 mobile 71/220/174/6 ; bf16 optimisé Mac 36/112/120/4 ; 8 bits pur 32/92/124/8 ; 8 bits mobile 49/127/142/6 ; 8 bits Mac 46/127/144/6 ; int4-mixed pur 18/77/127/6 ; mobile 28/116/146/5 ; Mac 16/67/133/7 ; int4-all pur 19/79/137/5 ; mobile 23/105/140/5.

Pics par phase (chargement / plan / sémantique / NAR / VAE), optimisé mobile : bf16 4,8/5,1/5,1/4,7/2,8 Go ; 8 bits 3,1/3,3/3,6/3,1/2,8 ; int4-mixed 2,0/2,1/2,5/2,3/2,8 ; int4-all 2,0/2,1/2,5/2,1/2,9.

## Lecture
- **Résidence + VAE fp16/256, sans les limites mobiles** : coût temps nul (bf16 +1,5 %, 4 bits −2 %), mémoire −47 % (bf16) et −58 % (4 bits). Le 8 bits fait exception (+26 % dans les deux modes optimisés, 323-324 s) sans explication à ce stade : à remesurer sur GPU libre avant de conclure.
- **Les limites mobiles de la 0.2.1 (cache 256 Mo, seuil 3 Go) coûtent du temps** : +73 % en bf16 (working set au-dessus du seuil de récupération du cache), **+32 % en 4 bits** (294 vs 223 s) alors que le working set tient sous 3 Go : c'est le cache de 256 Mo, trop petit pour les tampons d'un pas (libérés vers Metal puis réalloués). Correctif : limites calées sur la mémoire disponible (`mobileLimitsMB`, cache ≈ 1 Go, seuil = disponible − 1,25 Go), mesure ci-dessous.
- **Sous 4 Go** : 8 bits intégral 3,6 Go, 4 bits 2,8 Go ; bf16 5,1 Go, hors budget même optimisé.
- **Sur Mac, 8 bits n'est pas plus rapide que bf16** (256 vs 269 s) ; le 4 bits l'est (227 s, −15 %, surtout sur l'AR : 18 s contre 32-36 s). Les deux 4 bits se valent : `int4-all` n'apporte rien qui justifie sa perte de qualité NAR (E2).
- **CPU vs GPU** (traces propres du 21/09) : phases AR sur un seul cœur à 60-96 % avec GPU à 68-79 % ; NAR CPU 2 %, GPU 79 %. Phase sémantique isolée : ≈ 16 ms de CPU par token dont plus de la moitié en noyau (soumission Metal) ; à 10 ms/token sur Mac, l'AR est bornée par le dispatch CPU, pas par le calcul.

## Réglage des limites mobiles (4 bits `int4-mixed-head`, profil mobile, mêmes conditions)

| Cache MLX / seuil | Temps | Pic process | Cache observé |
|---|---|---|---|
| 256 Mo / 3 Go (0.2.1), run 1 | 294 s | 2 820 Mo | 260-506 Mo |
| 256 Mo / 3 Go (0.2.1), run 2 | 234 s | 2 844 Mo | 246-319 Mo |
| 512 Mo / 4 Go | 276 s | 2 967 Mo | 413-574 Mo |
| **adaptatif** (1 Go / 4,75 Go pour 6 Go disponibles) | **221 s** | 3 522 Mo | 1 025-1 282 Mo |
| profil Mac (2/4/1 Go, référence) | 223 s | 4 392 Mo | 2-3,3 Go |

Deux runs identiques à 256 Mo s'écartent de 26 % : le bruit de l'entraînement tiers est du même ordre que l'effet cherché, la hiérarchie 512 < 256 n'a pas de sens physique et n'est pas à lire. Ce qui tient : le réglage adaptatif retrouve le temps du profil Mac (221 vs 223 s) pour +0,7 Go de pic (3,5 Go, toujours sous 4 Go avec 6 Go disponibles). Retenu par défaut (`mobileLimitsMB`), à confirmer sur l'iPhone et sur GPU libre.

## Remesure sur GPU libre, protocole complet (soir du 2026-09-26, 19:36-20:19)

120 s de refroidissement avant chaque run, ordre A/B/B/A par pack (MLX pur → optimisé → optimisé → MLX pur), aucun processus GPU concurrent (vérifié avant chaque run). Mode optimisé = résidence par étape + VAE fp16/256 + profil mobile avec **limites adaptatives** (cache 1 Go, seuil 4,75 Go pour 6 Go supposés disponibles). Temps hors chargement ; pic = footprint process.

| Pack | MLX pur (2 runs) | Optimisé (2 runs) | Contrôle | Pic MLX pur → optimisé |
|---|---|---|---|---|
| bf16 | 89,4 / 87,2 s | 98,2 / **134,7** s | ±2,5 % | 14,8 → 5,2 Go |
| 8 bits (`qint8-all --quant-head`) | 83,5 / 86,2 s | 84,9 / 85,6 s | ±3 % | 11,5 → 4,3 Go |
| 4 bits (`int4-mixed-head`) | 76,6 / 79,0 s | 76,6 / **109,5** s | ±3 % | 10,4 → 3,3-3,5 Go |

Par phase (abc / sémantique / NAR / VAE, s) : bf16 pur 8,7-9,0 / 25,7-25,9 / 50,9-53,5 / 1,5 ; bf16 optimisé 15,4-25,6 / 25,3-47,2 / 56,3-60,5 / 1,3 ; 8 bits pur 5,8-6,0 / 18,1 / 57,8-60,8 / 1,5 ; 8 bits optimisé 5,8 / 18,0-18,1 / 59,9-60,5 / 1,2 ; 4 bits pur 3,0 / 13,0-13,7 / 59,1-60,6 / 1,5-1,6 ; 4 bits optimisé 2,8-5,9 / 12,8-25,7 / 59,7-76,3 / 1,2-1,5.

### Lecture
- **Les contrôles tiennent (±3 %)** : ces temps sont des références, contrairement à ceux de la journée (entraînement tiers : ×3 sur tout).
- **8 bits et 4 bits : la résidence ne coûte rien** (84,9-85,6 vs 83,5-86,2 s ; 76,6 vs 76,6-79,0 s) pour une mémoire divisée par 2,7 et 3. Le « +26 % du 8 bits » de la journée était bien un artefact.
- **Un run optimisé sur deux est sorti de la bande** en bf16 (134,7 s) et en 4 bits (109,5 s), jamais en 8 bits ni en MLX pur. bf16 : cause identifiée dans la trace — phase plan à 30 % de GPU et 107-110 % de CPU (un second thread occupé), signature de la récupération du cache MLX au seuil (working set 4,3 Go + cache 1 Go ≈ seuil 4,75 Go) ; sans objet pour l'iPhone (bf16 hors budget). 4 bits : trace sans anomalie (GPU 90-98 %, CPU identique au run 1, toutes phases +43 % dont le NAR pur GPU) — plutôt un état d'horloge GPU ponctuel après 45 min de charge quasi continue, le contrôle suivant étant revenu à 79 s ; à surveiller, non reproduit (1 run sur 12).
- **Hiérarchie des packs sur Mac** : 4 bits 77 s < 8 bits 84 s < bf16 88 s. Le gain vient des phases AR (16 / 24 / 34 s), bornées par la bande passante et le dispatch ; le NAR (8 bits dans les deux packs quantifiés) est constant à ≈ 59-60 s et légèrement plus lent qu'en bf16 (51-53 s) : le matmul quantifié ne gagne rien sur les grandes matrices. Le NAR fait désormais 75 % du temps du 4 bits.
- **Mémoire avec limites adaptatives** : 4 bits 3,3-3,5 Go (cache 1 Go compris), 8 bits 4,3 Go, bf16 5,2 Go. Sur un iPhone à 6,1 Go disponibles, seuls les deux packs quantifiés tiennent avec marge.

