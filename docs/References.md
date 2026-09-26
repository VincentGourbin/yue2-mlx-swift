# Les six configurations de référence

`yue2 references` les liste, `yue2 generate --reference <id>` les applique, `YuE2ReferenceProfile.all` les expose à une app. Chacune fixe **tous** les réglages qui comptent : pack de poids, précision de calcul, façon de calculer le NAR, décodeur VAE, résidence des poids, profil mémoire, pas d'ODE. Rien n'est laissé au hasard entre deux machines : une référence produit la même chanson pour la même graine, et un temps comparable pour un matériel comparable.

Source : `Sources/YuE2Core/Configuration/ReferenceProfiles.swift`. Mesures : [Benchmarks.md](Benchmarks.md).

## Le tableau

Chanson de 70,0 s d'audio (1 750 tokens sémantiques), 32 pas d'ODE, MacBook Pro M3 Max 96 Go, GPU libre, un passage après 120 s de refroidissement.

| Id | Pack | Calcul | Temps | Pic process | Pour qui |
|---|---|---|---|---|---|
| `4bit-fast` | `int4-mixed-head` (2,5 Go) | tout résident, NAR dé-quantifié en bf16, VAE fp16 tuile 1024, caches Mac | **65,7 s** | 8,9 Go | Mac : le seul profil qui génère plus vite que le temps réel à 32 pas |
| `4bit-lean` | `int4-mixed-head` | résidence par étape, NAR packé, fp16, VAE fp16 tuile 256, limites mobiles | 77,7 s | **3,4 Go** | iPhone 15 Pro Max et tout Mac à 8 Go |
| `8bit-fast` | `qint8-all-head` (3,7 Go) | tout résident, NAR dé-quantifié en bf16, VAE fp16/1024 | 76,5 s | 9,9 Go | Mac, quand on veut la voie AR en 8 bits |
| `8bit-lean` | `qint8-all-head` | résidence, NAR packé, fp16, VAE fp16/256, limites mobiles | 79,3 s | 4,4 Go | iPhone 8 Go avec marge, Mac 8-16 Go |
| `16bit-fast` | bf16 (7,3 Go) | tout résident, VAE fp16/1024 | 84,9 s | 12,0 Go | référence qualité |
| `16bit-lean` | bf16 | résidence par étape, VAE fp16/256, caches Mac | 84,9 s | 6,9 Go | Mac 16-24 Go |

Lecture rapide : le 4 bits est le plus rapide parce que ses phases autorégressives (partition puis tokens sémantiques) sont bornées par la bande passante mémoire et le dispatch ; le NAR, en 8 bits dans les deux packs quantifiés, coûte ≈ 50 s quel que soit le profil et représente les trois quarts du temps. Les profils économes divisent la mémoire par deux à trois pour un surcoût de temps nul (8 bits) à 18 % (4 bits, dû au NAR packé plutôt qu'à la résidence).

## Ce que chaque réglage fait

| Réglage | Valeurs | Effet |
|---|---|---|
| Pack (`quant`, `quantizeHead`) | `none` bf16 · `qint8-all` + head · `int4-mixed` + head | Poids sur disque et en mémoire. `int4-mixed` = voie AR, embeddings et head en 4 bits, voie NAR en 8 bits (le NAR à 4 bits échoue la parité : l'erreur se compose sur 64 évaluations). `+head` = `lm_head` quantifié et calculé par `quantizedMatmul` sur les lignes autorisées par la phase. |
| `precision` | `bf16` · `fp16` | Dtype des paramètres flottants et des activations. fp16 pour l'iPhone (GPU et Neural Engine) ; bf16 sur Mac. Les deux sont validés à l'écoute. |
| `narCompute` | `packed` · `dequantized` | `packed` calcule les projections NAR par `quantizedMatmul` (mémoire minimale). `dequantized` les dé-quantifie une fois en bf16 avant le solve : +1,4 Go pour un NAR 8 bits, ≈ 10 % de NAR en moins sur Mac (le matmul quantifié est plus lent qu'un GEMM bf16 sur les grandes activations). Écart numérique 1,3 % (poids arrondis en bf16), sous le budget de parité. |
| `vaePrecision`, `vaeCoreFrames` | fp16 · fp32 ; 256 · 1024 | Le décodeur Oobleck en fp16 est inaudible vs fp32 (SNR 53-57 dB) et plus rapide. La tuile fixe le pic transitoire du décodage : 1024 ≈ 2,9 Go actifs en fp16, 256 ≈ 2,0 Go. |
| `releaseWeightsBetweenStages` | bool | Résidence par étape : voie AR seule au chargement, échange AR → NAR quand le cache de préfixe est prêt, LM entier libéré avant le VAE. Le budget mémoire devient le max des étapes au lieu de leur somme. Coût temps nul (mesuré A/B/B/A). |
| `memoryProfile` | `mac` · `mobile` | Caches MLX par étape (Mac : 2 / 4 / 1 Go) ou limites calées sur la mémoire disponible (mobile : cache ≈ 1 Go, seuil de récupération = disponible − 1,25 Go). Les limites mobiles font thrasher un working set bf16 : `16bit-lean` reste en `mac`. |
| `odeSteps` | nil = 32 | Pas du solveur midpoint (2 évaluations par pas). 24 pas ramènent le 8 bits sous 70 s, 16 pas tout sous 45 s ; validation à l'oreille en cours. |

## Choisir

- **Mac 32 Go et plus** : `4bit-fast`. Si vous voulez la voie AR en 8 bits : `8bit-fast`. Pour une référence de qualité : `16bit-fast`.
- **Mac 16-24 Go** : `16bit-lean` (6,9 Go), ou `8bit-lean` / `4bit-lean` si d'autres apps tournent.
- **Mac 8 Go, iPhone** : `4bit-lean` (3,4 Go). `8bit-lean` tient aussi (4,4 Go) sur un iPhone 15 Pro Max avec l'entitlement mémoire.
- **Encore plus vite, mémoire libre** : hors des six, `--quant int4 --quant-head` (voie AR en 4 bits, NAR en bf16, pack `int4-head` 3,5 Go) donne 66-69 s ; c'est le candidat à un septième profil.

## Ajouter ou modifier un profil

1. Ajouter l'entrée dans `YuE2ReferenceProfile.all` (chaque champ est un réglage existant, rien de nouveau à câbler).
2. Mesurer avec `Scripts/bench-references.sh <request.json> 1750 2` (deux passes, protocole de [Benchmarks.md](Benchmarks.md)).
3. Écouter le WAV produit contre `16bit-fast` sur la même graine.
4. Reporter la ligne dans `BENCHMARKS.md` et la décision dans `docs/knowledge/decisions/reference-profiles.md`.

## Équivalents en ligne de commande

Les profils ne sont que des raccourcis ; chaque réglage existe en option :

```bash
# 4bit-fast
yue2 generate --request song.json --quant int4-mixed --quant-head --nar-compute dequantized --vae-precision fp16 --out run/
# 4bit-lean (YUE2_MEMORY_PROFILE=mobile active la résidence et la tuile 256 par défaut)
YUE2_MEMORY_PROFILE=mobile yue2 generate --request song.json --quant int4-mixed --quant-head --precision fp16 --vae-precision fp16 --out run/
# 16bit-lean
yue2 generate --request song.json --release-weights-between-stages --vae-precision fp16 --vae-core-frames 256 --out run/
```
