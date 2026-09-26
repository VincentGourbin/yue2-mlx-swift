# Décision — six configurations de référence (4/8/16 bits × rapide/économe)

**Contexte** (2026-09-26, demande de Vincent) : pré-qualifier six modèles pré-configurés, câblés dans le dépôt, éventuellement republiés comme poids combinés sur Hugging Face.

**Décision** : `YuE2ReferenceProfile.all` (`Sources/YuE2Core/Configuration/ReferenceProfiles.swift`), exposé par `yue2 references` et `yue2 generate --reference <id>` ; l'app peut itérer la même liste. Chaque profil fixe : pack (préset + head), précision LM, calcul NAR (packé / dé-quantifié bf16), VAE (précision, tuile), résidence par étape, profil mémoire, pas d'ODE (nil = 32). `Scripts/bench-references.sh` remesure la matrice avec le protocole de CLAUDE.md.

| Id | Temps 70 s (M3 Max) | Pic process | Rôle |
|---|---|---|---|
| 4bit-fast | 65,7 s | 8,9 Go | Mac, le seul sous 70 s à 32 pas |
| 4bit-lean | 77,7 s | 3,4 Go | iPhone |
| 8bit-fast | 76,5 s (24 pas : 67,3) | 9,9 Go | Mac, qualité 8 bits |
| 8bit-lean | 79,3 s (20 pas : 64,4) | 4,4 Go | iPhone 8 Go avec marge |
| 16bit-fast | 84,9 s | 12,0 Go | référence qualité |
| 16bit-lean | 84,9 s | 6,9 Go | Mac 16-24 Go |

**Choix motivés** : les profils rapides dé-quantifient le NAR en bf16 pour le solve (le matmul 8 bits est plus lent qu'un GEMM bf16 sur les grandes activations ; écart rel mesuré 1,3 % vs packé, sous le budget E2, à écouter) ; les profils économes gardent le NAR packé, la résidence par étape, fp16 et la tuile VAE 256 avec les limites mobiles adaptatives ; `16bit-lean` garde les caches Mac (les limites mobiles font thrasher un working set bf16 : 98-135 s). Le pas de décodage compilé a été mesuré sans gain et n'est activé nulle part.

**Ouvert** : un septième pack « int4-head + NAR bf16 » (66,1 s, 3,5 Go sur disque) ; le pas d'ODE par défaut (24 met le 8 bits sous 70 s) ; la republication Hugging Face des packs `int4-mixed-head`, `qint8-all-head` (et `int4-head`) — décisions listées dans `.local-runs/a-valider-2026-09-27/README.md`.
