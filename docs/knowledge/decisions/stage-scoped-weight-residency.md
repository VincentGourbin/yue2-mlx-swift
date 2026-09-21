# Décision — résidence des poids par étape (iPhone)

**Contexte** (2026-09-21) : sur le Mac, le pipeline `int4-mixed` culmine à 10,3 Go de footprint process alors que le pack ne pèse que 3,1 Go. Les traces `swift-mlx-profiler` montrent que le pic n'est pas en fin de NAR (comme supposé dans la note « YuE2 Memory Budget ») mais **dans le décodage VAE** : tuile de 1024 frames en fp32 → +5,5 Go d'activations transitoires, au-dessus des 3,6 Go de poids LM encore résidents alors qu'aucune étape ne les lit plus.

**Constat de base** : chaque étape ne lit qu'une partie du modèle — plan/sémantique : voie AR (756 Mo int4) + `embed_tokens` (203 Mo) + `lm_head` (722 Mo bf16, 190 Mo avec `--quant-head`) ; NAR (cache de préfixe réutilisé, un seul chunk) : voie NAR seule (1 428 Mo qint8) + `vae2llm`/`llm2vae`/`time_embedder` ; VAE : rien du LM. Le budget iPhone doit donc compter le **max** des étapes, pas leur somme.

**Décision** : implémenter la résidence par étape en MLX pur, sans changer de runtime :
- `YuE2ForCausalLM.releaseWeights(of:)` / `releaseAllWeights()` : remplace les paramètres d'une voie par des zéros *non évalués* (structure du module intacte, buffers GPU libérés).
- `LMWeightLoader.load(residency:)` / `load(path:into:)` : ne matérialise que la voie demandée (le chargement safetensors est paresseux ; ne pas `eval` = ne pas allouer).
- `ModelSession.loadWeights(of:)` / `releaseWeights(of:)` suivent l'état ; `YuE2Pipeline(releaseWeightsBetweenStages: true)` charge l'AR seul, puis au moment où le cache de préfixe NAR est prêt (`Synthesizer.onBeforeChunk(needsARPath:)`) libère l'AR et charge le NAR, puis libère tout le LM avant le VAE. Le `ModelSession` est alors « consommé » : recharger avant une autre chanson (l'app iOS reprend de toute façon chaque étape depuis ses artefacts).
- Tuile VAE et précision paramétrables (`vaeCoreFrames`, fp16) ; `YUE2_MEMORY_PROFILE=mobile` pour mesurer les limites iPhone sur le Mac.

**Mesure** (même chanson 64 s, `int4-mixed`, profil Mac, cache AR 2 Go / NAR 4 Go / VAE 1 Go) : footprint process max 10 319 → **6 438 Mo** (pic désormais dans la phase AR, poids + cache 2 Go) ; NAR stable à 2,55 Go actifs / 3,8 Go process ; VAE (fp16, tuile 256, LM libéré) 1,1 Go actifs / 2,3 Go process. Voir `log.md` (2026-09-21) pour le tableau complet, le profil mobile et les temps A/B/B/A.

**Non retenu** : LiteRT (reconversion complète depuis PyTorch, pas d'ANE iOS) — le problème est la discipline de résidence, pas le planificateur mémoire du runtime.
