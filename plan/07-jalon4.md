## 7. Jalon 4 — Optimisations et GUI de bench (≈ 5-8 jours, ⛔ GATE G-5 en entrée)

Règle : **une optimisation = une tâche = une mesure A/B/B/A** (protocole thermique `h3 CLAUDE.md` : refroidir avant chaque point, contrôle dans le même état thermique, juger contre la dispersion intra-variante ±2 %). Toute optimisation est **exacte** (parité inchangée) sauf O5 qui est validée par parité + écoute.

| # | Optimisation | Étape | Gain attendu | Source | Validation |
|---|---|---|---|---|---|
| O1 | lm_head restreint par phase (déjà en 2.6) | AR | −17 % d'octets/token en bf16 ; exact | — | greedy identique |
| O2 | **Réutilisation du cache KV** de la phase sémantique comme cache du préfixe NAR (chunk unique, `guidance == 1`) : le cache contient déjà `prefix + tokens` ; ajouter `MUSIC_END` par un pas de décodage si tronqué (sinon il vient d'être consommé — l'ajouter au cache à l'avant-dernier pas de `generate`) | NAR | − 1 prefill de ≈ 7 k tokens (≈ 10-20 s) | `convertvoxtral cloneKVCaches` | tier 1 tiny `KVCacheReuseTests` : velocity depuis cache réutilisé == prefill frais (1e-6) |
| O3 | `asyncEval` pipelining sampling/décodage (déjà en 2.7) et prefill par blocs 2048 | AR | TTFT et tok/s | `flux Qwen3Generator` | mesure |
| O4 | `Memory.cacheLimit` par étape (AR : 2 Go ; NAR : 4 Go ; VAE : 1 Go) + `clearCache()` entre étapes | tout | stabilité mémoire | `netflix-void VOIDMemoryManager` | pic mémoire |
| O5 | **Quantification 8-bit affine (group 64) de la voie AR uniquement** (`self_attn.*_proj`, `mlp.*`, optionnellement `lm_head`) — voie NAR, embed, norms, modules NAR restent bf16 ; preset `none/qint8/int4` ; export pré-quantifié `mlx-prequantized/` | AR | ×1,6-2 sur le décodage | `h3 Configuration/QuantizationConfig.swift` (71 l.) + `h3 Loading/PrequantizedCheckpoint.swift` | tier 2 : logits rel < 5e-2 vs bf16, ≥ 7/8 tokens greedy identiques ; écoute Vincent (G-5 autorise) ; **jamais** `asType(weight.dtype)` sur un `QuantizedLinear` (§9 n°10) |
| O6 | NAR : tuilage des requêtes (`queryChunkSize`, ex. 1024) **si** le profiler montre une allocation de scores `[16, L_nar, L_ar+L_nar]` ; sinon ne rien faire | NAR | mémoire | `nar.py::attention` | parité inchangée |
| O7 | VAE : `compile(shapeless: true)` de `SnakeBeta` + des `ResidualUnit` ; tuile 2048 frames si mémoire OK | VAE | ×1,3-2 | — | parité 1.9 inchangée ; mesure |
| O8 | NAR : précalcul par chunk de `pe[0..L_nar)`, time embedding calculé une fois par appel, `K_ar/V_ar` stockés contigus `[8, L_ar, 128]` pour éviter un `concat` par couche (pré-allouer `[8, L_ar + L_nar, 128]` et écrire la partie NAR en place) | NAR | −10-20 % | — | parité 3.2 inchangée |
| O9 | CFG mode off : batch 2 (positif/négatif) avec left-padding au lieu de deux forwards | AR (off) | ×1,6 en mode off | `qwen38 Qwen4ExpBatchPadding.swift` | parité greedy mode off |
| O10 | GUI `yue2-bench-ui` (SwiftUI) : onglet Génération (style, lyrics, cot, seed, abc, cfg ; progression par étape ; lecture audio `AVAudioPlayer` ; export) ; panneau Mesures (TTSMetrics, mémoire) ; onglet Historique (table + CSV) ; bouton « Export trace » ; sélecteur de modèle/quant présents sur disque | — | — | `gemma-4 Sources/Gemma4BenchUI/ContentView.swift`, `qwen38 PLAN §4.3` | démo ⛔ **GATE G-8** |

Ordre imposé : O4 → O2 → O8 → O7 → O5 → O6/O9 (si mesures le justifient) → O10. Chaque ligne ajoutée à `BENCHMARKS.md` avec commit et trace.


