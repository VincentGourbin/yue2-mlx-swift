## 6. Jalon 3 — NAR flow matching et chanson complète (≈ 4-6 jours)

**Objectif** : `yue2 generate --request song.json --out dir/` produit `audio.wav` ; parité tiny exacte du solveur ; parité real de la vitesse sous tolérance ; première écoute. Démo G-4.

### 6.1 Fichiers cibles

```
Models/LM/NARModules.swift        // TimestepEmbedder (sinus fp32 → MLP), latent pos lookup, shiftT(raw) en bf16
Synthesis/NARChunk.swift          // songChunks(prefix, codec, seed|noise, context) → [Chunk(arTokens, noise)]
Synthesis/CachedNAR.swift         // prefill AR (K/V par couche) ou adoption d'un KVCache existant ; velocity(state, rawT) ; solve(steps, onProgress, cancel)
Synthesis/Synthesizer.swift       // synthesize(): boucle chunks, concat, contrôle fini
Pipeline/YuE2Pipeline.swift       // plan → semantic → synthesize → decode ; timings ; SongResult.saveArtifacts (result.json, config.json, request.json, hachages)
```

### 6.2 Tâches

| # | Tâche | Source à copier / adapter | Cible | Critère de sortie |
|---|---|---|---|---|
| 3.1 | `TimestepEmbedder`, `shiftT` (`sigmoid` en bf16), lookup `pe` ; `songChunks` avec bruit injectable | `modeling_yue2.py::TimestepEmbedder/_shift_t_value`, `nar.py::song_chunks` | `Models/LM/NARModules.swift`, `Synthesis/NARChunk.swift` | tier 1 tiny : `time_embedder(t)` == fixture pour raw ∈ {20, 0, -2.3} (1e-4) ; `chunkRanges(11 frames, prefix 2, context 15) == [(0,5),(5,10),(10,11)]` ; `arTokens` du chunk 1 == `[2,3] + [8+5..8+9] + [7]` (constantes monkeypatchées) |
| 3.2 | `CachedNAR` : prefill AR causal + stockage K/V post-RoPE ; `velocity` (§2.4 point 5) avec `MLXFast.RoPE(offset: L_ar)` et SDPA sans masque sur `concat(K_ar, k)` | `nar.py::CachedNAR` ; `Attention.projectQKV` (2.1) | `Synthesis/CachedNAR.swift` | tier 1 tiny (= `test_cached_velocity_matches_dense_original_forward`) : `velocity(noise, raw)` == fixture `velocity_expected_{20,0,-2.3}` (max abs 1e-4 fp32) |
| 3.3 | `solve(steps:)` midpoint (§2.4 point 6) : `raw` en Double, clamp ±20 ; état bf16 en real, fp32 en tiny (dtype = celui du modèle) ; progression `(completed, total)` ; annulation | `nar.py::solve` | `Synthesis/CachedNAR.swift` | tier 1 tiny (= `test_midpoint_solution_matches_dense_original_solver`) : `solve(steps: 4)` == fixture (1e-4) ; exactement 8 appels de velocity ; le prefill AR n'a lieu qu'une fois par chunk |
| 3.4 | `Synthesizer.synthesize()` multi-chunks + contrôle fini + concat fp32 | `nar.py::synthesize` | `Synthesis/Synthesizer.swift` | tier 1 tiny : 3 chunks (context 15) ⇒ résultat == concat des `solve` individuels (bit-à-bit) ; NaN ⇒ erreur `YuE2Error.nonFiniteLatents` |
| 3.5 | Parité **real** NAR : `real_fixtures.py nar` (§8.2) + `RealNARParityTests` + `yue2 parity nar` | — | `Tests/`, `CLI/` | préfixe `song.json` + `score.abc` fourni + 16 tokens sémantiques fixes + bruit fixe : `velocity` à raw ∈ {20, 0, -2.3} rel < 3e-2 (bf16, MPS vs Metal) ; `solve(steps: 4)` rel < 5e-2 |
| 3.6 | `YuE2Pipeline` + `yue2 generate` (options : `--request`, `--style/--lyrics/--cot/--seed/--abc-file/--cfg-scale`, `--vae`, `--ode-steps`, `--out`) ; `saveArtifacts` (fichiers §1 + `result.json` avec timings, `truncated`, SHA-256) ; `yue2 synthesize --semantic dir/` pour reprendre depuis les tokens | `pipeline.py::__call__/save_artifacts`, `storage.py` (simplifié) | `Pipeline/YuE2Pipeline.swift`, `CLI/GenerateCommand.swift` | tier 2 `GenerationSmokeTests` : requête `song.json` avec `abc_sampling.max_tokens 64`, `semantic.min_tokens 0`, `semantic.max_tokens 100`, `ode_steps 4` ⇒ `audio.wav` fini, durée == `(T·1920 - 64)/48000`, RMS ∈ [1e-3, 0.7] ; puis chanson complète seed 831001 : ⛔ **GATE G-4** (écoute Vincent) |
| 3.7 | `TTSMetrics` du profiler branché sur les 4 étapes ; `BENCHMARKS.md` créé avec la première ligne | profiler `TTSProfiling.swift` ; `gemma-4 BENCHMARKS.md` (structure « comment contribuer ») | `Pipeline/`, `BENCHMARKS.md` | `yue2 generate --profile` écrit le rapport + la trace ; la ligne §3.3 est remplie |

**Critères d'acceptation — ⛔ GATE G-4 puis fin de jalon** : tier 1/2 verts ; chanson complète écoutée ; temps par étape mesurés ; pic mémoire noté ; `docs/knowledge/log.md` à jour.

