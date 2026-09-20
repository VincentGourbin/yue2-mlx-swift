## 5. Jalon 2 — Backbone AR : plan ABC et tokens sémantiques (≈ 4-6 jours)

**Objectif** : `yue2 plan` produit un `score.abc` ; `yue2 semantic` produit les tokens sémantiques ; parité tiny exacte et parité real sous tolérance ; débit mesuré. Démo G-3.

### 5.1 Fichiers cibles

```
Models/LM/YuE2Config.swift          // décodage de config.json (champs §2.2), validation latent_type == "vae"
Models/LM/RMSNorm.swift             // MLXFast.rmsNorm wrapper (weight bf16)
Models/LM/Attention.swift           // YuE2Attention : q/k/v/o + q_norm/k_norm + RoPE ; projectQKV(x, offset) et forward(cache)
Models/LM/MLP.swift                 // SwiGLU
Models/LM/DecoderLayer.swift        // voies AR et NAR (mêmes types, clés self_attn/mlp vs nar_self_attn/nar_mlp)
Models/LM/Backbone.swift            // embed_tokens, layers, norm, rotary (offset)
Models/LM/KVCache.swift             // cache par couche, append post-RoPE, offset ; clone()
Models/LM/YuE2ForCausalLM.swift     // lm_head + modules NAR (vae2llm, llm2vae, time_embedder, latent_pos_embed) ; forwardAR(tokens, cache, keepLast)
Models/LM/RestrictedHead.swift      // O1 : lm_head tranché par phase + table id local → global
Loading/LMWeightLoader.swift        // charge les 628 tenseurs, diff bidirectionnel, eval par tenseur
Generation/LogitsProcessor.swift    // chaîne §2.3 (pure MLX, sans aller-retour hôte)
Generation/TokenGenerator.swift     // prefill par blocs + boucle décode + CFG + asyncEval pipelining + callbacks
Pipeline/Planner.swift              // plan(): phase ABC ; SymbolicPlan
Pipeline/SemanticGenerator.swift    // generateSemantic(): phase sémantique ; SemanticResult
```

### 5.2 Tâches

| # | Tâche | Source à copier / adapter | Cible | Critère de sortie |
|---|---|---|---|---|
| 2.1 | `YuE2Config` + `RMSNorm` + `MLP` + `Attention` (q_norm/k_norm avant RoPE, `MLXFast.RoPE(traditional:false, base:1e6)`, `MLXFast.scaledDotProductAttention(scale: 1/√128, mask: .causal ou nil)`) | `flux-2 Sources/FluxTextEncoders/Model/Qwen3/{Qwen3Attention,Qwen3MLP,Qwen3DecoderLayer}.swift` (**garder** q_norm/k_norm : YuE2 les a) ; RMSNorm : `MLXFast.rmsNorm` (fallback `flux RMSNorm.swift` si promotion de type) | `Models/LM/*.swift` | compile ; test unitaire tier 1 : RoPE Swift == fixture `rope_expected` (tiny_lm) sur 5 positions dont offset ≠ 0 |
| 2.2 | `DecoderLayer` (deux voies), `Backbone`, `YuE2ForCausalLM.forwardAR` (voie AR seule, cache optionnel, `keepLast`) ; `KVCache` (append post-RoPE, `offset`, `clone`) | `flux Qwen3Model.swift` (structure) ; `flux Flux2Core/Transformer/TransformerKVCache.swift` (struct simple) ; `convertvoxtral VoxtralTTSModeling.swift cloneKVCaches` | `Models/LM/*.swift` | tier 1 tiny : logits pleine séquence == fixture (max abs 1e-4 fp32 ; la fixture tiny est **fp32** des deux côtés) |
| 2.3 | `LMWeightLoader` : `loadArrays` du fichier unique, `update(parameters:verify:)`, **eval tenseur par tenseur** ; `pe` chargé comme paramètre gelé | `h3 WeightLoader.apply` ; pitfall §9 n°8 | `Loading/LMWeightLoader.swift` | tier 1 : `parity/tiny_lm.safetensors` charge sans clé manquante ni inutilisée ; tier 2 : le vrai checkpoint charge en < 20 s, `Memory.activeMemory` ≈ 7,3 Go, 0 diff |
| 2.4 | Parité cache : prefill par blocs (`prefillStepSize` 2048) puis décodage 1 token ; positions explicites | `convertvoxtral VoxtralModeling.swift` L1163-1190 (prefill par blocs) | `Generation/TokenGenerator.swift` (partie prefill) | tier 1 tiny (= `test_cached_chunks_equal_full`, `test_explicit_position_ids`) : blocs (0,2),(2,5),(5,6) ⇒ logits identiques à la passe pleine (1e-4) |
| 2.5 | `LogitsProcessor` (§2.3, pure MLX) : masque autorisé, min_tokens, pénalité fenêtrée (vecteur `window` MLXArray ≤ 100 ids ; `counts` = `(ids[:,None] == ids[None,:]).sum(1)` ; `alpha = penalty^counts` ; scatter `logits[ids] = where(l<0, l·alpha, l/alpha)`), température, top-k (seuil via `sorted`/`take`), top-p (garde 1 ou 3), softmax, `MLXRandom.categorical(key:)` | `flux Generation/Qwen3Generator.swift` L318-440 (`sampleTopPGPU`) ; `sampling.py::distribution` | `Generation/LogitsProcessor.swift` | tier 1 : `parity/sampling.safetensors` (logits aléatoires [1, 184704], historiques, 6 configs incl. legacy_off et temperature 0) ⇒ scores **avant softmax** == fixture (max abs 1e-5, `-inf` identiques) ; argmax identique |
| 2.6 | `RestrictedHead` (O1, §7) : deux `Linear` construits à partir de `lm_head.weight` (lignes `[0, EOD) ∪ {ABC_END}` ; `[CODEC_OFFSET, +32768) ∪ {MUSIC_END}`) + mapping local→global ; le processeur travaille sur l'espace local | — (nouveau, 60 lignes) | `Models/LM/RestrictedHead.swift` | tier 1 tiny : logits restreints == `full_logits[allowed]` exactement ; mapping bijectif ; tier 2 : les 8 premiers tokens greedy sont identiques avec/sans restriction |
| 2.7 | `TokenGenerator.generate(prefix:sampling:seed:phase:negative:cfgScale:legacyOff:onToken:cancel:)` : boucle §2.3, CFG à deux caches, `asyncEval` du token suivant pendant la lecture hôte du précédent, timings (`prefill_seconds`, `ttft`, `output_tps`) | `flux Qwen3Generator.swift` L88/L215 (asyncEval), `sampling.py::generate_tokens` | `Generation/TokenGenerator.swift` | tier 1 tiny : `temperature 0` ⇒ suite de 8 tokens == fixture `greedy_tokens` (le tiny utilise des constantes de protocole monkeypatchées comme `test_nar.py` : `CODEC_OFFSET 8`, `MUSIC_END 7`, `EOD 5`, `ABC_END 6` — passer les bornes en paramètres du processeur) ; `min_tokens` respecté ; arrêt sur END ; `truncated` correct |
| 2.8 | `Planner.plan()` et `SemanticGenerator.generateSemantic()` + artefacts (`score.abc`, `abc_tokens.npy`, `prefix.npy`, `plan.json`, `semantic.npy`) | `pipeline.py::plan / generate_semantic`, `SymbolicPlan.save/load` | `Pipeline/*.swift` | tier 1 : `plan` avec `abc` fourni ⇒ préfixe == `parity/protocol.json` ; `SymbolicPlan.load` refuse un manifeste altéré |
| 2.9 | Parité **real** : `Scripts/reference/real_fixtures.py lm` (§8.2) + `RealLMParityTests` + `yue2 parity lm` | `h3 ParityCommand.swift` | `Tests/`, `CLI/` | logits du dernier token du préfixe `song.json` (≈ 150 tokens) : rel < 2e-2, argmax identique ; 8 tokens greedy phase ABC identiques ; 8 tokens greedy phase sémantique (ABC fourni `examples/score.abc`) identiques ; ⛔ **GATE G-2** : présenter les erreurs |
| 2.10 | CLI `yue2 plan --request song.json --out dir/` et `yue2 semantic --plan dir/ --out dir/` (+ `--max-tokens`, `--seed`, `--temperature`…) avec progression stderr (tokens, tok/s) | `h3 MiniMaxH3CLI.swift` (entrée), `gemma-4 ProfileCommand.swift` (métriques) | `CLI/*.swift` | `yue2 plan` sur `song.json` ⇒ `score.abc` syntaxiquement ABC (commence par `X:`/`L:`/`M:`/`K:` sur les 8 premières lignes) en < 60 s ; `yue2 semantic` ≥ 30 tok/s bf16 ; mesures dans `docs/knowledge/log.md` |
| 2.11 | Profiling : `ProfilingSession` autour de prefill/décode (`LLMMetrics`), `yue2 profile plan|semantic` → Chrome Trace | `gemma-4 Gemma4CLI/ProfileCommand.swift` ; profiler `LLMProfiling.swift` | `CLI/ProfileCommand.swift` | trace ouvrable dans Perfetto ; TTFT / prefill tok/s / decode tok/s affichés |

**Critères d'acceptation — ⛔ GATE G-3 (démo Vincent)** : tier 1 + tier 2 verts ; `yue2 plan` puis `yue2 semantic` sur `song.json` (seed 831001) ; Vincent lit le `score.abc` ; débit noté dans `BENCHMARKS.md` (§3.3, colonnes ABC/sém.).

