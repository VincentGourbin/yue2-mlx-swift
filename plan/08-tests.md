## 8. Stratégie de tests et fixtures de parité

Deux tiers (convention h3/ltx/qwen38), framework **swift-testing** (`import Testing`, `@Suite`, `@Test`, `#expect`) :

- **Tier 1 — sans poids, toujours vert** : logique pure (protocole, npy, WAV) + parité **tiny** sur des modèles aléatoires seedés dont les poids **et** les sorties attendues sont dans `parity/*.safetensors` (committés, < 5 Mo). Les tiny reproduisent la topologie exacte (mêmes clés, mêmes modules) en fp32 ⇒ tolérance max abs **1e-4**.
- **Tier 2 — vrais poids**, `@Suite(.enabled(if: ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"] != nil))` + vérification d'existence du dossier (un `YUE2_MODELS_DIR` périmé **skippe**, ne fait pas échouer). Fixtures real sous `$YUE2_MODELS_DIR/parity/`, générées par `real_fixtures.py` (jamais committées). Tolérances bf16 : `relativeError = mean|a−b| / mean|b|` < 2e-2 (logits), < 3e-2 (velocity), max abs < 2e-3 (VAE fp32).
- Exécution : `Scripts/run-tests.sh` (relais automatique de toutes les `YUE2_*` en `TEST_RUNNER_YUE2_*`, `TEST_RUNNER_SWT_EXPERIMENTAL_MAXIMUM_PARALLELIZATION_WIDTH=1`, watchdog) ; en cas de doute sur un crash de worker, `xcodebuild build-for-testing` puis `YUE2_MODELS_DIR=… xcrun xctest .xcodebuild/Build/Products/Debug/YuE2Tests.xctest`.
- **Garde de couverture avant toute comparaison** : `update(parameters:)` no-op silencieux sur une clé inconnue ⇒ chaque test de parité asserte d'abord « 0 paramètre non alimenté, 0 clé inutilisée ».
- **Test de bissection** pour chaque composant (VAE : par bloc ; LM : par couche `hidden_states[i]` ; NAR : `after_vae2llm`, `after_time`, `after_pos`, `after_layer_i`, `velocity`) : la fixture contient les taps intermédiaires, le test compare dans l'ordre et imprime `PARITY <tap>: rel <e>` — on localise la **première** divergence.

### 8.1 `Scripts/reference/tiny_fixtures.py` (tier 1, CPU fp32, seedé — à écrire en tâche 1.3, avant tout code Swift de modèle)

Utilise **les classes upstream** (`from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM, StaticKVCache`, `from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig`, `from yue2 import nar`, `from yue2.sampling import distribution`, `from yue2.protocol import Sampling`) — jamais une réimplémentation. `torch.manual_seed` fixe, `torch.set_num_threads(1)`, `torch.inference_mode()`. Chaque fichier porte `metadata={"kind":…, "seed":…, "upstream_sha":…, "generated": date}`.

| Fichier | Modèle | Contenu (clés safetensors) |
|---|---|---|
| `parity/tiny_vae.safetensors` | `YuE2VAEConfig(encoder_config=dict(channels=1, c_mults=[1]*6, strides=[2,2,4,4,5,6], use_snake=True, in_channels=2, latent_dim=4), decoder_config=dict(…, out_channels=2, latent_dim=2, snake_type="vanilla", final_tanh=False), latent_dim=2)`, seed 231 | tous les `decoder.*` tels que stockés (`weight_g`, `weight_v`, …) **+** `expected_merged.<clé>` (poids fusionnés, layout torch) ; `input_T{1,15,17,33}` `[1,2,T]` ; `expected_full_T{…}` ; taps `tap_T33.after_layer_{0..8}` ; `expected_tiled_T{17,33}_core{8,16}` |
| `parity/tiny_lm.safetensors` | `YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4, vocab_size=32, max_position_embeddings=128, latent_dim=64, max_latent_frames=128)`, seed 173 (comme `tests/test_nar.py`) | tous les tenseurs du modèle (mêmes clés que le real) ; `ids_full` `[1,6]` + `logits_full` ; `logits_chunks` pour blocs (0,2),(2,5),(5,6) ; `positions_explicit` + `logits_explicit` ; `rope_cos/sin` pour positions [0,1,4,9,10] ; `chunk.ar_tokens = [2,3,4,5]`, `noise [5,64]` seed 42, `velocity_raw{20,0,-2.3}` + taps par couche ; `solve4_noise [3,64]` seed 381 + `solve4_expected` ; `greedy_prefix`, `greedy_tokens` (8 tokens, temperature 0, bornes monkeypatchées `EOD 5, ABC_END 6, MUSIC_END 7, CODEC_OFFSET 8, CODEC_SIZE 20`) ; `time_emb_raw{…}` |
| `parity/sampling.safetensors` | aucun (logits aléatoires seed 7, `[1, 184704]` fp32 ; second jeu bf16 pour legacy_off) | pour 6 configs (ABC défaut, sémantique défaut, `temperature 0`, `top_p 1`, `legacy_off`, `min_tokens` actif) : `history` (int32), `step`, `scores_expected` (sortie de `distribution()`, avec `-inf`) |
| `parity/protocol.json` | — | pour `examples/song.json` et `examples/tonight-awake.json` × cot ∈ {off, melody, full} × abc ∈ {none, `examples/score.abc`} : `text`, `prefix`, `negative` (ids exacts via tiktoken) |
| `parity/tokenizer_corpus.json` | — | ≥ 8 textes (song.json, 3 ABC, lyrics chinois, instruction, chaîne avec `<|im_start|>` littéral, emoji/accents) → ids tiktoken `encode_ordinary` après NFC |

### 8.2 `Scripts/reference/real_fixtures.py {vae|lm|nar|song} --models-dir $YUE2_MODELS_DIR` (tier 2)

- `vae` (CPU fp32) : `latent_40 [1,64,40]` (seed 11) → `audio_40` ; `latent_1100` → `audio_1100_tiled` (core 1024) et `audio_1100_full` ; écrit `$YUE2_MODELS_DIR/parity/vae.safetensors`.
- `lm` (**MPS bf16** — le pipeline upstream supporte `device="mps"` ; ≈ 1-3 min) : `prefix_abc` (song.json, cot full) → `logits_last_abc` ; 8 tokens greedy phase ABC ; `prefix_semantic` (ABC = `examples/score.abc`) → `logits_last_semantic` ; 8 tokens greedy phase sémantique ; `hidden_after_layer_{0,7,14,27}` sur le préfixe (bissection).
- `nar` (MPS bf16) : préfixe sémantique de `lm` + 16 tokens sémantiques fixes (les 16 premiers greedy) + `noise [16,64]` seed 42 → `velocity_raw{20,0,-2.3}` + taps ; `solve4_expected`.
- `song` (MPS, ≈ 15-40 min) : exécute `YuE2Pipeline(song.json, abc_sampling max_tokens 64, semantic max_tokens 100, ode_steps 4)` et sauvegarde les artefacts (`latent.npy`, `semantic.npy`, `audio.flac`) — sert à `yue2 decode` (1.11) et au smoke (3.6). Optionnel : la chanson complète seed 831001 pour l'écoute comparative G-4.

### 8.3 Inventaire des tests Swift (`Tests/YuE2Tests/`)

| Fichier | Tier | Ce qu'il prouve |
|---|---|---|
| `ProtocolTests.swift` | 1 | constantes, `text()`, préfixes, négatif, `chunkRanges`, validations `SongRequest`/`Sampling` (cas de `test_protocol.py`), `parity/protocol.json` |
| `NumpyAndWavTests.swift` | 1 | round-trip `.npy` ; en-tête WAV ; nombre d'échantillons |
| `VAEWeightLoaderTests.swift` | 1 | fusion weight-norm, layouts, couverture |
| `VAEDecoderTests.swift` | 1 | parité tiny + bissection + tiled == full + longueur naturelle + refus halo |
| `LMLoaderTests.swift` | 1 (+2) | tiny charge sans diff ; real charge, mémoire ≈ 7,3 Go |
| `BackboneParityTests.swift` | 1 | RoPE, logits pleine séquence, cache par blocs, positions explicites, taps par couche |
| `SamplingTests.swift` | 1 | `parity/sampling.safetensors` ; pénalité fenêtrée vs implémentation CPU Swift naïve ; top-k égalités ; top-p garde 1/3 ; mapping restreint |
| `TokenGeneratorTests.swift` | 1 | greedy 8 tokens tiny ; min_tokens ; arrêt END ; truncated ; CFG deux caches (tiny, cfg 1.5 : logits combinés == fixture) |
| `NARParityTests.swift` | 1 | time embedder, `songChunks`, velocity ×3, solve(4), synthesize 3 chunks |
| `KVCacheReuseTests.swift` | 1 | O2 : cache réutilisé == prefill frais |
| `TokenizerTests.swift` | 2 | corpus, round-trip, aucun id spécial |
| `RealVAEParityTests.swift` | 2 | 40 frames, 1 100 frames tuilé |
| `RealLMParityTests.swift` | 2 | logits, greedy ABC/sémantique, bissection couches |
| `RealNARParityTests.swift` | 2 | velocity, solve(4) |
| `GenerationSmokeTests.swift` | 2 | chanson courte : WAV fini, durée, RMS ; artefacts + `result.json` |
| `QuantizationTests.swift` | 2 | O5 : rel < 5e-2, ≥ 7/8 greedy |

