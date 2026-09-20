## 10. Écosystème — brique → dépôt → fichier

| Brique YuE2 | Dépôt | Fichier(s) | Adaptation |
|---|---|---|---|
| Attention Qwen3 (GQA, q/k-norm, RoPE), MLP, couche | flux-2-swift-mlx | `Sources/FluxTextEncoders/Model/Qwen3/{Qwen3Attention,Qwen3MLP,Qwen3DecoderLayer,Qwen3Model}.swift` | dupliquer la couche en deux voies ; `projectQKV` exposé ; RMSNorm → `MLXFast.rmsNorm` |
| RMSNorm manuel (fallback) | flux-2-swift-mlx | `Sources/FluxTextEncoders/Model/RMSNorm.swift` | seulement si promotion de type observée |
| Sampling top-p GPU, asyncEval | flux-2-swift-mlx | `Sources/FluxTextEncoders/Generation/Qwen3Generator.swift` L88, L215, L318-440 | pénalité → pure MLX (§5.2 2.5) ; ajouter top-k, min_tokens, masque |
| KV cache simple + clone | flux-2 / convertvoxtral | `Sources/Flux2Core/Transformer/TransformerKVCache.swift` ; `Sources/VoxtralCore/TTS/VoxtralTTSModeling.swift` (`cloneKVCaches` ≈ L697) | append post-RoPE, offset |
| Prefill par blocs | convertvoxtral | `Sources/VoxtralCore/VoxtralModeling.swift` L1163-1190 | bloc 2048 |
| Loader safetensors + diff bidirectionnel | h3-swift-mlx | `Sources/MiniMaxH3/Loading/WeightLoader.swift` (`loadShardedWeights`, `apply`) | fichier unique |
| Fusion weight-norm + layouts conv | h3-swift-mlx | `Sources/MiniMaxH3/Loading/VAEWeightLoader.swift` (en-tête) | idem, décodeur seul |
| SnakeBeta | ltx-video-swift-mlx | `Sources/LTXVideo/Models/AudioVAE/BigVGANVocoder.swift` L15-40 | copie verbatim |
| Décodeur audio fp32 en Sequential | h3-swift-mlx | `Sources/MiniMaxH3/Models/AudioVAE/H3AudioVAEDecoder.swift` | topologie Oobleck (§2.5) |
| Export WAV | ltx-video-swift-mlx | `Sources/LTXVideo/Utils/AudioExporter.swift` | + variante float32 |
| Téléchargeur HF (URLSession, reprise) | gemma-4-swift-mlx | `Sources/Gemma4Swift/Download/*` | catalogue YuE2 + tokenizer Qwen2.5 |
| Tokenizer swift-transformers | flux-2-swift-mlx | `Sources/FluxTextEncoders/Tokenizer/TekkenTokenizer.swift` L223-240 (`AutoTokenizer.from(modelFolder:)`) | NFC + sans tokens spéciaux |
| Presets de quantification + filtre | h3-swift-mlx | `Sources/MiniMaxH3/Configuration/QuantizationConfig.swift` | exclusions : `nar_*`, `embed_tokens`, `*norm*`, `vae2llm`, `llm2vae`, `time_embedder`, `latent_pos_embed` |
| Export pré-quantifié | h3-swift-mlx | `Sources/MiniMaxH3/Loading/PrequantizedCheckpoint.swift` | O5 |
| Gestion mémoire par étape | netflix-void-swift-mlx | `Sources/VOIDCore/.../VOIDMemoryManager.swift` L88-98 | 3 étapes |
| Profiling TTS (`TTSMetrics`, Chrome Trace) | swift-mlx-profiler 1.5.0 | `Sources/MLXProfiler/{TTSProfiling,LLMProfiling,ChromeTraceExporter}.swift` | déjà YuE2-shaped |
| CLI (entrée, `info`, `parity`, `profile`) | h3 / gemma-4 | `Sources/MiniMaxH3CLI/{MiniMaxH3CLI,ParityCommand}.swift` ; `Sources/Gemma4CLI/ProfileCommand.swift` | — |
| Script de tests + watchdog | gemma-4-swift-mlx | `Scripts/run-tests.sh` | préfixe `YUE2_` |
| Bench guard thermique | h3-swift-mlx | `scripts/bench-guard.sh` | Jalon 4 |
| GUI de bench | gemma-4-swift-mlx | `Sources/Gemma4BenchUI/ContentView.swift` | O10 |
| Scripts de référence Python | ltx / qwen38 | `ltx scripts/*_reference.py` + `scripts/README.md` ; `qwen38 Scripts/qwen4-exp-*-reference.py` | §8 |
| Structure CLAUDE.md / knowledge base | h3 / ltx | `CLAUDE.md`, `docs/knowledge/` | Annexe B |

