## 12. Décisions actées et questions restantes

**Actées (rév. 1)** :
- Pas de dépendance `mlx-swift-lm` ; port autonome sur `mlx-swift 0.31.6` (raisons §4.1).
- Tokenizer = `tokenizer.json` Qwen2.5 sans `added_tokens` (équivalence vérifiée le 2026-09-16), chargé par swift-transformers.
- Fixtures tiny committées (fp32, < 5 Mo) + fixtures real sous `$YUE2_MODELS_DIR/parity/`.
- Bruit ODE : `MLXRandom.normal` seedé côté Swift (non identique au RNG PyTorch, assumé) ; injection possible pour la parité.
- VAE fp32 ; LM bf16 ; quantification limitée à la voie AR (O5, opt-in).
- Sortie audio : WAV float32 + int16 (pas de FLAC v1).
- Comments de code en **anglais** (repo public) ; PLAN/knowledge en français.

**Questions ancrées aux GATES** : nom du dépôt et emplacement des poids (G-0) ; tolérances real (G-2) ; autorisation quantification AR (G-5) ; suites v2 — encodeur VAE pour covers audio, serveur, iOS (G-8).

