# Fiches — ordre imposé (une fiche = un commit = une validation)

État : `à faire` · `en cours` · `validée` · `bloquée`. Une fiche ne démarre que si tous ses prérequis sont `validée`. ⛔ = se termine par une question dans `tasks/ASK.md`.

| Fiche | Titre | Prérequis | État |
|---|---|---|---|
| T-1.0 | Squelette du paquet, scripts, CLAUDE.md ⛔ G-0 | — | validée |
| T-1.1 | Environnement Python de référence + pytest upstream | T-1.0 | validée |
| T-1.2 | Protocole : constantes, SongRequest, Sampling, préfixes | T-1.0 | validée |
| T-1.3 | Fixtures tiny (Python) + corpus tokenizer | T-1.1, T-1.2 | validée |
| T-1.4 | Catalogue + téléchargement des poids, `yue2 info` | T-1.0 | validée |
| T-1.5 | Tokenizer (conversion + Swift) | T-1.3, T-1.4 | validée |
| T-1.6 | Lecture/écriture `.npy` + export WAV | T-1.0 | validée |
| T-1.7 | Loader safetensors + fusion weight-norm VAE | T-1.3 | validée |
| T-1.8 | Décodeur VAE (SnakeBeta, Oobleck) + parité tiny | T-1.7 | validée |
| T-1.9 | Décodage tuilé | T-1.8 | validée |
| T-1.10 | Parité real VAE + `yue2 parity vae` | T-1.9, T-1.4 | validée |
| T-1.11 | `yue2 decode` + gestion mémoire ⛔ G-1 | T-1.10, T-1.6 | validée |
| T-2.1 | Config LM, RMSNorm, MLP, Attention (+ RoPE) | T-1.11 | validée |
| T-2.2 | DecoderLayer (2 voies), Backbone, KVCache, forwardAR | T-2.1 | validée |
| T-2.3 | Loader LM (628 tenseurs) | T-2.2 | validée |
| T-2.4 | Prefill par blocs et positions explicites | T-2.3 | validée |
| T-2.5 | Processeur de logits (chaîne §2.3) | T-2.4 | validée |
| T-2.6 | lm_head restreint par phase | T-2.5 | validée |
| T-2.7 | TokenGenerator (boucle, CFG, asyncEval) | T-2.6 | validée |
| T-2.8 | Planner / SemanticGenerator + artefacts | T-2.7 | validée |
| T-2.9 | Parité real LM + `yue2 parity lm` ⛔ G-2 | T-2.8 | validée (GATE G-2 posée) |
| T-2.10 | CLI `yue2 plan` / `yue2 semantic` | T-2.9 | validée |
| T-2.11 | Profiler LLM + `yue2 profile` ⛔ G-3 | T-2.10 | validée (GATE G-3 posée) |
| T-3.1 | Modules NAR (time embedder, shiftT, pe, chunks) | T-2.11 | validée |
| T-3.2 | CachedNAR : prefill + velocity | T-3.1 | validée |
| T-3.3 | Solveur midpoint | T-3.2 | validée |
| T-3.4 | Synthesizer multi-chunks | T-3.3 | validée |
| T-3.5 | Parité real NAR + `yue2 parity nar` | T-3.4 | validée |
| T-3.6 | Pipeline complet + `yue2 generate` ⛔ G-4 | T-3.5 | validée |
| T-3.7 | Profiler TTS, `--profile`, BENCHMARKS.md | T-3.6 | validée |
| T-4.0 | Mesures de référence avant optimisation ⛔ G-5 | T-3.7 | validée |
| T-4.1 | O4 : cacheLimit par étape | T-4.0 | validée |
| T-4.2 | O2 : réutilisation du cache KV pour le NAR | T-4.1 | validée |
| T-4.3 | O8 : NAR — K/V contigus, précalculs | T-4.2 | validée (retirée, gain < 5 %) |
| T-4.4 | O7 : VAE — compile SnakeBeta/ResidualUnit | T-4.3 | validée |
| T-4.5 | O5 : quantification 8-bit de la voie AR | T-4.4 | validée |
| T-4.6 | O6/O9 : tuilage requêtes NAR, CFG batché (si mesures) | T-4.5 | validée (aucune des deux nécessaire) |
| T-4.7 | O10 : GUI de bench ⛔ G-8 | T-4.6 | validée (GATE G-8 posée) |
| T-5.1 | Encodeur VAE Oobleck + parité (hors plan initial, choix de Vincent après G-8) | T-4.7 | validée |
| T-5.2 | Ingestion audio + `yue2 encode` + round-trip réel | T-5.1 | validée |
| T-6.0 | `YuE2Core` compile pour iOS | T-5.2 | validée |
| T-6.1 | E1 : décodeur VAE fp16 (MLX), SNR + écoute | T-6.0 | validée (écoute en attente) |
| T-6.2 | E2 : preset `int4-all`, `pe` recalculé, fp16, pack complet | T-6.1 | validée (`int4-mixed` — budget de quantification, écoute finale en attente) |
| T-6.3 | E3 : VAE Core AI (GPU et Neural Engine), `VAEDecoding` | T-6.1 | validée — **rouverte pour le GPU le 2026-09-21** : 54,7 dB tuilé après correction du fenêtrage (`planVAETiles`), `PARITY OK` ; option C de fait (GPU valide, ANE impasse) |
| T-6.4 | E4 : pile NAR complète en Core AI, GPU vs ANE | T-6.2, T-6.3 | validée (GPU vert, ANE crash — verdict Mac consigné, décision attend G-9) |
| T-6.5 | E5 : projet Xcode (manuel) + bench sur l'iPhone ⛔ G-9 | T-6.4 | à faire |
| T-6.6 | `YuE2Runtime` + profil mémoire/thermique iOS | T-6.5 | à faire |
| T-6.7 | Pipeline reprenable dans l'app | T-6.6 | à faire |
| T-6.8 | Téléchargement du pack int4 + licence ⛔ G-11 | T-6.7 | à faire |
| T-6.9 | Intégration VAE Core AI dans l'app (si E3 go) | T-6.8 | à faire |
| T-6.10 | Intégration pile NAR Core AI (si E4 go) | T-6.9 | **fermée** (T-6.4b : Core AI GPU 5-8× plus lent que MLX, > seuil 2× de Vincent) — 2026-09-21 : ≈ 0,9 Go de moins que MLX sur l'étape NAR, à garder comme scénario « mémoire contrainte » si G-9 l'exige |
| T-6.11 | Première chanson sur l'iPhone + BENCHMARKS iOS ⛔ G-10 | T-6.9 | à faire |
| T-6.12 | Réglages appareil (ODE, tuile, KV, thermique) | T-6.11 | à faire |
| T-6.13 | E6 : décodage AR à états en Core AI (optionnel, décidé à G-9) | G-9 | à faire |
