## 4. Jalon 1 — Fondations et VAE (≈ 2-3 jours)

**Objectif** : le paquet compile, se teste, télécharge les poids, charge le tokenizer et le décodeur VAE, et `yue2 decode --latents latent.npy` reproduit l'audio de référence Python. Démo G-1.

### 4.1 Squelette du paquet

```
Package.swift
Sources/YuE2Core/
  YuE2Core.swift                 // version, YuE2Error, YuE2Debug (log gated par env YUE2_DEBUG)
  Protocol/Constants.swift       // §2.1 : ids, CONTEXT, instructions
  Protocol/SongRequest.swift     // validation, text(), guidance
  Protocol/Sampling.swift        // Sampling, GenerationConfig (défauts §2.3), décodage de yue2_generation_config.json
  Protocol/Prefixes.swift        // tokenPrefixes, negativePrefix, chunkRanges
  Tokenizer/YuE2Tokenizer.swift  // swift-transformers + NFC ; encode/decode
  Loading/SafetensorsLoader.swift// loadArrays + diff bidirectionnel (h3 WeightLoader.apply)
  Loading/VAEWeightLoader.swift  // fusion weight-norm + layouts (h3 VAEWeightLoader)
  Loading/LMWeightLoader.swift   // Jalon 2
  Loading/ModelDownloader.swift  // gemma-4 Download/* (URLSession, resume)
  Loading/ModelCatalog.swift     // repos HF, fichiers attendus, licences
  Models/VAE/SnakeBeta.swift     // ltx BigVGANVocoder.swift L15-40
  Models/VAE/OobleckDecoder.swift// §2.5
  Models/VAE/YuE2VAE.swift       // decode, decodeTiled, naturalOutputLength
  Models/LM/…                    // Jalon 2
  Audio/AudioExporter.swift      // ltx AudioExporter.swift (+ variante float32)
  Audio/NumpyIO.swift            // lecture/écriture .npy (int32 1-D, float32 2-D) pour les artefacts
  Pipeline/…                     // Jalon 3
  Memory/YuE2MemoryManager.swift // netflix-void VOIDMemoryManager pattern (cacheLimit par étape)
Sources/YuE2CLI/                 // yue2 : AsyncParsableCommand ; sous-commandes info (défaut), download, decode, plan, semantic, synthesize, generate, parity, bench, profile
Sources/YuE2BenchUI/             // Jalon 4 (cible créée vide, un ContentView « Hello »)
Tests/YuE2Tests/                 // swift-testing ; tier 1 sans poids, tier 2 gardé par YUE2_MODELS_DIR
  TestUtilities/ParityHelpers.swift  // relativeError, maxAbs, loadFixture(name), gate env
parity/                          // fixtures tiny committées (générées par Scripts/reference/tiny_fixtures.py)
Scripts/run-tests.sh             // copie gemma-4 (deadlock ABBA, watchdog, relais TEST_RUNNER_YUE2_*)
Scripts/setup-reference-env.sh   // uv venv .venv-ref + pins + clone upstream + pytest upstream
Scripts/convert-tokenizer.py     // §2.7
Scripts/reference/tiny_fixtures.py, real_fixtures.py   // §8
docs/knowledge/{index.md, log.md, pitfalls/, benchmarks/, decisions/}
CLAUDE.md, AGENTS.md (généré), README.md, BENCHMARKS.md, LICENSE (Apache-2.0 pour le code ; poids CC-BY-NC-4.0 rappelés)
```

Dépendances (`swift-tools-version: 6.0`, `platforms: [.macOS(.v15)]`) — **pas de `mlx-swift-lm`** (le MoT, la voie NAR, le lm_head restreint et le sampling fenêtré sont tous custom ; `TokenIterator` n'apporterait rien et son `LogitProcessor` casse le pipelining) :
```swift
.package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),   // Tokenizers uniquement
.package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
.package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.5.0"),
```
Produits : `.library(name: "YuE2Core")`, `.executable(name: "yue2")`, `.executable(name: "yue2-bench-ui")`. Cibles MLX : `MLX`, `MLXNN`, `MLXRandom`, `MLXFast`.

### 4.2 Tâches

| # | Tâche | Source à copier / adapter | Cible | Critère de sortie |
|---|---|---|---|---|
| 1.0 | `git init`, `.gitignore` (`.xcodebuild/`, `.build/`, `.venv-ref/`, `.local-runs/`, `*.profraw`), `Package.swift`, cibles vides, `CLAUDE.md` (Annexe B), `Scripts/run-tests.sh` | `gemma-4 Scripts/run-tests.sh` ; `h3 CLAUDE.md` (structure) | racine | `xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build` vert ; `Scripts/run-tests.sh` exécute 1 test trivial. ⛔ **GATE G-0** avant ce commit |
| 1.1 | Environnement de référence Python : `Scripts/setup-reference-env.sh` (uv, `torch==2.10.0 transformers==4.57.6 tiktoken==0.12.0 safetensors==0.7.0 numpy==2.2.6 soundfile==0.13.1 huggingface-hub==0.36.2`, `pip install -e <clone YuE>`), puis `pytest tests/` upstream | `qwen38 Scripts/references/README.md` (idée : upstream vendorisé = source de vérité) | `Scripts/`, `.venv-ref/` (gitignoré) | pytest upstream vert (les tests sont CPU, sans checkpoint) ; le SHA du commit upstream est noté dans `docs/knowledge/log.md` |
| 1.2 | Constantes, `SongRequest`, `Sampling`, `GenerationConfig`, `tokenPrefixes`, `negativePrefix`, `chunkRanges` | `protocol.py` (ligne à ligne) | `Protocol/*.swift` | `ProtocolTests` (tier 1) : mêmes cas que `tests/test_protocol.py` + fixture `parity/protocol.json` (préfixes attendus pour `song.json` avec ids d'ABC fictifs) |
| 1.3 | Fixtures tiny + corpus tokenizer : `Scripts/reference/tiny_fixtures.py` (§8.1) génère `parity/tiny_vae.safetensors`, `parity/tiny_lm.safetensors`, `parity/sampling.safetensors`, `parity/tokenizer_corpus.json`, `parity/protocol.json` | `qwen38 Scripts/qwen4-exp-*-reference.py` (structure, `metadata.seed`) ; `tests/test_vae.py::tiny_config`, `tests/test_nar.py::model` upstream | `parity/` | fichiers présents, < 5 Mo au total, `metadata` renseigné ; script idempotent (même seed ⇒ mêmes octets) |
| 1.4 | Téléchargement : `ModelCatalog` (YuE2-3B : `config.json`, `generation_config.json`, `yue2_generation_config.json`, `weights_manifest.json`, `model.safetensors`, `qwen.tiktoken`, `examples/tonight-awake.json` ; YuE2-Vae / -legacy : `config.json`, `model.safetensors`, `weights_manifest.json` ; + `tokenizer.json` depuis `Qwen/Qwen2.5-0.5B`) ; `ModelDownloader` ; vérification SHA-256 via `weights_manifest.json` | `gemma-4 Sources/Gemma4Swift/Download/*` (764 l.) ; `ltx LTXModelCatalog.swift` (licences) | `Loading/ModelDownloader.swift`, `ModelCatalog.swift` | `yue2 download --models-dir $YUE2_MODELS_DIR` remplit `$YUE2_MODELS_DIR/{YuE2-3B,YuE2-Vae}` ; `yue2 info` affiche ✓/✗ par fichier + SHA vérifié |
| 1.5 | Tokenizer : `Scripts/convert-tokenizer.py` (§2.7) + `YuE2Tokenizer` | `gemma-4 Pipeline/Gemma4TokenizerLoader.swift` | `Tokenizer/YuE2Tokenizer.swift` | `TokenizerTests` (tier 2, `YUE2_MODELS_DIR`) : encodage identique à `parity/tokenizer_corpus.json` (≥ 6 textes, dont chinois, ABC, `<|im_start|>` littéral) ; `decode(encode(x)) == x` ; aucun id ≥ 151643 produit |
| 1.6 | `.npy` I/O + `AudioExporter` (int16 + float32) | `ltx Utils/AudioExporter.swift` (102 l.) | `Audio/*.swift` | tier 1 : round-trip npy ; en-tête WAV valide (relecture par `AVAudioFile`), nombre d'échantillons exact, stéréo entrelacé |
| 1.7 | Loader générique + `VAEWeightLoader` (fusion weight-norm, layouts, filtre `decoder.`) | `h3 Loading/WeightLoader.swift` (`loadShardedWeights`, `apply` diff bidirectionnel) + `h3 Loading/VAEWeightLoader.swift` (en-tête = les 4 conversions) | `Loading/*.swift` | tier 1 sur `parity/tiny_vae.safetensors` : fusion weight-norm == tenseur `expected_merged_*` de la fixture (max abs 1e-6) ; **garde de couverture** : 0 paramètre non alimenté, 0 clé inutilisée |
| 1.8 | `SnakeBeta`, `ResidualUnit`, `DecoderBlock`, `OobleckDecoder`, `YuE2VAE.decode` | `ltx Models/AudioVAE/BigVGANVocoder.swift` (SnakeBeta) ; `h3 Models/AudioVAE/H3AudioVAEDecoder.swift` (structure Sequential fp32) ; `modeling_vae.py` | `Models/VAE/*.swift` | `VAEDecoderTests` (tier 1, tiny) : sortie == `expected_full` (max abs 1e-4, fp32) pour T ∈ {1, 15, 17, 33} ; longueur `1920·T - 64` ; **test de bissection** : taps `after_conv_in`, `after_block_1..6`, `after_final` comparés un à un (fixture) et premier écart imprimé |
| 1.9 | `decodeTiled(core:halo:)` + progression | `modeling_vae.py::decode_tiled` | `Models/VAE/YuE2VAE.swift` | tier 1 tiny : `tiled == full` pour (frames, core) ∈ {(1,1), (15,7), (17,8), (33,16), (65,32), (257,256)}, rel 3e-5 / abs 3e-6, y compris ±128 échantillons autour de chaque couture ; refus si halo < 16 |
| 1.10 | Parité **real** VAE : `Scripts/reference/real_fixtures.py vae` (§8.2) + `RealVAEParityTests` (tier 2) + `yue2 parity vae` | `h3 Sources/MiniMaxH3CLI/ParityCommand.swift` (structure) | `Tests/`, `CLI/ParityCommand.swift` | 40 frames de latents réels (issus d'un `latent.npy` upstream ou aléatoires seedés) : max abs < 2e-3, rel < 1e-3 vs torch CPU fp32 ; 1 100 frames tuilé (core 1024) == plein |
| 1.11 | `yue2 decode --latents latent.npy --vae standard\|legacy --out audio.wav [--core-frames 1024]` + `YuE2MemoryManager` | `netflix-void VOIDMemoryManager.swift` | `CLI/DecodeCommand.swift` | Décode un `latent.npy` produit par la référence Python (obtenu via `yue2 real_fixtures.py song` ou fourni par Vincent) ; corrélation > 0,999 avec `audio.flac` de référence sur la même longueur ; écoute par Vincent |

**Critères d'acceptation — ⛔ GATE G-1 (démo Vincent)** : build Release vert ; `Scripts/run-tests.sh` vert (tier 1) ; tier 2 vert avec `TEST_RUNNER_YUE2_MODELS_DIR` ; `yue2 decode` d'un latent de référence écouté et jugé identique à l'audio Python ; `docs/knowledge/log.md` contient le temps de décodage VAE mesuré (3,6 min d'audio).

