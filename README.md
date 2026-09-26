# yue2-mlx-swift

Port Swift / MLX de **YuE2** (m-a-p, septembre 2026) pour Apple Silicon : paroles et style → partition ABC éditable → tokens sémantiques → latents acoustiques par flow matching → chanson stéréo 48 kHz. Le pipeline complet tourne en local sur un Mac ou un iPhone 15 Pro Max, sans Python à l'exécution.

**Version 1.0.0** : six configurations de référence mesurées et reproductibles, de 65,7 s pour 70 s d'audio (Mac, 4 bits) à 3,4 Go de pic mémoire (iPhone).

| Documentation | |
|---|---|
| [Les six configurations de référence](docs/References.md) | quel pack, quels réglages, pour quelle machine |
| [Benchmarks : méthode, résultats, reproduction](docs/Benchmarks.md) | protocole, tableaux Mac et iPhone, Core AI, comment contribuer |
| [Ligne de commande](docs/CLI.md) | toutes les commandes et options de `yue2` |
| [API Swift](docs/API.md) | `ModelSession`, `YuE2Pipeline`, résidence, checkpoint, porte GPU, backends VAE |
| [iPhone](docs/iOS.md) | intégration, perte du GPU en arrière-plan, mesures sur l'appareil |
| [Poids et packs](docs/Weights.md) | téléchargement, packs pré-quantifiés, licence, republication |
| [Base de connaissances](docs/knowledge/index.md) | journal des mesures, décisions et pièges (OKF, lisible par humains et agents) |
| [Changelog](CHANGELOG.md) | 0.1 → 1.0.0 |

## Démarrer

Prérequis : macOS 15+ (macOS 27 pour Core AI), Xcode 26+ (Swift 6), Apple Silicon, ≈ 8 Go d'espace pour les poids.

```bash
# build (xcodebuild, jamais `swift build` pour un binaire : la metallib de MLX ne serait pas trouvée)
xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build
BIN=.xcodebuild/Build/Products/Release/yue2

# poids (Hugging Face, CC-BY-NC-4.0) et tokenizer
export YUE2_MODELS_DIR=$HOME/Library/Caches/models
$BIN download --models-dir $YUE2_MODELS_DIR
$BIN info

# une chanson complète avec une configuration de référence
$BIN references
$BIN generate --request reference/yue/examples/song.json --reference 4bit-fast --out outputs/song --profile
```

La première utilisation d'un pack quantifié le produit localement (quelques minutes, une fois). Le fichier de requête suit le format de référence : `style`, `lyrics` (balises `[Verse]`, `[Chorus]`…), `cot`, `seed`, `abc` optionnel. Les artefacts (`score.abc`, `semantic.npy`, `latent.npy`, `audio.wav`, `result.json`, `trace.json`) sont écrits dans `--out`.

Par étapes, chacune reprenable :

```bash
$BIN plan       --request song.json --out outputs/plan
$BIN semantic   --plan outputs/plan --out outputs/song
$BIN synthesize --plan outputs/plan --semantic outputs/song --out outputs/song --checkpoint-dir outputs/song
$BIN decode     --latents outputs/song/latent.npy --out outputs/song/audio.wav --precision fp16
```

Depuis Swift :

```swift
let profile = YuE2ReferenceProfile.named("4bit-lean")!; profile.applyGlobalPolicy()
let session = try await ModelSession.load(modelsDir: dir, quant: profile.quant, quantizeHead: profile.quantizeHead,
                                          precision: profile.precision, residency: [.ar])
let vae = try await loadVAEBackend(.mlx, directory: dir.appendingPathComponent("YuE2-Vae"), precision: .fp16)
let song = try await YuE2Pipeline(session: session, vae: vae, config: session.config).generate(request: request)
```

## Les six configurations de référence

Chanson de 70 s d'audio, 32 pas d'ODE, MacBook Pro M3 Max, GPU libre, 120 s de refroidissement ([détail](docs/References.md)).

| Id | Pack | Temps | Pic mémoire | Pour |
|---|---|---|---|---|
| `4bit-fast` | int4-mixed-head, tout résident, NAR bf16 | **65,7 s** | 8,9 Go | Mac : plus vite que le temps réel |
| `4bit-lean` | int4-mixed-head, résidence par étape, fp16, tuile 256 | 77,7 s | **3,4 Go** | iPhone 15 Pro Max, Mac 8 Go |
| `8bit-fast` | qint8-all-head, tout résident, NAR bf16 | 76,5 s | 9,9 Go | Mac, voie AR en 8 bits |
| `8bit-lean` | qint8-all-head, résidence, fp16, tuile 256 | 79,3 s | 4,4 Go | iPhone avec marge, Mac 8-16 Go |
| `16bit-fast` | bf16, tout résident | 84,9 s | 12,0 Go | référence qualité |
| `16bit-lean` | bf16, résidence par étape, tuile 256 | 84,9 s | 6,9 Go | Mac 16-24 Go |

Sur l'iPhone 15 Pro Max (`4bit-lean`) : 30 s de chanson en 273 s, 60 s en ≈ 12,7 min avec le throttling thermique, première écoute validée ([mesures](docs/iOS.md)).

## Ce que fait le port

1. **Reproduire fidèlement l'inférence YuE2** en MLX Swift, module par module, avec une preuve numérique à chaque étape (fixtures PyTorch, tolérances fixées avant le portage : VAE fp32 max 2e-3, LM bf16 rel < 2 %, tokens greedy exacts).
2. **Tenir dans un téléphone** : quantification 4 / 8 bits par voie, résidence des poids par étape (le budget est le max des étapes, pas leur somme), checkpoint par pas d'ODE et porte GPU pour survivre à iOS qui coupe le GPU en arrière-plan, limites mémoire calées sur la mémoire disponible.
3. **Mesurer avant d'affirmer** : tout chiffre vient de `swift-mlx-profiler`, avec refroidissement, A/B/B/A et contrôle ; le journal des mesures, des décisions et des pièges est dans `docs/knowledge/`.
4. **Servir de banc d'essai pour l'exécution d'un plan par des agents de code** : fiches de tâches dans `tasks/`, plan dans `plan/`.

Hors périmètre : MERT2 et SheetSage2 (covers depuis l'audio), `cot: off` (supporté, non mesuré), reproduction bit à bit du générateur aléatoire PyTorch (parité sur bruit injecté).

## Parité et tests

`Scripts/run-tests.sh` (swift-testing, parallélisation 1 : interblocage connu dans mlx-swift). Deux niveaux : sans poids (fixtures tiny sous `parity/`, 149 tests, toujours verts) et avec les vrais poids (`YUE2_MODELS_DIR` ; `Real*`, `Quantization`, `GenerationSmoke`). `yue2 parity vae|lm|nar` contre `Scripts/reference/real_fixtures.py`. Environnement Python de référence : `Scripts/setup-reference-env.sh`.

## Organisation du dépôt

```
Sources/YuE2Core/      Protocol, Tokenizer, Loading, Models/{LM,VAE}, Generation, Synthesis, Pipeline, Audio, Memory, Backends, CoreAI, Configuration
Sources/YuE2CLI/       yue2 : info, download, generate, plan, semantic, synthesize, decode, encode, references, parity, profile, bench-coreai-nar
Sources/YuE2BenchUI/   yue2-bench-ui : génération, mesures (profiler), historique, lecteur
Tests/YuE2Tests/       swift-testing, deux niveaux
docs/                  documentation (ci-dessus) et base de connaissances
BENCHMARKS.md          toutes les mesures, une ligne par run, source unique = swift-mlx-profiler
Scripts/               build, tests, parité, fixtures Python, bancs, export Core AI
parity/                fixtures tiny committées
plan/, tasks/          le plan de portage et les fiches d'exécution par agent
```

## Licences

Code de ce dépôt : **MIT**. Port indépendant, sans code copié, de l'architecture publiée par [m-a-p](https://huggingface.co/m-a-p) et de son [implémentation de référence](https://github.com/multimodal-art-projection/YuE) (Apache-2.0).

Poids `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae` et tokenizer : **CC-BY-NC-4.0**, usage non commercial, téléchargés séparément depuis Hugging Face et jamais distribués ici. Le `tokenizer.json` est dérivé de celui de `Qwen/Qwen2.5-0.5B` (Apache-2.0), dont l'équivalence avec `qwen.tiktoken` a été vérifiée token par token.
