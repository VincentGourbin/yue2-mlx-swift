# yue2-mlx-swift

Port Swift / MLX de **YuE2** (m-a-p, septembre 2026) pour Apple Silicon : paroles et style → partition ABC éditable → tokens sémantiques → latents acoustiques par flow matching → chanson stéréo 48 kHz. Le pipeline complet tourne en local sur un Mac, sans Python à l'exécution.

> Chanson de test du dépôt de référence (`examples/song.json`, seed 831001) : 63 s d'audio générées en **65 s** sur un MacBook Pro M3 Max en bf16, **56 s** avec la voie autorégressive quantifiée en 8 bits. Détails et méthode dans `BENCHMARKS.md`.

## Ce que ce dépôt cherche à faire

1. **Reproduire fidèlement l'inférence YuE2** en MLX Swift, module par module, avec une preuve numérique à chaque étape : chaque composant est comparé à l'implémentation PyTorch de référence sur des fixtures avant d'être considéré comme porté (voir « Parité » ci-dessous).
2. **Réutiliser les briques déjà validées** dans les autres ports MLX Swift de l'auteur (attention Qwen3, chargeur safetensors strict, décodeur audio DAC/Oobleck, export WAV, profiler, GUI de bench) plutôt que de réécrire.
3. **Servir de banc d'essai pour l'exécution d'un plan par des agents de code** : le port a été réalisé par des agents à partir de fiches de tâches (`tasks/`), chacune avec ses lectures, ses validations en ligne de commande et son entrée de journal. Le plan lui-même est dans `plan/`.

## Où on en est (20 septembre 2026)

| Jalon | Contenu | État |
|---|---|---|
| 1 | Fondations : protocole de prompt, tokenizer, téléchargement, décodeur VAE Oobleck fp32, décodage tuilé exact, `yue2 decode` | ✅ validé, parité VAE réelle max 1,4e-3 |
| 2 | Backbone LM (Qwen3-style, deux voies par couche), cache KV, chaîne de sampling, `lm_head` restreint par phase, génération avec CFG, `yue2 plan` / `yue2 semantic`, profiler | ✅ validé, parité LM réelle : 8/8 tokens greedy identiques sur les deux phases |
| 3 | Voie NAR : flow matching sur le cache KV du préfixe, solveur midpoint 32 pas, `yue2 generate` bout en bout | ✅ validé, première chanson écoutée le 18 septembre |
| 4 | Optimisations mesurées A/B/B/A : réutilisation du cache KV pour le NAR, activation SnakeBeta compilée (VAE −30 %), quantification 8 bits de la voie AR (AR −30 %, opt-in `--quant qint8`), GUI de bench SwiftUI | ✅ validé ; deux optimisations tentées puis retirées faute de gain |
| 5 | Encodeur VAE (audio → latents), `yue2 encode`, round-trip audio réel | ✅ code et parité validés ; **écoute du round-trip encore en attente** |
| — | `yue2 remix-experimental` : réécriture SDEdit d'un latent réel sous un nouveau conditionnement | ⚠️ expérimental, hors plan, non validé |

Ce qui a été volontairement laissé de côté : MERT2 et SheetSage2 (transcription pour les covers, inutiles pour générer depuis du texte), le mode `cot: off` (supporté, jamais mesuré), la reproduction bit-à-bit du générateur aléatoire PyTorch (le bruit initial est tiré côté Swift ; la parité se fait avec un bruit injecté).

## Ce qu'il reste à faire

- **Écoutes humaines en suspens** : le round-trip encodeur → décodeur sur un fichier réel (`tasks/ASK.md`, T-5.2) et la comparaison bf16 / `qint8` sur une chanson complète (T-4.5). Les parités numériques sont vertes, mais le juge final est l'oreille.
- **Backend iPhone** : faire tourner le pipeline sur iOS 27 avec les poids en 4 bits et le VAE et la voie NAR en Core AI (GPU ou Neural Engine, choisi par mesure). Plan détaillé : `plan/13-ios-backend.md`, fiches `tasks/T-6.*`.
- **Mesures manquantes** : mode `cot: off` (deux branches CFG), chansons longues (3 à 6 min, plusieurs chunks NAR), `int4` sur la voie AR.
- **Hors périmètre pour l'instant** : serveur d'inférence, covers audio complètes (transcription), quantification de la voie NAR.

## Installation et usage

Prérequis : macOS 15+, Xcode 26+ (Swift 6), Apple Silicon, environ 8 Go d'espace pour les poids.

```bash
# build (xcodebuild, jamais `swift build` pour produire un binaire : la metallib de MLX ne serait pas trouvée)
xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build
BIN=.xcodebuild/Build/Products/Release/yue2

# poids (Hugging Face, CC-BY-NC-4.0) et tokenizer
export YUE2_MODELS_DIR=$HOME/Library/Caches/models
$BIN download --models-dir $YUE2_MODELS_DIR
$BIN info

# une chanson complète
$BIN generate --request reference/yue/examples/song.json --out outputs/song --profile
#   options : --cot full|melody|off  --seed N  --abc-file score.abc  --cfg-scale X  --vae standard|legacy  --quant qint8

# par étapes (chaque étape est reprenable)
$BIN plan      --request song.json --out outputs/plan        # partition ABC
$BIN semantic  --plan outputs/plan --out outputs/song        # tokens sémantiques
$BIN synthesize --semantic outputs/song                      # latents (flow matching)
$BIN decode    --latents outputs/song/latent.npy --out outputs/song/audio.wav
$BIN encode    --audio input.wav --out outputs/enc --roundtrip   # audio → latents (→ audio)
```

Le fichier de requête suit le format de référence : `style`, `lyrics` (avec balises `[Verse]`, `[Chorus]`…), `cot`, `seed`, `abc` optionnel. Les artefacts d'un run (`score.abc`, `semantic.npy`, `latent.npy`, `audio.wav`, `result.json` avec les temps et les hachages) sont écrits dans `--out`.

GUI de bench : `xcodebuild -scheme yue2-bench-ui -configuration Release -derivedDataPath .xcodebuild build` puis lancer le binaire.

## Parité et tests

Deux niveaux, exécutés par `Scripts/run-tests.sh` (parallélisation swift-testing forcée à 1 à cause d'un interblocage connu dans mlx-swift) :

- **Sans poids, toujours vert** : fixtures « tiny » committées sous `parity/`, générées par `Scripts/reference/tiny_fixtures.py` avec les classes PyTorch upstream sur des modèles aléatoires seedés, comparées à 1e-4. Chaque test de parité vérifie d'abord qu'aucun paramètre n'est resté non alimenté.
- **Avec les vrais poids** (`YUE2_MODELS_DIR`) : fixtures produites par `Scripts/reference/real_fixtures.py {vae|lm|nar|song}`, comparées par `yue2 parity {vae|lm|nar}` et par les tests `Real*`. Tolérances : VAE fp32 max 2e-3, LM bf16 erreur relative < 2 %, argmax et tokens greedy exacts.

Suite actuelle : 128 tests. Environnement Python de référence : `Scripts/setup-reference-env.sh` (torch 2.10, transformers 4.57.6, paquet `yue2` upstream).

## Organisation du dépôt

```
Sources/YuE2Core/      bibliothèque : Protocol, Tokenizer, Loading, Models/LM, Models/VAE, Generation, Synthesis, Pipeline, Audio, Memory
Sources/YuE2CLI/       yue2 : info, download, plan, semantic, synthesize, decode, generate, encode, parity, profile
Sources/YuE2BenchUI/   yue2-bench-ui : génération, mesures (profiler), historique, lecteur
Tests/YuE2Tests/       swift-testing, deux niveaux
parity/                fixtures tiny committées
plan/                  le plan de portage, un fichier par section (PLAN.md n'est que le sommaire)
tasks/                 fiches de tâches, état, journal et questions (le protocole d'exécution par agent)
Scripts/               build, tests, parité, fixtures Python, bench
docs/knowledge/        journal des mesures et des pièges rencontrés
BENCHMARKS.md          toutes les mesures, source unique = swift-mlx-profiler
```

Faits d'architecture à connaître avant de toucher au code : `plan/02.1-vocabulaire.md` à `plan/02.8-dtypes.md`. Pièges connus : `plan/09-pieges.md`. Conventions : `CLAUDE.md`.

## Licences

Code de ce dépôt : **MIT**. Port indépendant, sans code copié, de l'architecture publiée par [m-a-p](https://huggingface.co/m-a-p) et de son [implémentation de référence](https://github.com/multimodal-art-projection/YuE) (Apache-2.0).

Poids `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae` et tokenizer : **CC-BY-NC-4.0**, usage non commercial, téléchargés séparément depuis Hugging Face et jamais distribués ici. Le `tokenizer.json` est dérivé de celui de `Qwen/Qwen2.5-0.5B` (Apache-2.0), dont l'équivalence avec `qwen.tiktoken` a été vérifiée token par token.
