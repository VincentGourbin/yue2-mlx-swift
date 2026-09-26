# Ligne de commande `yue2`

Binaire produit par `xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build` (jamais `swift build` pour un binaire : la `default.metallib` de MLX ne serait pas trouvée). `yue2 <commande> --help` donne toujours la liste complète des options.

## Commandes

| Commande | Rôle |
|---|---|
| `info` | version et état des checkpoints |
| `download` | télécharge LM et/ou VAE depuis Hugging Face |
| `generate` | chanson complète : partition ABC → tokens sémantiques → NAR → VAE |
| `plan` | la partition ABC seule (étape 1, reprenable) |
| `semantic` | les tokens sémantiques depuis un plan sauvé (étape 2) |
| `synthesize` | NAR + VAE depuis plan et tokens sauvés (étapes 3-4), avec checkpoint par pas |
| `decode` | latents `.npy` → WAV (VAE seul, MLX ou Core AI) |
| `encode` | audio → latents (encodeur VAE), option aller-retour |
| `references` | liste les six configurations de référence |
| `parity vae|lm|nar` | parité numérique contre les dumps PyTorch |
| `profile plan|semantic` | profil d'une phase AR (TTFT, tok/s, mémoire, trace) |
| `bench-coreai-nar` | NAR MLX contre Core AI GPU/ANE, temps et footprint |
| `remix-experimental` | réécriture SDEdit d'un latent encodé (non validé) |

Variables d'environnement communes : `YUE2_MODELS_DIR` (checkpoints), `YUE2_MEMORY_PROFILE=mac|mobile`, `YUE2_CACHE_LIMIT_MB`, `YUE2_MEMORY_LIMIT_MB`, `YUE2_NAR_COMPUTE=packed|dequantized`, `YUE2_EVAL_PER_LAYER=1|0`, `YUE2_COMPILED_DECODE=1`, `YUE2_DEBUG=1` (journal sur stderr), `YUE2_DISABLE_COMPILE=1` (SnakeBeta non compilé), `YUE2_DISABLE_COREAI=1`.

## `generate`

```bash
yue2 generate --request song.json --reference 4bit-fast --out run/ [--profile]
```

Le fichier de requête : `style`, `lyrics` (balises `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`), `cot` (`full`, `melody`, `off`), `seed`, `id`, `abc` optionnel (partition imposée), `cfg_scale` optionnel.

| Option | Défaut | Effet |
|---|---|---|
| `--reference <id>` | — | applique une des six configurations ([References.md](References.md)) ; les options ci-dessous restent utilisables pour s'en écarter |
| `--quant` | `none` | `none`, `qint8`, `int4` (voie AR seule), `qint8-all`, `int4-all`, `int4-mixed` (les deux voies) |
| `--quant-head` | off | `lm_head` quantifié, calculé par `quantizedMatmul` sur les lignes de la phase |
| `--precision` | `bf16` | `fp16` pour l'iPhone |
| `--ode-steps` | 32 | pas du solveur midpoint |
| `--vae-precision`, `--vae-core-frames` | `fp32`, 1024 (256 en profil mobile) | décodeur fp16 inaudible et plus rapide ; la tuile fixe le pic transitoire |
| `--vae-backend` | `mlx` | `coreai-gpu` (parité 54,7 dB, macOS/iOS 27), `coreai-ane` (impasse aujourd'hui) ; repli MLX journalisé |
| `--release-weights-between-stages` | off (on en profil mobile) | résidence par étape |
| `--nar-compute` | `packed` | `dequantized` : NAR en bf16 pour le solve, +1,4 Go, ≈ 10 % plus rapide sur Mac |
| `--compiled-decode` | off | pas de décodage compilé (mesuré sans gain, laissé en option) |
| `--abc-max-tokens`, `--abc-min-tokens`, `--semantic-max-tokens`, `--semantic-min-tokens` | config du checkpoint | longueur des phases ; 25 tokens sémantiques = 1 s d'audio (1 750 = 70 s) |
| `--temperature`, `--top-p`, `--top-k` | config du checkpoint | sampling des deux phases |
| `--style`, `--lyrics`, `--cot`, `--seed`, `--abc-file`, `--cfg-scale`, `--id` | — | surchargent la requête |
| `--vae` | `standard` | `legacy` pour l'ancien VAE |
| `--format` | `int16` | `float32` |
| `--profile` | off | rapport par phase, métriques TTS, `trace.json` dans `--out` |

Artefacts dans `--out` : `score.abc`, `abc_tokens.npy`, `prefix.npy`, `semantic.npy`, `latent.npy`, `audio.wav`, `plan.json`, `request.json`, `config.json`, `result.json` (temps et hachages), `trace.json` avec `--profile`.

## Par étapes

```bash
yue2 plan       --request song.json --out run/plan
yue2 semantic   --plan run/plan --out run/song
yue2 synthesize --plan run/plan --semantic run/song --out run/song --checkpoint-dir run/song [--resume]
yue2 decode     --latents run/song/latent.npy --out run/song/audio.wav --precision fp16 --core-frames 256
```

`synthesize --checkpoint-dir` écrit `nar-checkpoint.json` et `nar-state.npy` (192 Ko) après chaque pas d'ODE et les efface à la fin ; `--resume` repart du pas sauvegardé, bit-exact avec un solve ininterrompu (même plan, mêmes tokens, même graine, mêmes pas). C'est ce qui rend la perte du GPU (iOS en arrière-plan, crash) inoffensive : un pas perdu, pas l'étape.

`decode` imprime une ligne `DECODE backend=… tiles=… padded_tiles=… seconds=… footprint_before_load_mb=… footprint_after_load_mb=… footprint_decode_peak_mb=… mlx_decode_peak_mb=…`.

## `download`, `info`

```bash
yue2 download --models-dir $YUE2_MODELS_DIR            # lm,vae ; --model lm,vae,vae-legacy
yue2 info
```

Poids `m-a-p/YuE2-3B` et `m-a-p/YuE2-Vae` (CC-BY-NC-4.0), tokenizer converti de `Qwen/Qwen2.5-0.5B`. Les packs pré-quantifiés se créent à la première utilisation d'un preset (`$YUE2_MODELS_DIR/YuE2-3B/mlx-prequantized/<preset>[-head]/`) — voir [Weights.md](Weights.md).

## `references`

```
$ yue2 references
4bit-fast   quant=int4-mixed+head precision=bf16 nar=dequantized compiled=false vae=fp16/1024 residency=all memory=mac ode=32
            pack int4-mixed-head (2,5 Go), tout résident, NAR dé-quantifié en bf16 pour le solve
…
```

## `parity`, `profile`, `bench-coreai-nar`

```bash
yue2 parity vae [--backend coreai-gpu --core-frames 256] [--precision fp16]
yue2 parity lm
yue2 parity nar [--backend coreai-gpu] [--sweep-family …]
yue2 profile semantic --plan run/plan --out prof/ --quant int4-mixed --quant-head
yue2 bench-coreai-nar --frames 256,1024 --backends mlx,coreai-gpu --quant int4-mixed [--iterations 6]
```

Les fixtures réelles viennent de `Scripts/reference/real_fixtures.py {vae|lm|nar|song}` (environnement Python de `Scripts/setup-reference-env.sh`). Verdicts : `PARITY OK …` / `PARITY FAILED …` en dernière ligne.

## `encode`, `remix-experimental`

```bash
yue2 encode --audio input.mp3 --out run/enc --roundtrip
yue2 remix-experimental --request song.json --source-latent run/enc/latent.npy --frames 1000 --noise-fraction 0.7 --out run/remix
```

L'encodeur est validé numériquement (parité, aller-retour écouté) ; le remix est un essai SDEdit hors plan, sans validation.
