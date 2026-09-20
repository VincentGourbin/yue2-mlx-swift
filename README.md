# YuE2Swift

Portage Swift MLX de YuE2 (m-a-p/YuE2-3B + YuE2-Vae) : paroles + style → partition ABC →
tokens sémantiques → latents acoustiques (flow matching) → chanson stéréo 48 kHz.

## Attribution et licence
Portage indépendant (réécriture Swift/MLX, pas de code copié) de l'architecture décrite par
[m-a-p/YuE2](https://huggingface.co/m-a-p) et son
[implémentation de référence](https://github.com/multimodal-art-projection/YuE) (Apache-2.0).

Code de ce dépôt : **MIT** (voir `LICENSE`). Poids et tokenizer : **CC-BY-NC-4.0** (non
commerciaux, m-a-p) — jamais distribués dans ce dépôt, téléchargés séparément depuis Hugging
Face ; ne pas les redistribuer sans licence séparée de m-a-p.

## Commandes
```bash
# build
xcodebuild -scheme yue2 -configuration Release -derivedDataPath .xcodebuild build
# tests tier 1
Scripts/run-tests.sh
# tests tier 1 + 2
YUE2_MODELS_DIR=/chemin/vers/models Scripts/run-tests.sh
# environnement de référence + fixtures
Scripts/setup-reference-env.sh && .venv-ref/bin/python Scripts/reference/tiny_fixtures.py
YUE2_MODELS_DIR=… .venv-ref/bin/python Scripts/reference/real_fixtures.py vae   # puis lm, nar, song
# usage
yue2 download --models-dir $YUE2_MODELS_DIR
yue2 info
yue2 decode --latents outputs/song/latent.npy --out outputs/song/audio.wav
yue2 plan --request examples/song.json --out outputs/plan
yue2 semantic --plan outputs/plan --out outputs/song
yue2 generate --request examples/song.json --out outputs/song [--vae legacy] [--cot melody] [--abc-file edited.abc] [--profile]
yue2 parity {vae|lm|nar} --models-dir $YUE2_MODELS_DIR
```

## Spécification
Voir `PLAN.md` (§2 les faits d'architecture, §9 la checklist de pièges) et `CLAUDE.md`.
