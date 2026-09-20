## Annexe C — Commandes de référence

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

