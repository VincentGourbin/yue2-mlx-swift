# 14 — Portage de SheetSage2 (audio → partition ABC) en MLX Swift

Décidé le 2026-10-03 (Vincent : « allez lançons le portage »). Chiffrage et mesures de départ : `docs/knowledge/benchmarks/sheetsage2-port-estimate-2026-10-03.md`. But : transcrire un enregistrement en partition ABC **sur l'iPhone**, puis générer une cover avec `cot: melody` (recette Mac validée à l'oreille le 2026-10-02).

## Référence amont

- `m-a-p/SheetSage2` révision `398b22834dac7dd05e09b9c4e40a39fc479ec502` (code + adaptateurs + décodeur, CC-BY-NC-4.0, gated).
- Parent `m-a-p/MERT-v2-FullSong` révision `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42` (épinglée par `config.json` amont, vérifiée par SHA-256).
- Environnement : `Scripts/setup-sheetsage2-env.sh` → `.venv-sheetsage2/` (Python 3.11, torch 2.8, transformers 4.45.2) et `reference/sheetsage2/` (gitignorés).

## Emplacement

Cible `SheetSage2Core` (produit du même nom) dans ce package : l'app n'épingle toujours qu'un package, la transcription reste séparée du moteur YuE2 (aucune dépendance de `YuE2Core` vers elle). Commande `yue2 transcribe`. Poids convertis (LoRA fusionnés) sous `$YUE2_MODELS_DIR/SheetSage2/`.

## Faits d'architecture (lus dans le code, ne pas redériver)

1. **Front-end mel** (torchaudio) : STFT n_fft 2048, win 2048 Hann **périodique**, hop 240, `center=True` réflexion, puissance 2 ; MelScale 128 bandes **HTK**, sans normalisation, 0-12 kHz ; `10·log10(max(x, 1e-10))`, sans top_db ; **dernière trame retirée** ; `(mel − mean) / max(std, 1e-5)` (buffers du checkpoint). Toujours en float32.
2. **Fenêtre fixe** : l'audio (24 kHz mono, non normalisé) est complété par des zéros jusqu'à 300 s (7 200 000 échantillons → 30 000 trames mel → 7 500 trames encodeur). Le masque de trames valides ne sert qu'en aval ; l'encodeur voit toute la fenêtre.
3. **ConvNeXt** (3 blocs, profondeurs 3/4/5, canaux 128/512/1024, strides 1/2/2) : rééchantillonnage `LayerNorm(eps 1e-6) → Conv1d k2 stride s` (absent pour le bloc 0) ; couche = conv depthwise k7 pad 3 → LayerNorm → Linear ×4 → GELU exact → **GRN sur l'axe du temps** (norme L2 sur T, puis moyenne sur les canaux, + 1e-6) → Linear, résiduel.
4. **Conformer macaron** ×24 (1024, FFN 4096 GELU exact, 16 têtes de 64, LN eps 1e-5) : `x + ½·FFN1(LN x)` ; `x + Attn(LN x)` ; `x + Conv(x)` ; `x + ½·FFN2(LN x)` ; `LN final`. Attention avec biais, **RoPE non entrelacée** (`rotate_half`, base 10 000, positions = trames), SDPA non causale. Module conv : LN → pointwise 1024→2048 sans biais → GLU → depthwise k31 pad 15 sans biais → LN → GELU → pointwise sans biais.
5. **Mélange** : `softmax(layer_weight[25])`, poids 0 sur la sortie du ConvNeXt, puis un poids par bloc ; projection 1024 → 512 (avec biais) = mémoire du décodeur.
6. **Décodeur BART** (transformers 4.45) : 6 couches post-LN, 512, 8 têtes, FFN 2048 GELU, embeddings partagés avec la projection de sortie (sans biais), **positions apprises avec décalage 2**, `layernorm_embedding` après la somme, pas de LN final, pas de mise à l'échelle des embeddings, q mis à l'échelle par `head_dim^-0.5`. Attention croisée sans masque sur toute la mémoire (7 500 trames).
7. **Génération** : gloutonne, contrainte par une grammaire de champs (`generation_sheetsage2.py`), cache KV ; ensuite tokenizer → événements → notation ABC (`tokenization`, `notation`, `chord_spelling`).

## Étapes et portes

**État au 2026-10-03 : S-1 à S-5 passées** (tests `SheetSage2*` dans `Tests/YuE2Tests`), S-6 faite côté Mac (`yue2 transcribe`, `yue2 download --model sheetsage2`, mesures), **reste l'intégration et la mesure sur l'iPhone**, confiée à la session de l'app. Non porté : le validateur d'invariants de l'ABC (`validate_serialized_abc`, contrôle seul, sans effet sur la sortie) et les sorties MIDI/LAB/rendu.

| Étape | Contenu | Porte (l'exécutant s'arrête si elle n'est pas passée) |
|---|---|---|
| S-1 | Environnement de référence ; conversion (fusion LoRA, `save_pretrained` fusionné) ; fixtures tiny (commitées, < 5 Mo) et réelles (CPU fp32 : mel, taps ConvNeXt/Conformer, mémoire, logits des premiers pas, jetons gloutons, ABC) | fixtures déterministes (deux générations identiques) |
| S-2 | Front-end mel + ConvNeXt + Conformer + mélange en MLX | parité tiny ≤ 1e-4 ; réel fp32 rel < 1e-3 sur la mémoire ; précision réduite jugée sur les jetons et l'ABC (fp16 retenu, bf16 écarté) |
| S-3 | Décodeur BART + cache KV + logits | logits réels fp32 rel < 1e-3 sur les premiers pas |
| S-4 | Grammaire de génération + tokenizer | **jetons gloutons identiques** à l'amont sur les deux extraits |
| S-5 | Événements → ABC (chemin `melody_only` d'abord, puis complet) | `score.abc` **identique** à l'amont sur les deux extraits |
| S-6 | `yue2 transcribe`, intégration iOS (étape à part, résidence par étape, porte GPU), quantification int8 éventuelle | G-9 : temps, pic et thermique mesurés sur l'iPhone ; écoute d'une cover |

## GPU

Les fixtures tournent en CPU fp32 (pas de GPU). Les tests de parité Swift utilisent le GPU par courtes rafales ; les mesures de temps (S-2 fin, S-6) demandent le GPU libre et se font sur demande à Vincent. Le simulateur iOS ne fait pas tourner MLX : il sert à compiler et à tester le code symbolique pur (S-4, S-5) et l'interface.

## Pièges connus

- `transformers` 4.45 ne copie pas tous les modules relatifs d'un dépôt local : copier les `.py` dans `~/.cache/huggingface/modules/transformers_modules/<nom du dossier>/`.
- Le cache de modules porte le **nom du dossier** du dépôt local, et le disque du Mac ignore la casse : deux copies nommées `SheetSage2` et `sheetsage2` se marchent dessus (`No module named 'transformers_modules.sheetsage2'`). Une seule copie, en minuscules (`reference/sheetsage2`).
- PyTorch MPS s'effondre au-delà de ≈ 3 000 pas de décodage (mémoire pilote 105 Gio à 300 s) : référence en CPU, ne pas prendre MPS pour vérité sur les longs passages.
- Réduire la fenêtre change la transcription (GRN et attention globales) : 300 s par défaut, toute réduction se valide par genre.
