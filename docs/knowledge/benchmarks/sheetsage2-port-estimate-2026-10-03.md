# SheetSage2 sur iPhone — mesures Mac et chiffrage du portage (2026-10-03)

> **Mise à jour le soir même : le portage est fait** (v1.5.0, `SheetSage2Core`). Port MLX Swift, M3 Max, fp16 : 30 s en 3,0 s, 500 s en 21 s, 1,6 Go chargé, pic 3,3-3,5 Go — à comparer aux colonnes PyTorch ci-dessous. L'estimation iPhone reste à mesurer (G-9). Détails : `log.md`, `docs/iOS.md`.

Question de Vincent : que coûterait SheetSage2 (audio → partition ABC) sur l'iPhone 15 Pro Max ? Rien n'est porté ; ce document chiffre à partir de mesures Mac (PyTorch MPS, bf16) et des rapports Mac → iPhone **mesurés sur YuE2** (`iphone-15-pro-max-2026-09-22.md`). Les chiffres iPhone sont des **estimations** tant que la porte G-9 (mesure appareil) n'est pas passée.

## Architecture (lue dans le code amont)

| Bloc | Paramètres | Nature |
|---|---|---|
| Front-end mel (STFT 2048 / hop 240, 128 mels, 100 Hz) | — | DSP |
| Sous-échantillonnage ConvNeXt (128 → 512 → 1024, ×4 → 25 trames/s) | 1,2 M | conv1d depthwise k7, GRN |
| MERT-v2-FullSong : 24 blocs Conformer macaron (1024, FFN 4096, 16 têtes, RoPE, conv depthwise k31 + GLU, LayerNorm partout) | 631 M | calcul pur |
| Adaptateurs LoRA rang 64 (fusionnés au chargement) | 12,6 M | à fusionner hors ligne |
| Mélange pondéré des 25 états (softmax : 0,846 sur la dernière couche, ≈ 0 sur les 16 premières) + projection 1024 → 512 | 0,5 M | toutes les couches restent nécessaires |
| Décodeur BART 6 couches × 512, attention croisée, vocabulaire 31 678 | 44 M | AR glouton contraint par grammaire |

Poids amont en fp32 : 2,5 Go (MERT) + 0,23 Go (SheetSage2). Licences : CC-BY-NC-4.0 toutes les deux, accès restreint (gated) — même contrainte que YuE2 ; la redistribution d'un pack converti est à vérifier.

## Mesures Mac (M3 Max, MPS bf16, GPU libre)

| Audio | Total | Encodeur | Décodeur | Pas AR | ms/pas |
|---|---|---|---|---|---|
| 13 s (pop) | 6,0 s | 4,6 s | 1,1 s | 208 | 5,5 |
| 30 s (Beethoven) | 7,9 s | 4,4 s | 3,1 s | 566 | 5,4 |
| 180 s | 20,7 s | 4,4 s | 14,4 s | 2 271 | 6,3 |
| 300 s | 287 s | 6,8 s | 272 s | 3 837 | **70,8** |

- **L'encodeur coûte toujours une fenêtre de 300 s** (7 500 trames) : l'amont complète l'audio par du silence, exprès (« the encoder attends to the complete fixed window »).
- 300 s : le décodeur s'effondre et la mémoire pilote MPS monte à 105 Gio (le Mac swappe). Artefact de PyTorch MPS (le cache KV réel tient en ≈ 150 Mo), pas une propriété du modèle ; à ne pas reprendre dans un port MLX.
- Densité de jetons : 12 à 19 pas AR par seconde d'audio.

### Fenêtre réduite (essai)

| Fenêtre encodée | Pop 13 s : notes à ±0,25 s de la référence | Beethoven 30 s | Temps total pop / Beethoven |
|---|---|---|---|
| 300 s (référence) | 41/41 | 109/109 | 5,7 / 7,7 s |
| 120 s | 40/41 | 57/109 (84 notes) | 2,5 / 4,0 s |
| 60 s | 39/41 | 43/109 (68 notes) | 1,8 / 2,8 s |
| durée réelle | 37/41 (ABC refusé : note finale hors grille) | 59/109 (79 notes) | 1,3 / 2,7 s |

La pop tolère une fenêtre de 60 s (5× moins de trames) ; l'orchestre avec rubato et points d'orgue se dégrade. **Optimisation possible, à valider par genre et à l'oreille, pas une hypothèse de chiffrage.**

## Estimation iPhone 15 Pro Max (fp16, MLX)

Rapports mesurés sur YuE2 : calcul lourd (NAR) 4,1-5,7× plus lent qu'au Mac à froid, ×1,6 de plus après ≈ 60 s de GPU soutenu ; AR (borné par la répartition des noyaux) ≈ 2,7×. Hypothèse : MLX ≈ PyTorch MPS sur l'encodeur.

| Scénario | Encodeur | Décodeur | Total |
|---|---|---|---|
| Extrait de 30 s, fenêtre 300 s | 18-25 s | ≈ 8 s (566 × ≈ 15 ms) | **≈ 30 s** |
| Extrait de 30 s, fenêtre 60 s (pop) | 3-5 s | ≈ 8 s | ≈ 12 s |
| Chanson de 3 min | 18-25 s | ≈ 35 s, throttlé ≈ 55 s | **≈ 1-1,5 min** |
| Chanson > 5 min | une fenêtre par 100 s au-delà de 300 s (recouvrement 200 s) | | plusieurs minutes |

Mémoire (étape séparée, avant la génération, avec résidence par étape) :

| Précision MERT | Poids | Activations (fenêtre 300 s, SDPA fusionnée) | Pic estimé |
|---|---|---|---|
| fp16 | 1,26 Go + 0,11 Go décodeur | ≈ 0,3-0,4 Go (STFT ≈ 0,25 Go, FFN 61 Mo, KV ≈ 0,15 Go) | **≈ 1,7 Go** |
| int8 | 0,68 Go + 0,11 Go | idem | ≈ 1,1 Go |

Sous le pic de `4bit-lean` (3,4 Go) : ça tient dans le budget actuel de l'app. Téléchargement supplémentaire : 0,8 Go (int8) à 1,4 Go (fp16). La fenêtre fixe (forme statique 7 500 trames, LayerNorm sans BatchNorm) fait aussi de l'encodeur un bon candidat Core AI GPU/ANE (expérience à part, avec repli MLX).

## Effort de portage

Code amont : 5 666 lignes Python, dont **le réseau ≈ 800** (`modeling_mert2.py`, `modeling_sheetsage2.py`) et **le symbolique ≈ 3 100** (`notation_sheetsage2.py` 1 570, `tokenization_sheetsage2.py` 735, `generation_sheetsage2.py` 634, `chord_spelling` 196). Le symbolique (grammaire de décodage, détokenisation, grille de sous-temps, écriture ABC) est le gros du travail, pas le réseau.

| Étape | Contenu | Porte |
|---|---|---|
| 1 | Conversion : fusion LoRA hors ligne, safetensors fp16/int8, fixtures Python (mel, sortie encodeur, logits décodeur, jetons) | fixtures écrites |
| 2 | Front-end mel + ConvNeXt + Conformer en MLX Swift | parité encodeur (rel < 2 % bf16) |
| 3 | Décodeur BART + cache KV + génération contrainte | jetons gloutons identiques sur les deux extraits |
| 4 | Tokenizer + notation → ABC (chemin `melody_only` d'abord) | `score.abc` identique à l'amont |
| 5 | Intégration iOS (étape séparée, résidence, porte GPU) + mesure appareil | G-9 : temps, pic, thermique |

Ordre de grandeur : **8 à 11 jours** de sessions (réseau 2-3, décodage 2, notation 3-4, iOS 1-2), à comparer aux 11 jours du port YuE2, plus gros mais sans post-traitement symbolique.
