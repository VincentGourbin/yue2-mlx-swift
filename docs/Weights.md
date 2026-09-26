# Poids, packs et licence

## Ce que le moteur charge

| Répertoire (`$YUE2_MODELS_DIR/…`) | Source | Taille | Licence |
|---|---|---|---|
| `YuE2-3B/` | `m-a-p/YuE2-3B` (bf16, 628 tenseurs) + `tokenizer.json` converti de `Qwen/Qwen2.5-0.5B` | 7,3 Go | CC-BY-NC-4.0 (tokenizer : Apache-2.0) |
| `YuE2-Vae/` | `m-a-p/YuE2-Vae` (décodeur et encodeur Oobleck, fp32) | 0,5 Go | CC-BY-NC-4.0 |
| `YuE2-Vae-legacy/` | ancien VAE, optionnel | 0,5 Go | CC-BY-NC-4.0 |
| `YuE2-3B/mlx-prequantized/<preset>[-head]/model.safetensors` | **produit localement** à la première utilisation d'un preset | 2,4-5,5 Go | dérivé, même licence |
| `YuE2-Vae/coreai/*.aimodel`, `YuE2-3B/coreai/*.aimodel` | `Scripts/coreai/export_*.py` | 0,13 / 1,3 Go | dérivé |

`yue2 download` récupère les deux premiers. Le code de ce dépôt est MIT ; les poids ne sont jamais distribués ici.

## Les packs pré-quantifiés

| Pack | Contenu | Taille | Profils |
|---|---|---|---|
| `int4-mixed-head` | voie AR, embeddings, head en 4 bits (groupe 64, affine), voie NAR en 8 bits, `latent_pos_embed.pe` recalculé | 2,5 Go | `4bit-fast`, `4bit-lean` |
| `qint8-all-head` | tout en 8 bits (groupe 64) | 3,7 Go | `8bit-fast`, `8bit-lean` |
| `int4-head` | voie AR, embeddings, head en 4 bits, NAR bf16 | 3,5 Go | le plus rapide (66-69 s), candidat à un septième profil |
| `qint8`, `qint8-head`, `int4`, `int4-all-head` | variantes de mesure | 2,4-5,5 Go | — |

La première utilisation d'un preset charge le checkpoint bf16 entier (7,3 Go), quantifie, écrit l'export puis continue : plusieurs minutes et 8-14 Go de pic, une seule fois par machine. Les chargements suivants lisent l'export directement (`LMWeightLoader.load`), y compris voie par voie pour la résidence par étape.

Parités des packs (fixtures réelles, `yue2 parity lm|nar`) : AR int4 greedy 8/8 sur les deux phases, rel 0,0075 ; NAR 8 bits rel 0,049 sur un solve de 32 pas (budget 0,10), validé à l'écoute ; NAR 4 bits rel 0,15, écarté.

## Republier des packs de référence sur Hugging Face

Pour éviter la quantification locale (et les 7,3 Go de bf16 sur un iPhone), les packs de référence peuvent être publiés comme dérivés, sous la même licence CC-BY-NC-4.0 avec attribution à m-a-p, usage non commercial. Ce n'est pas fait par ce dépôt : c'est une décision et un compte à l'auteur.

Proposition : un dépôt `VincentGourbin/yue2-mlx-packs` avec un dossier par pack (`int4-mixed-head/`, `qint8-all-head/`, `int4-head/`), le `model.safetensors` exporté tel quel (métadonnées `format`, `quantization`, `quantize_head`, `bits`, `group_size`, `pe`), un `README.md` de carte de modèle, et le SHA-256 de chaque fichier. `Scripts/publish-packs.sh` prépare l'arborescence et imprime les commandes `hf upload` ; il n'envoie rien tout seul.

Une fois publiés, `yue2 download --model packs` (à écrire) les placerait sous `mlx-prequantized/` ; l'app iOS les téléchargerait au premier lancement (2,5 Go pour `4bit-lean`).

## Carte de modèle (brouillon)

```
# YuE2 MLX packs — poids pré-quantifiés pour yue2-mlx-swift

Dérivés de m-a-p/YuE2-3B (CC-BY-NC-4.0) pour le port Swift/MLX
https://github.com/VincentGourbin/yue2-mlx-swift (MIT). Usage non commercial, attribution m-a-p.

| Pack | Bits | Taille | Profils | Parité |
| int4-mixed-head | AR/embed/head 4, NAR 8 | 2,5 Go | 4bit-fast, 4bit-lean | AR greedy 8/8 ; NAR rel 0,049 (32 pas) |
| qint8-all-head | 8 partout | 3,7 Go | 8bit-fast, 8bit-lean | AR rel 0,005 ; NAR rel 0,027 |
| int4-head | AR/embed/head 4, NAR bf16 | 3,5 Go | (rapide) | AR greedy 8/8 |

Format : safetensors MLX (poids packés uint32 + scales + biases, groupe 64, affine), métadonnées
`format=yue2-prequantized-v1`. Chargement : `yue2 generate --reference <profil>` ou
`LMWeightLoader.load(directory:config:quantization:quantizeHead:)`.
```
