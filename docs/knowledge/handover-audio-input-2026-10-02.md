# Transmission — entrée audio : où on en est, ce qui a été réfuté, par où reprendre (2026-10-02)

Écrit pour la prochaine session (contexte remis à zéro). Lire ceci avant de retoucher quoi que ce soit à l'entrée audio.

## 1. Verdict de Vincent, à l'oreille (2026-10-02)

- `yue2 inpaint` (extrait réel gardé bit-exact au milieu d'une chanson générée) : « l'extrait est simplement mis au milieu de la chanson, c'est juste une chanson avec l'extrait en plein milieu ». Collage, pas remix.
- `--anchor 0.3 / 0.5 / 0.7` (extrait imposé seulement sur les premiers pas du solveur, puis relâché) : « encore pire, un gros gloubi-boulga qui ne veut rien dire ».
- Conclusion de Vincent : **on prend le problème à l'envers.** Injecter des latents réels dans un solveur conditionné par des jetons qui ne les connaissent pas ne donne rien de musical, quel que soit le dosage.

Les mécanismes restent dans le code (expérimentaux, commits locaux non publiés, voir §4) mais **ne sont pas la voie**.

## 2. Ce que le modèle sait faire, et ce qu'il ne saura jamais faire ici

- Entrées du modèle à l'inférence : texte (style, paroles), **partition ABC** (`SongRequest.abc`, `cot` = `full` pour mélodie + accords, `melody` pour mélodie seule et accompagnement libre), jetons sémantiques (sortie de l'AR), latents (état du solveur). Rien d'audio.
- Les jetons sémantiques viennent, à l'entraînement, d'un **tokenizer audio sémantique non publié** (`modeling_vae.py` amont : « this audio VAE is not the unreleased semantic audio tokenizer »). Donc audio → jetons est **impossible** avec les poids disponibles. C'est la raison de fond de l'échec du § 1 : le modèle n'entend pas l'extrait, et aucune astuce sur les latents ne le lui fera entendre.
- L'encodeur VAE (`yue2 encode`) produit des latents acoustiques fidèles (round-trip validé), mais ces latents n'ont pas de sens pour l'AR.

## 3. La bonne voie, celle de l'amont : le symbolique

Le dépôt amont (`docs/covers.md`, `docs/editing.md`, `skills/yue2-music/`) fait exactement « on aime bien ce refrain, on fait un truc autour » en passant par la **partition** :

1. **Transcription** de l'enregistrement par **SheetSage2** (`m-a-p/SheetSage2`, PyTorch, charge MERT-v2-FullSong ; `infer.py source.wav --output cover-score --melody-only` → `score.abc` avec les mélodies vocale et instrumentale, sans accords). Environnement Python séparé (3.10/3.11, torch 2.8, FFmpeg 6.1). La doc amont vise CUDA ; **à vérifier sur Mac** (MPS ou CPU, 632 M paramètres pour MERT, acceptable pour un extrait de 13 s).
2. **Génération** avec cette partition imposée et **`cot = melody`** (recommandé pour les covers : la mélodie est tenue, l'accompagnement est libre, le style vient du prompt). Notre port supporte déjà `cot: melody` et `--abc-file`. `cot = full` garde aussi l'harmonie.
3. **Édition** : la partition est le point d'édition (accords, tempo, sections, paroles), puis re-rendu. Les helpers amont (`abc_tools.py inspect/compare`) vérifient que l'édition ne touche que ce qu'on veut.

Ce que ça donne pour le scénario « extrait → un truc autour » : extrait → SheetSage2 → mélodie du refrain en ABC → on la pose en `% chorus` dans une partition (avec `--abc-prefix-file`, le planificateur écrit le reste autour : intro, couplets, reprises, dans la même tonalité) → `cot melody` → une **cover originale construite sur ce refrain**, chantée par le modèle avec les paroles qu'on donne. C'est un vrai usage, légalement plus propre qu'un collage, et c'est celui que l'amont démontre.

Notre `MelodyTranscriber` (YIN monophonique, CPU) reste valable pour un **fredonnement ou un chant a cappella** ; il n'est pas fait pour un mix. SheetSage2 est le transcripteur polyphonique qu'il faut pour un vrai morceau.

## 4. État du code (commits locaux, non publiés, au-dessus de v1.3.0)

| Commit | Contenu | Statut |
|---|---|---|
| v1.3.0 (publié) | `yue2 melody`, `vary`, `regenerate` ; `MelodyTranscriber`, `NAREdit.variation/.keep`, `SongResult.load` | validé sur fredonnement synthétique ; Vincent a écouté le bench sifflé (techno) sans verdict formel |
| 37a1489 | sifflement : `--min-hz/--max-hz/--octave-shift`, absorption des glissés | utile, à garder |
| 97ac345 | `yue2 inpaint`, `NAREdit.inpaint(ranges:)`, masque par plage, frames gardées exactes en float32 | **réfuté à l'oreille** (collage) |
| 4896941 | `SongRequest.abcPrefix` / `--abc-prefix-file` : début de partition forcé, le planificateur continue | utile, **c'est la brique du § 3** |
| 9c2ab4c | `--anchor` sur inpaint | **réfuté à l'oreille** (bouillie) |

Tests : 156 sans poids verts avant inpaint/anchor ; `NAREditTests` couvre `.variation` et `.keep`, pas `.inpaint`. Les docs CLI/API ne décrivent ni `inpaint`, ni `--abc-prefix-file`, ni les options sifflement. Avant publication : soit retirer `inpaint`/`anchor`, soit les marquer expérimentaux dans la doc avec le verdict.

Fichiers d'écoute (gitignorés) : `.local-runs/bench.noindex/{audio-in,whistle,magic}/release/`. Scripts : `.local-runs/audio-in-bench.sh`, `.local-runs/audio-in-validate.sh`.

## 5. Par où reprendre (proposition)

1. ~~Faire tourner **SheetSage2 sur le Mac** sur l'extrait Magic System 52–65 s (env Python séparé, MPS ou CPU) et regarder le `score.abc` obtenu. C'est la seule inconnue technique.~~ **Fait le 2026-10-02** : MPS, 43 s, partition en La mineur (voir `log.md`) ; recette dans `.local-runs/sheetsage2-bench.sh`.
2. ~~Si la transcription est bonne : `yue2 generate --abc-file score.abc --cot melody` en deux styles, puis la variante « refrain posé en préfixe, le modèle écrit le reste ». Écoute.~~ **Fait** : quatre rendus (`.local-runs/bench.noindex/sheetsage2/release/`), verdict de Vincent : « c'est vraiment bien le rendu ».
3. ~~Décider du sort de `inpaint`/`anchor` (retrait ou doc « réfuté »), documenter `--abc-prefix-file` et le sifflement~~ **Fait** : `inpaint`/`anchor` retirés (revert de 9c2ab4c et 97ac345, Vincent : « je vois pas trop ce que ça pourrait nous apporter »), recette SheetSage2 dans `docs/CLI.md`, CHANGELOG 1.4.0 rédigé. Reste : publier la 1.4.0.
4. Plus tard, si l'usage le justifie : port Swift de SheetSage2 (MERT-v2-FullSong 632 M + tête de transcription) pour l'iPhone, ou service Mac.

Détails de mesure et pièges de la 1.3.0 : `decisions/audio-input-via-model-entries.md`.
