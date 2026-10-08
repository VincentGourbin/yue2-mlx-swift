# Passage de relais : minutage des chansons pour l'app (2026-10-08)

Pour la session de l'app (PocketAnthem). Ce que le moteur fournit pour que la lecture se recale sur les mesures, les notes et les paroles sans rien recalculer. Mesures et méthode : `benchmarks/instrumental-and-karaoke-2026-10-07.md`, section « Minutage par les têtes d'alignement ».

## Ce que l'app appelle

Dans `StageRunner`, juste après `SemanticGenerator(...).generateSemantic(...)`, **avant** `releaseWeights(of: .ar)` :

```swift
let timeline = try AttentionTimeline.timeline(model: s.model, tokenizer: s.tokenizer, semantic: r, cancel: cancel)
try timeline.write(to: songDir.appendingPathComponent("timeline.json"))
```

La passe attend la porte GPU (`YuE2GPUGate`) avant chaque couche et lève `.cancelled` comme les autres étapes. Elle survit donc à un passage en arrière-plan.

- **Coût** : une passe sur les couches 0 à 18, en réutilisant le cache KV de la génération (sans CFG). Sur M3 Max, 1 à 2 s pour 40 à 75 s de musique. Sur iPhone, ce n'est pas mesuré : c'est le premier chiffre à relever.
- **Mémoire** : pas de cache supplémentaire quand celui de la génération est réutilisé. Avec CFG (`guidance` ≠ 1), la passe construit son propre cache pour ces 19 couches.
- **Langue** : déduite du style (« French » → fr, sinon en). Le paramètre `language:` la force.
- **Erreurs** : une chanson sans partition (`cot off`) lève une erreur. À traiter comme « pas de minutage », sans faire échouer la génération.
- **Anciennes chansons (migration)** : `AttentionTimeline.reanalyze(song: dossierDeLaChanson, model:tokenizer:cancel:)` lit `plan/plan.json` et `semantic.npy` (la disposition de l'app), avec les poids AR chargés une fois pour toutes les chansons. Sur le Mac, les 8 chansons du jeu d'essai, copiées telles quelles, passent en 10 s, chargement du modèle compris, avec les mêmes résultats qu'à la génération. La politique reste celle de l'app : à la demande, ou une migration explicite, jamais en douce. L'équivalent Mac est `yue2 karaoke --song <dir> --song <dir>… --skip-existing`.

## Le fichier (`LyricTimeline`, `version` 1)

Tout est trié par temps, en secondes depuis le début de l'audio. `bar` est l'index 0 de la mesure dans la partition. `beat` est le temps dans la mesure, compté à partir de 1 et fractionnaire (2,5 = « et » du 2e temps).

| Champ | Contenu |
|---|---|
| `bars[]` | `index`, `start`, `beats[]` (début de chaque temps), `section` (`intro`, `verse`… si une section s'ouvre là) |
| `notes[]` | notes chantées de la partition, notes liées fusionnées (une note tenue = une note) : `start`, `end`, `midi`, `bar`, `beat`, `beats` (durée écrite en temps) |
| `lines[]` | `text`, `section`, `start`, `end`, `bar`, `beat`, `words[]` |
| `words[]` | `text`, `start`, `end`, `bar`, `beat`, `onScore`, `syllables[]` |
| `syllables[]` | `text`, `start`, `end`, `bar`, `beat`, `notes` (> 1 : mélisme), `firstNote` (index dans `notes`, -1 hors partition) |
| `duration` | secondes d'audio couvertes ; ce que l'audio n'atteint jamais (vers après la coupe à 40 s…) est retiré |
| `grid` | `attention` (horloge des têtes du LM) |
| `tempo`, `meter` | en-tête de la partition, pour l'affichage |

À la lecture : `LyricTimeline.load(from:)` une fois, puis `timeline.position(at: lecteur.currentTime)` à chaque image. Le résultat donne `bar`, `beat` (fractionnaire), `note` (seulement pendant qu'elle sonne), `line`, `word` (dans la ligne) et `syllable` (dans le mot). Ce sont des dichotomies, sans aucun calcul lourd. En détail :
- `bars` sert au curseur de partition, `notes` à la note allumée, `lines`/`words`/`syllables` au karaoké ;
- entre deux temps de `beats`, interpolation linéaire pour une position fine dans la partition ;
- `onScore: false` : le mot est chanté hors des notes écrites (la voix a quitté la mélodie prévue, ou la partition s'arrête avant les paroles). On l'affiche au bon moment, sans allumer de note.

Un instrumental reçoit ses `bars` et ses `notes` (vides si la voix `Vocal` n'a que des silences), sans `lines`.

## Précision mesurée

8 chansons, dont les 4 chansons chantées du jeu d'essai de l'app (`testset-2026-10-07`, int4 iPhone) :
- **Débuts de mots** : 71 à 97 % à ≤ 300 ms de la référence (MMS sur la voix isolée), erreur médiane de 59 à 80 ms. Exception : « Air sifflé », 122 ms, dont la partition s'arrête après 5 mesures.
- **Mesures** : 60 à 160 ms d'écart médian avec une transcription SheetSage2 du rendu.
- **Avance d'affichage** : l'app affiche aujourd'hui les vers 0,35 s en avance (un souffle) et la partition 0,4 s en avance. Ces avances servent l'affichage, pas le recalage ; elles restent un choix de l'app, appliqué sur des temps désormais justes.

## À ne pas refaire côté app

- Recaler la partition sur l'audio par HPSS ou flux spectral : le fichier donne déjà l'horloge réelle, y compris quand le rendu saute une mesure d'intro.
- Répartir les vers par section ou par phrase : les vers chantés hors partition sont déjà placés.

## État côté moteur

Ces changements ne sont ni commités ni publiés au moment de l'écriture. L'app devra monter de version de moteur quand ils le seront. À valider sur iPhone : le temps de la passe, et l'absence de pic mémoire avant la libération des poids AR.
