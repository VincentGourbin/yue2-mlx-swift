# AGENTS.md — instructions pour l'agent qui exécute le plan

Tu es l'exécutant du portage **YuE2 → MLX Swift** décrit dans le dossier `plan/` (un fichier par section ; `PLAN.md` n'est qu'un sommaire de 40 lignes). Tu travailles **une fiche à la fois**, en français dans les documents, en anglais dans le code et ses commentaires. Ce fichier est ton mode d'emploi ; tu n'as pas besoin d'en lire d'autre pour démarrer.

## 1. Démarrage d'une session (dans cet ordre, rien d'autre)

1. `read tasks/STATE.md` — indique la fiche en cours et son état.
2. `read tasks/JOURNAL.md` depuis la fin (les 3 dernières entrées : `grep -n "^## " tasks/JOURNAL.md` puis lire à partir de l'avant-dernière ligne trouvée).
3. `read tasks/T-<x>.<y>.md` — la fiche en cours (ou la suivante `à faire` dans `tasks/INDEX.md` dont tous les prérequis sont `validée`).
4. Mets `STATE.md` à jour : `fiche T-x.y · en cours · <date>`.

Le plan est découpé : une fiche renvoie à un fichier `plan/<section>.md` (300 à 1 000 mots). Ouvre **uniquement** le fichier cité, jamais deux fichiers de `plan/` dans le même pas, et jamais `Scripts/plan-concat.sh`. Ne lis pas `PLAN.md` pour chercher une information : il ne contient qu'un sommaire.

## 2. Pendant une fiche

- Lis **uniquement** ce que la section *À lire* de la fiche indique, par tranches de **200 lignes** au plus. Les sources à copier sont sous `reference/` (liens vers les dépôts frères et la référence Python).
- Outils : `read`, `write`, `edit`, `bash`, `grep`, `find`, `ls`. Pour chercher, `grep` un identifiant précis (nom de type, de fonction, de clé de tenseur), jamais une phrase.
- Crée les fichiers listés dans *Fichiers à créer*, avec les noms exacts. Un fichier Swift fait **200 lignes au plus** ; au-delà, découpe comme la fiche l'indique.
- Copie d'abord, adapte ensuite : quand la fiche nomme un fichier source à copier, reprends sa structure et ses noms, puis applique les seules modifications listées.
- Après chaque fichier écrit : `Scripts/check-build.sh`. Ne continue pas sur un `BUILD FAILED`.
- Lance les validations de la fiche **dans l'ordre**, arrête-toi au **premier** échec, corrige, relance.
- Trois échecs consécutifs sur la même erreur ⇒ écris la question dans `tasks/ASK.md` (gabarit ci-dessous), mets `STATE.md` à `bloquée`, termine ta réponse.

## 3. Fin d'une fiche

Une fiche est **validée** quand chacune de ses commandes de validation a affiché sa ligne attendue (`BUILD OK`, `TESTS OK (n tests)`, `PARITY OK …`, `FIXTURES OK`) dans la sortie d'un outil pendant cette session. Alors :

1. Ajoute l'entrée de journal (gabarit de la fiche) à la **fin** de `tasks/JOURNAL.md`, avec les lignes de validation recopiées telles quelles.
2. Passe la fiche à `validée` dans `tasks/INDEX.md` et écris `fiche suivante : T-x.y · à faire` dans `tasks/STATE.md`.
3. Si la fiche porte **⛔ GATE** : écris la question dans `tasks/ASK.md` et **termine** (ne commence pas la suivante).
4. Termine ta réponse par : `FICHE T-x.y VALIDÉE` (ou `BLOQUÉE`) suivi des lignes de validation.

Tu ne fais **rien** qui ne soit pas dans la fiche : pas de refactoring, pas d'optimisation, pas de fichier supplémentaire, pas de dépendance ajoutée. Une amélioration que tu juges utile va dans `tasks/ASK.md`, pas dans le code.

## 4. Commandes de validation (les seules à utiliser)

| Commande | Attendu en dernière ligne |
|---|---|
| `Scripts/check-build.sh` | `BUILD OK` |
| `Scripts/check-tests.sh <Suite>` | `TESTS OK (<n> tests)` |
| `Scripts/check-parity.sh <composant>` | `PARITY OK <composant> …` |
| `Scripts/check-fixtures.sh` | `FIXTURES OK` |

En cas d'échec, la sortie contient les 20 premières lignes utiles ; le journal complet est dans `.local-runs/*.log` (`grep -n "error:" .local-runs/build.log`, puis lire 10 lignes autour). Ne lis jamais un log en entier.

## 5. Interdits (les erreurs sont silencieuses)

- `swift build` pour produire un binaire ou lancer les tests (metallib introuvable) ; `swift build` seulement pour vérifier une compilation rapide si la fiche le dit.
- `branch:` dans `Package.swift` ; changer une version épinglée.
- `eval(model.parameters())` sur le modèle LM.
- Chemin absolu de ta machine dans un fichier committé ; modifier `parity/*.safetensors` autrement que par le script Python.
- `git push`, upload, publication ; supprimer un fichier sous `reference/` ou `$YUE2_MODELS_DIR`.
- Lire un log ou un fichier source de plus de 200 lignes d'un seul appel ; ouvrir plus d'un fichier de `plan/` par pas ; lancer `Scripts/plan-concat.sh`.

## 6. Gabarits

**Entrée de journal** (`tasks/JOURNAL.md`, en fin de fichier) :
```
## T-x.y — <titre> — <AAAA-MM-JJ> — validée|bloquée
- Fait : <fichiers créés/modifiés, une ligne>
- Bug réel rencontré : <un, ou « aucun »>
- Validation : `<commande>` → `<ligne exacte>` (une ligne par validation)
- Limites connues : <ce qui n'est pas couvert, ou « aucune »>
- Pas d'agent : <n> · appels d'outils : <n>
```

**Question à Vincent** (`tasks/ASK.md`, en fin de fichier) :
```
## ASK — T-x.y — <AAAA-MM-JJ>
- Contexte : <2 lignes>
- Ce que j'ai essayé : <3 lignes max>
- Question : <une question fermée si possible>
- Options : A) … B) …
```

## 7. Repères

- Dépôt : `Sources/YuE2Core` (bibliothèque), `Sources/YuE2CLI` (`yue2`), `Sources/YuE2BenchUI`, `Tests/YuE2Tests`, `parity/` (fixtures tiny), `Scripts/`, `tasks/`, `reference/` (liens, non committés).
- Poids : `$YUE2_MODELS_DIR/{YuE2-3B,YuE2-Vae,YuE2-Vae-legacy,parity}`. Jamais en dur.
- Faits d'architecture : `plan/02.1-vocabulaire.md` … `plan/02.8-dtypes.md` (un fichier par section). Pièges : `plan/09-pieges.md` (16 points, cités par numéro dans les fiches).
