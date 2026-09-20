# T-x.y — <titre>

Jalon : x · Prérequis : T-… `validée` · Tier : 1|2 · ⛔ GATE : non|G-n

## Objectif
<une phrase>

## À lire (liste fermée, dans cet ordre)
- `plan/<fichier de section>.md` (un seul fichier de `plan/` par fiche, jamais `PLAN.md`)
- `reference/<dépôt>/<fichier>` lignes L1-L2 (ou `grep "<terme>"` puis 60 lignes)

## Fichiers à créer / modifier
- `Sources/…` (≤ 200 lignes) — rôle

## Étapes
1. …

## Validation (dans l'ordre, arrêt au premier échec)
1. `Scripts/check-build.sh` → `BUILD OK`
2. `Scripts/check-tests.sh <Suite>` → `TESTS OK (n tests)`

## Critère de fin
Toutes les validations ci-dessus vertes dans cette session.

## Pièges à cocher (plan/09-pieges.md)
n°…

## Entrée de journal
```
## T-x.y — <titre> — <date> — validée
- Fait : …
- Bug réel rencontré : …
- Validation : `Scripts/check-build.sh` → `BUILD OK` ; `Scripts/check-tests.sh …` → `TESTS OK (…)`
- Limites connues : …
- Pas d'agent : … · appels d'outils : …
```
