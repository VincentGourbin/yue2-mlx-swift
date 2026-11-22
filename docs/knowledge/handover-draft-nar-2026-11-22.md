# Relais : brouillon NAR pour PocketAnthem (2026-11-22)

Destinataire : l'agent qui intègre le moteur dans l'app `yue2-ios` (Vincent, 2026-11-22 : le moteur ici, l'intégration chez un autre agent).

## Ce que le moteur fournit

- `YuE2ReferenceProfile.generationConfig(_:quality:)` avec `.full` (calendrier du profil, AB2 à 32 pas) ou `.draft` (AB2 à `draftODESteps` = 8 pas, 9 évaluations). Mode d'emploi, chiffres et pièges : `docs/iOS.md`, section « Draft NAR ».
- Écoute de Vincent : ab2-8 « top » en 8 bits, qualité bonne aussi en 4 et 16 bits (« Wrinkles and Rain », 251 s).

## État des branches

- Moteur : AB2 et le brouillon sont sur `main` en local (dernier commit fonctionnel `bc1046d`). Rien n'est publié : la dernière version publiée est la 1.10.0. La 1.11.0 attend le « publie » de Vincent.
- App : la branche `engine-ab2` (partie de `release/1.7`, commit `e20f6ed`) passe déjà le moteur en local (`Scripts/pin-engine.sh --local`). Elle branche AB2 (`StageRunner` via `profile.generationConfig`) et l'estimation de durée (`Calibration`, `SongStore.fittedMinutes`). Elle est installée sur l'iPhone et validée (pause, kill et reprise), mais pas fusionnée. Le brouillon n'y est pas branché. Une fois la 1.11.0 publiée : `pin-engine.sh v1.11.0`.

## À prévoir côté app

- Enregistrer la qualité dans l'état de la chanson. Un point de reprise NAR refuse un autre calendrier (`invalidRequest`).
- Brouillon → rendu complet : garder `semantic.npy`, effacer le point de reprise NAR, refaire le NAR et le VAE en `.full`.
- Le réglage de pas de l'utilisateur ne s'applique qu'en `.full`.
- L'estimation de durée d'un brouillon se calcule avec `ODEMethod.ab2.evaluations(steps: 8)`. Une édition `.keep` repasse en midpoint (16).
- Bogue connu de l'app : `StageRecord.seconds` ne compte que la dernière reprise d'une étape interrompue.
