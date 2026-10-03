# Mémoire d'agent — ce qu'une nouvelle session doit savoir sans le redécouvrir

Transcription dans le dépôt de la mémoire persistante de l'agent (Claude Code) au 2026-10-02, pour qu'un contexte neuf ou un autre agent parte du même point. Mise à jour à chaque jalon. Rien ici n'est dérivable du code ou de l'historique git ; ce qui l'est vit dans `CLAUDE.md`, `docs/` et `plan/`.

## Vincent et ses conventions

- Écrit en français ; les documents publics (README, docs/*.md, CHANGELOG, notes de release, résumés de profils) sont en **anglais**, le plan, les fiches et la base de connaissances restent en français.
- Mac M3 Max 96 Go, macOS 27, Xcode 27, Swift 6.4 ; iPhone 15 Pro Max (8 Go, iOS 27) pour les mesures. Dépôts frères sous le même dossier parent : flux-2-swift-mlx, gemma-4-swift-mlx, h3-swift-mlx, ltx-video-swift-mlx, convertvoxtral, qwen38-mlx-swift, swift-mlx-profiler, et l'app **yue2-ios** (PocketAnthem, privée).
- Attentes : mlx-swift épinglé `exact`, jamais `branch:` ; build par `xcodebuild` ; swift-testing en deux paliers ; fixtures de parité dumpées depuis Python avant de porter ; `Scripts/run-tests.sh` (largeur 1) ; aucun chemin absolu commité ; plans en français avec des GATES où l'exécutant s'arrête et demande ; démo en fin de jalon ; profileur d'abord, jamais de `print`.
- « Réalise les correctifs dans la foulée » : implémenter, pas seulement rapporter. Toujours mesurer **le temps avec la mémoire** et raisonner par scénario d'appareil (contraint vs. à l'aise). Ne pas écarter une voie (Core AI, un backend) sur un résultat ancien : vérifier si le blocage est intrinsèque ou un artefact de notre intégration (le VAE Core AI à 33,8 dB était notre zéro-padding ; 54,7 dB une heure après).
- Exemples audio sur les **releases GitHub** (AAC), jamais dans le dépôt. Avant tout test GPU : `pgrep` des jobs tiers (entraînement LoRA gemma 4, serveurs) et attendre ; Vincent prévient quand il a des inférences en cours.
- Publication : `main` distant = un commit squashé par version (`git commit-tree HEAD^{tree} -p origin/main`, push `"${C}:refs/heads/main"`), tag annoté, `gh release create` ; l'historique de travail reste local. HF : namespace `VincentGOURBIN` (majuscules). zsh : toujours `"${C}:refs/…"`.

## État du projet (2026-10-02)

- Publié : v1.4.0 (covers via SheetSage2), v1.3.0 (entrée audio par les entrées natives du modèle), sept profils de référence, trois packs HF, PocketAnthem sur l'App Store (id 6815755463). Page équipe (artefact Claude) à jour en version 9.
- Mémoire 2026-09 : le pic était le VAE (tuile 1024 fp32), pas le NAR ; résidence par étape ; Core AI VAE GPU valide, NAR Core AI 5-7× plus lent ; iOS coupe le GPU en arrière-plan (porte + checkpoints) ; 2 bits « plus de la musique », 3 bits à la limite ; `4bit-tiny` 2,6 Go.
- **SheetSage2 porté en MLX Swift le 2026-10-03** (`SheetSage2Core`, `yue2 transcribe`, v1.5.0) : parité octet pour octet en fp32, fp16 par défaut, pic 3,3-3,5 Go sur Mac ; l'intégration iPhone est confiée à une autre session (Vincent), point d'entrée `docs/iOS.md` § Transcription et plan/14.
- **Entrée audio : voir `handover-audio-input-2026-10-02.md`.** Résumé : fredonnement/chant → ABC fonctionne ; variation et régénération fonctionnent ; inpaint d'un extrait réel et son ancrage réfutés à l'oreille puis **retirés** ; **SheetSage2 → ABC → `cot melody` validé à l'oreille le 2026-10-02** (tourne sur Mac en MPS, env Python séparé, recette dans `docs/CLI.md`). **v1.4.0 publiée le 2026-10-02** (covers SheetSage2, `--abc-prefix-file`, sifflement ; démo Beethoven 5 domaine public + sifflement réel en pièces jointes).
- Reste ouvert d'avant : écoute des profils rapides et des pas d'ODE réduits, remesure iPhone avec la 1.3.0, proposition de fork mlx-core (Q7).

## Pièges durables

- Swift 6 isolation par défaut : un fournisseur `UIColor { traits in … }` hérite de MainActor et piège sur l'appareil seulement (jamais au simulateur) ; `nonisolated` + `@Sendable`. Logs de crash iPhone : `xcrun devicectl … --domain-type systemCrashLogs`.
- `#expect(…, "\(x)")` : le commentaire doit être une chaîne. Swift 6.2 « failed to produce diagnostic » sur une application partielle : écrire la closure.
- Mesures : Spotlight indexe les WAV/npy fraîchement écrits (`.noindex`), un job GPU tiers triple les temps sans qu'aucun chiffre ne paraisse absurde seul.
