## 0. Mode d'exécution — ce plan sera exécuté par un agent (modèle plus petit)

Le « réalisateur » n'est pas une équipe humaine mais un agent IA de capacité inférieure au rédacteur de ce plan. Conséquences :

- **Aucune décision d'architecture n'est laissée à l'exécutant.** Chaque tâche nomme le fichier source à copier/adapter (dépôt frère + chemin), le fichier cible, et le critère de sortie vérifiable par une commande. En cas d'ambiguïté : STOP et question à Vincent, jamais d'improvisation.
- **Une tâche = un commit = un critère vérifiable** (test qui passe, parité sous tolérance, build vert). Interdiction d'enchaîner deux tâches sans avoir validé la première.
- **Les tests de parité sont le harnais de sécurité** : les fixtures Python sont générées AVANT le portage de chaque module (§8), pas après. **Un module sans fixture de référence ne se code pas.** Les fixtures « tiny » (poids aléatoires, < 5 Mo) sont committées ; les fixtures « real » (vrais poids) vivent sous `$YUE2_MODELS_DIR/parity/` et sont gardées par une variable d'environnement.
- **La checklist de pièges (`plan/09-pieges.md`) est un contrat** : chaque revue de tâche coche explicitement les 16 points.
- **Le profiler est l'outil de chasse aux bugs par défaut** : toute anomalie de perf ou de mémoire se diagnostique avec `swift-mlx-profiler` (rapport console + Chrome Trace), pas avec des `print`.
- **La CLI n'utilise que l'API publique de `YuE2Core`** (règle reprise de `convertvoxtral`) : si la CLI a besoin d'un accès interne, c'est l'API qui est incomplète.
- **Interdictions absolues** (erreurs silencieuses) : `swift build` pour produire un binaire (metallib introuvable ; `swift build` sert uniquement à vérifier la compilation) ; `branch:` dans `Package.swift` ; `eval(model.parameters())` global sur le LM avant que les poids soient tous chargés ; commit d'un chemin absolu local ; publication (push, upload HF) sans validation de Vincent ; modification des fixtures committées sans régénération par le script (elles portent un `metadata.seed`).
- **L'ordre des tâches est strict** : chaque tâche ne dépend que des précédentes. Les tâches « optionnelles » du Jalon 4 se font une par une, mesure avant/après obligatoire, et se retirent si le gain est < 5 %.
- **Fin de chaque jalon = démo à Vincent** (CLI puis GUI) : le jalon n'est pas terminé tant que Vincent n'a pas testé.

### 0.1 Points d'arrêt obligatoires (GATES)

Quand l'exécutant atteint une ligne marquée **⛔ GATE**, il s'arrête et pose la question à Vincent, même si une réponse lui semble évidente. Entre deux GATES, il ne pose PAS de question pour des choix déjà tranchés ici. Une ambiguïté réelle non couverte est une G-7.

| GATE | Où | Question à poser |
|---|---|---|
| G-0 | Avant le premier commit | « Plan rév. 1 validé ? Nom GitHub du dépôt (`yue2-swift-mlx`) et licence OK ? `$YUE2_MODELS_DIR` pointe où ? » |
| G-1 | Fin Jalon 1 | Démo : `yue2 decode` d'un `latent.npy` de référence → WAV écouté par Vincent. « On passe au Jalon 2 ? » |
| G-2 | Tâche 2.9 (parité real du backbone) | Présenter erreurs relatives logits/tokens greedy. « Tolérances acceptées ? » |
| G-3 | Fin Jalon 2 | Démo : `yue2 plan` produit un `score.abc` lisible ; `yue2 semantic` produit N tokens à X tok/s. « On passe au Jalon 3 ? » |
| G-4 | Tâche 3.6 (première chanson complète) | Vincent écoute la chanson `examples/song.json` seed 831001. « Qualité acceptable pour continuer ? » |
| G-5 | Début Jalon 4 | Présenter le tableau de mesures (§3.3 rempli). « Quelles optimisations prioriser ? Quantification 8-bit de la voie AR autorisée ? » |
| G-6 | Avant tout `git push` / upload HF / publication | Toujours. Rappel : poids **CC-BY-NC-4.0** (pas d'usage commercial) ; code upstream Apache-2.0. |
| G-7 | Toute déviation du plan (dépendance à ajouter, tâche à réordonner, API à changer) | Décrire la déviation proposée et attendre l'accord. |
| G-8 | Fin Jalon 4 | Démo GUI + BENCHMARKS.md rempli. « Suite : encodeur VAE (covers), serveur, iOS ? » |
