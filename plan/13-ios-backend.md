# 13. Jalon 6 — Backend iPhone : MLX 4 bits + Core AI (iOS 27)

> Rév. 2 du 2026-09-20. La rév. 1 raisonnait avec Core ML et un pont MLX ↔ `MLMultiArray` par couche ; elle est **remplacée** : iOS 27 / macOS 27 introduisent **Core AI**, qui succède à Core ML pour les réseaux de neurones (Core ML reste pour l'apprentissage classique), avec une chaîne d'outils PyTorch (`coreai-torch`), des états mutables pour le cache KV, des formes énumérées, du 4 bits natif et un choix GPU / Neural Engine **à la compilation** (`xcrun coreai-build compile --preferred-compute`). Vérifié localement le 2026-09-20 : `CoreAI.framework` présent dans les SDK iOS 27.0 et macOS 27.0 de Xcode 27 ; `coreai-torch 0.4.2` + `coreai-core 1.0.0b2` installables dans `.venv-ref` (Python ≥ 3.11, torch ≥ 2.8 ; nous avons 3.12 / 2.10).
> Même mode d'exécution que le reste du plan (`plan/00-mode-execution.md`), fiches `tasks/T-6.*`. Appareil de développement : **iPhone 15 Pro Max** (A17 Pro, 8 Go, ANE 35 TOPS), iOS 27, appairé (`xcrun devicectl list devices` → `00008130-000235E436F0001C`). Le projet Xcode de l'app est **créé à la main par Vincent** (procédure §13.9), jamais généré, dans le dépôt séparé `../yue2-ios` (YuE2 Studio) ; ce dépôt reste le moteur (`YuE2Core`).

## 13.0 Objectif et cible

« Backend inférable par un iPhone » signifie, dans l'ordre :

1. `YuE2Core` compile et tourne sur iOS (aujourd'hui `platforms: [.macOS(.v15)]`).
2. Un **pack de poids 4 bits** (voies AR et NAR, `embed_tokens`, `lm_head`) d'environ 2 Go, téléchargeable dans l'app.
3. Une app SwiftUI `YuE2Mobile` qui enchaîne les quatre étapes **de façon reprenable** (chaque étape écrit ses artefacts, comme la CLI).
4. Des **backends Core AI** pour les étapes à formes fixes, choisis par mesure, avec repli MLX : le VAE d'abord, la voie NAR ensuite, éventuellement le décodage AR.

**Cible « nord »** (à mesurer) : clip de **30 s en moins de 4 min**, de **60 s en moins de 8 min**, pic mémoire **< 3,5 Go**, sans arrêt thermique, sur l'iPhone 15 Pro Max. Cinq expériences (E1-E5, §13.5) disent si c'est atteignable et avec quels compromis ; c'est la GATE G-9.

**Hors périmètre** : App Store, exécution en arrière-plan (chaque étape est reprenable, l'app reste au premier plan écran allumé), `cot: off`, covers complètes.

## 13.1 Budget mémoire iPhone (8 Go ; limite applicative ≈ 5-6 Go avec l'entitlement `com.apple.developer.kernel.increased-memory-limit`)

| Composant | bf16 (Mac) | Cible iPhone | Note |
|---|---|---|---|
| voie AR (28 × attention + MLP) | 2,8 Go | **0,7 Go** int4 g64 | `--quant int4` existe (T-4.5) ; parité int4 à mesurer (E2) |
| voie NAR (idem) | 2,8 Go | **1,4 Go** qint8 | **mesuré E2 (2026-09-20)** : int4 g64 hors budget (velocity rel 0,15, solve4 rel 0,16 ; l'erreur se compose sur les 64 évaluations de l'ODE, croissance monotone couche 0 → 27) ; qint8 retenu ; int4 g32 = mesure optionnelle |
| `embed_tokens` + `lm_head` | 1,5 Go | **0,4 Go** int4 (0,75 Go qint8) | `RestrictedHead` déquantifie déjà des lignes (piège n°10) |
| `latent_pos_embed.pe` | 100 Mo | **0** | recalculé (sinus) : T-6.2 |
| cache KV (28 × 8 × 128 × 2) | 114 Ko/token | 57 Ko/token fp16 ; **0,17 Go** pour 3 000 tokens | ⚠️ le banc communautaire note un plafond pratique ≈ 1 024 tokens de KV avant jetsam sur iPhone 17 Pro pour des LLM Core AI ; à vérifier chez nous (E5) — d'où la limite de durée |
| activations NAR (L_nar ≈ 1 500 pour 60 s) | < 0,5 Go | < 0,5 Go | SDPA fusionné |
| VAE décodeur | 253 Mo fp32 | 127 Mo fp16 | tuiles de 256 frames |
| **Total résident attendu** | 14 Go | **≈ 3,0-3,5 Go** | NAR en 8 bits ; marge réduite mais dans la cible |

> **Rév. 2, addendum du 2026-09-21 — résidence par étape.** Les traces `swift-mlx-profiler` de la chanson de 64 s en `int4-mixed` (Mac) situent le pic du pipeline dans le **décodage VAE** (tuile 1024 fp32 : +5,5 Go d'activations au-dessus des 3,6 Go de LM encore résidents), pas en fin de NAR ; le NAR stable pèse poids + ≈ 0,3-0,8 Go. Le budget ci-dessus additionne des composants que **aucune étape ne lit tous à la fois** : plan/sémantique = voie AR + `embed_tokens` + `lm_head` ; NAR (cache de préfixe réutilisé) = voie NAR + `vae2llm`/`llm2vae`/`time_embedder` ; VAE = rien du LM. `YuE2Pipeline(releaseWeightsBetweenStages:)` (défaut sur le profil mobile) charge l'AR seul, échange AR → NAR quand le cache de préfixe est prêt, libère tout le LM avant le VAE ; tuile VAE 256 en fp16. Mesuré sur le Mac (profil Mac, caches 2/4/1 Go) : pic process 10,3 → **6,4 Go** (désormais la phase AR : poids + cache) ; NAR 2,55 Go actifs ; VAE 1,1 Go actifs. Le budget iPhone devient **max(étapes) ≈ 2,6-2,9 Go** avec `--quant-head` et le cache mobile de 256 Mo — voir `docs/knowledge/decisions/stage-scoped-weight-residency.md` et `log.md`. Core AI : le décodeur VAE GPU est **de nouveau valide** (54,7 dB tuilé, fenêtres à forme exacte, `pitfalls/coreai-enumerated-shape-zero-padding.md`) à vitesse égale à MLX et footprint plus bas (2,0 Go contre 2,4-3,2 Go) ; la pile NAR Core AI GPU reste 5-7× plus lente mais économise ≈ 0,9 Go sur l'étape NAR — scénario « mémoire contrainte » à trancher à G-9.

> **Addendum du 2026-09-22 — E5 mesuré.** Premier balayage réel sur l'iPhone 15 Pro Max (`docs/knowledge/benchmarks/iphone-15-pro-max-2026-09-22.md`) : AR 25 tok/s, NAR 665 ms → 6,5 s par évaluation de 256 à 1 536 frames, VAE fp16/256 6 s pour 41 s d'audio (Core AI GPU 5,4 s, même pic), pic process **3,2 Go** (préfixe NAR, voie AR résidente), thermique nominale sur 2 min. Projection : 30 s ≈ 3,2 min, 60 s ≈ 8,3 min. Proposition G-9 dans `tasks/ASK.md`.

## 13.2 Budget calcul (chanson de 60 s ≈ 1 500 frames, préfixe ≈ 2 300 tokens)

Mesures Mac (M3 Max, `BENCHMARKS.md`, 63 s) : AR 31,5 s bf16 / 22 s qint8, NAR 45-51 s, VAE 1,4 s. Points de comparaison publics (iPhone 17 Pro, A19 Pro, plus rapide que notre A17 Pro) : décodage Qwen3-0,6B **193 tok/s en Core AI GPU (moteur « pipelined »), 159 tok/s en MLX, ≈ 50 tok/s en Core AI Neural Engine** ; sous charge soutenue le GPU garde ≈ 38 % de son débit de pointe, l'ANE ≈ 67 % (thermique) ; un 2B int8 sur ANE décode à ≈ 27 tok/s.

| Étape | Nature | Mac mesuré | iPhone estimé (MLX GPU) | Levier Core AI | À trancher par |
|---|---|---|---|---|---|
| Plan + sémantique (≈ 2 100 tokens) | borné mémoire, un token par pas | 22-31 s | **≈ 2-3 min** (10-15 tok/s int4) | modèle **à état** Core AI (cache KV en place, recette Llama d'Apple : int4 + états = ×2 vs cache en E/S), moteur GPU pipeliné ≈ +20-30 % vs MLX d'après les bancs publics ; ANE plus lent mais tient mieux la chaleur | E6 (optionnel) |
| NAR : 64 évaluations × 28 couches × 1 500 | ≈ 270 TFLOP, **90 % de matmuls denses à formes fixes** | 45-51 s | **≈ 4-5 min** à 32 pas | **la pile NAR entière comme une fonction Core AI** à formes énumérées (L_nar par paliers), cache K/V du préfixe en entrée ; même asset compilé deux fois : `--preferred-compute gpu` et `neural-engine` ; l'ANE est le seul moyen d'accéder aux 35 TOPS | E4 |
| VAE (60 s) | ≈ 110 convs sur 2,9 M d'échantillons stéréo | 1,4 s | ≈ 8-12 s fp32 MLX | fp16 Core AI ; précédent public : **Stable Audio Open Small** (même famille Oobleck) décode 11,9 s d'audio en 185 ms sur M4 Max en fp16, validé à cos ≥ 0,9999 vs référence | E1 + E3 |
| **Total 60 s** | | 65 s | **≈ 7-9 min** | objectif **≈ 4-5 min** si E4 et E3 sont positifs | G-9 |

## 13.3 Ce qui est transférable en Core AI — analyse par composant (rév. 2)

Ce que Core AI change par rapport à l'analyse Core ML : (a) le **cache KV est un état mutable** du modèle (`state_names`, mis à jour en place) — le décodage AR devient convertible, Apple l'a montré sur Llama 3.1 8B (int4 + états : 33 tok/s sur M1 Max) ; (b) **SDPA est préservé comme opération composite** avec noyaux fournis, y compris sur GPU ; (c) les **formes énumérées et dynamiques** sont supportées ; (d) le placement GPU / ANE est un choix de compilation, testable sans réécrire ; (e) le pont par couche de la rév. 1 disparaît : une étape = une fonction, une traversée `NDArray` par appel.

| Composant | Formes | Ops (couverture `coreai-torch` vérifiée) | Précision | Verdict | Expérience |
|---|---|---|---|---|---|
| **Décodeur VAE** (Oobleck) | énumérées : `[1, 64, T]`, T ∈ {288, 544, 1056} | `convolution` (transposée incluse), `sin`, `pow`, `sigmoid` : supportés | fp16 | ✅ **candidat idéal** (précédent Stable Audio). ⚠️ Défaut ouvert (FB24322424) : la conv transposée donne des résultats faux **sur CPU** pour noyau ≥ 8 ou stride ≥ 16 — nos noyaux 12/10/8 sont concernés ⇒ **jamais de spécialisation CPU**, forcer GPU ou ANE ; `output_padding` corrigé en 0.4.2 | E1 (fp16 en MLX) puis E3 |
| **Encodeur VAE** | `[1, 2, S]` énumérées | idem | fp16 | ✅ même chemin (remix, covers) | E3 |
| **Pile NAR complète** (28 couches : normes, projections, RoPE, SDPA, MLP, `vae2llm`/`llm2vae`, time embedder, `pe`) | `x_t [L, 64]`, `t`, `K_ar/V_ar [28, L_ar, 8, 128]` en entrée ; L ∈ {256, 512, 1024, 1536, 2048}, L_ar énuméré ou dynamique | `rms_norm`, RoPE et SDPA composites ; `cat` pour concaténer le cache | fp16 (int4 sur les poids via `coreai-opt`) | ✅ **le gros enjeu** : une fonction, 64 appels par chanson, placement GPU vs ANE comparé sur le même asset | E4 |
| **Décodage AR** | prefill `[1, N]` dynamique + décodage `[1, 1]`, états `keyCache/valueCache [28, 1, 8, ctx, 128]` | comme la recette Llama d'Apple ; `RestrictedHead` = deux fonctions (ABC / sémantique) partageant les poids ; q/k-norm et RoPE composites | int4 + fp16 | ⚠️ **optionnel** : gain attendu +20-30 % vs MLX (GPU pipeliné), coût = seconde implémentation du sampling/CFG côté Swift ; à ne faire que si E5 montre que l'AR domine | E6 |
| Tokenizer, protocole, sampling, ODE midpoint | — | CPU / MLX | — | Swift inchangé | — |

Conclusion attendue : **MLX pour l'AR et l'orchestration** (code existant, un seul chemin Mac + iOS), **Core AI pour le VAE et la pile NAR** (formes fixes, compilées à l'avance, placement mesuré), l'AR en Core AI seulement si les chiffres l'exigent.

## 13.4 Architecture logicielle

```
Sources/YuE2Core/
  Backends/VAEDecoding.swift          // protocol { decodeTile(_ z: MLXArray) -> MLXArray } ; MLX = YuE2VAE existant
  Backends/NARVelocityBackend.swift   // protocol { velocity(state:, rawT:, prefixKV:) -> MLXArray } ; MLX = CachedNAR existant
  CoreAI/NDArrayBridge.swift          // MLXArray ↔ NDArray fp16/fp32 (copie explicite ; zero-copy MTLBuffer si l'API l'expose)
  CoreAI/CoreAIVAEDecoder.swift       // VAEDecoding via AIModel(contentsOf:) + InferenceFunction ; forme énumérée ≥ tuile, pad, crop ; repli MLX
  CoreAI/CoreAINARStack.swift         // NARVelocityBackend : bucket L, padding + masque, cache K/V passé une fois par chunk ; repli MLX
  CoreAI/CoreAIARDecoder.swift        // (E6, optionnel) fonctions prefill/decode à états
  Runtime/YuE2Runtime.swift           // precision, quant, backends (mlx|coreai-gpu|coreai-ane|auto), odeSteps, maxSemanticTokens, tileFrames, thermalPacing
  Memory/YuE2MemoryManager.swift      // profil iOS : cacheLimit 256 Mo, GPU.set(memoryLimit:), clearCache entre étapes
Apps/YuE2Mobile/                      // projet Xcode créé à la main (§13.9) ; .xcodeproj committé, DEVELOPMENT_TEAM hors dépôt (xcconfig gitignoré)
  App, RequestView, StageRunner, ProgressView, PlayerView, BenchView, Settings ; Resources/*.aimodel (ou téléchargés)
Scripts/coreai/
  export_vae.py                       // coreai-torch : torch.export du décodeur/encodeur upstream (weight-norm fusionnée), formes énumérées, fp16 → .aimodel
  export_nar_stack.py                 // pile NAR : module PyTorch re-écrit (« re-authoring ») autour de YuE2ForCausalLM upstream, K/V en entrée, buckets de L
  export_ar_stateful.py               // (E6) prefill/decode à états, deux têtes restreintes
  compile.sh                          // xcrun coreai-build compile <asset> --platform iOS|macOS --preferred-compute gpu|neural-engine
  check_asset.py                      // parité Python : sortie Core AI (runtime Python coreai) vs PyTorch fp32 → SNR / cos, PSNR par tenseur via coreai.save_intermediates
```

Principes : un backend = un protocole + repli MLX + kill-switch (`YUE2_DISABLE_COREAI=1`) ; les **fixtures de parité existantes servent telles quelles** (`yue2 parity vae|nar --backend coreai-gpu|coreai-ane`) ; le pipeline reste celui de la CLI ; `swift-mlx-profiler` mesure aussi sur iOS ; le **Core AI Debugger** (comparaison PSNR par opération contre les intermédiaires PyTorch) est l'outil de bissection pour tout écart de parité côté Core AI.

## 13.5 Expériences préalables (Mac d'abord) — chacune est une fiche

| # | Question | Méthode | Go si | Fiche |
|---|---|---|---|---|
| **E1** | Le décodeur VAE en **fp16** est-il inaudible vs fp32 ? | `YuE2VAE(precision: .fp16)` en MLX ; SNR et max abs sur les latents réels ; paire de WAV | SNR > 40 dB, écoute OK | T-6.1 |
| **E2** | **int4** sur toute la voie NAR (et AR) et activations fp16 gardent-ils la parité ? | preset `int4-all`, `precision fp16`, `yue2 parity lm/nar`, chanson complète à écouter, taille du pack | LM greedy ≥ 7/8, rel < 5e-2 ; NAR velocity rel < 5e-2, solve4 rel < 8e-2 ; écoute OK. **Résultat partiel** : AR int4 rel 0,0075, 8/8 ✅ ; NAR int4 ✗ (rel 0,15) ⇒ preset mixte **AR int4 + NAR qint8** (`int4-mixed`) | T-6.2 |
| **E3** | Le décodeur VAE **Core AI** est-il exact et rapide, sur GPU et sur ANE ? | `export_vae.py` → `.aimodel` fp16 ; `compile.sh` en `gpu` et `neural-engine` ; parité (SNR, cos) vs MLX fp32 ; latence par tuile sur le Mac ; **jamais CPU** | SNR > 40 dB (Stable Audio : cos ≥ 0,9999) ; latence ANE ≤ 3 × MLX Mac | T-6.3 |
| **E4** | La **pile NAR** en Core AI, GPU vs ANE, bat-elle MLX ? | `export_nar_stack.py` (buckets L, K/V en entrée) ; parité `velocity`/`solve4` vs fixtures NAR ; temps par évaluation par bucket sur Mac (GPU et ANE), coût de la traversée `NDArray` ; extrapolation 64 évaluations | parité tenue ; temps NAR extrapolé sur iPhone < 60 % du MLX (point iPhone fourni par E5) | T-6.4 |
| **E5** | Que fait vraiment l'**iPhone** ? | app Bench : chargement int4, 64 tokens AR (tok/s), 4 évaluations NAR MLX **et** Core AI (GPU, ANE) sur 256 et 1 024 frames, tuile VAE MLX et Core AI, pic mémoire, thermique | tableau rempli ; ⛔ **G-9** | T-6.5 |
| **E6** (optionnel, après G-9) | Le décodage AR en **Core AI à états** vaut-il une seconde implémentation ? | `export_ar_stateful.py` (recette Llama/Qwen3 d'Apple adaptée : q/k-norm, deux `lm_head` restreints, contexte énuméré 2 048/4 096), parité greedy 8/8, tok/s vs MLX sur l'iPhone | ≥ +25 % de débit à parité, sans dépasser le budget mémoire | T-6.13 |

## 13.6 Jalon 6 — fiches

Ordre imposé : T-6.0 → T-6.1 → T-6.2 → T-6.3 → T-6.4 → T-6.5 (⛔ G-9) → T-6.6 → T-6.7 → T-6.8 (⛔ G-11) → T-6.9 / T-6.10 (selon G-9) → T-6.11 (⛔ G-10) → T-6.12 → T-6.13 (si G-9 le demande).

| Fiche | Titre | Sortie |
|---|---|---|
| T-6.0 | `YuE2Core` compile pour iOS | build iOS vert, suite Mac inchangée |
| T-6.1 | E1 : VAE fp16 en MLX | SNR, paire à écouter |
| T-6.2 | E2 : preset `int4-all`, `pe` recalculé, fp16, pack complet | parités, taille, paire à écouter |
| T-6.3 | E3 : VAE Core AI (export, compile GPU/ANE, `VAEDecoding`) | `yue2 parity vae --backend coreai-*`, latences |
| T-6.4 | E4 : pile NAR Core AI (export, buckets, `NARVelocityBackend`) | parité NAR, temps par bucket GPU/ANE |
| T-6.5 | E5 : projet Xcode (manuel, §13.9) + écran Bench sur l'iPhone ⛔ G-9 | mesures réelles, décision |
| T-6.6 | `YuE2Runtime` + profil mémoire/thermique iOS | config unique, kill-switches, pacing |
| T-6.7 | Pipeline reprenable dans l'app | clip généré en plusieurs sessions |
| T-6.8 | Téléchargement du pack int4 + assets `.aimodel` + licence ⛔ G-11 | pack hébergé, reprise, SHA-256 |
| T-6.9 | Intégration VAE Core AI dans l'app (si E3 go) | VAE Core AI, repli MLX |
| T-6.10 | Intégration pile NAR Core AI (si E4 go) | NAR Core AI, parité inchangée |
| T-6.11 | Première chanson sur l'iPhone (30 s) + `BENCHMARKS.md` iOS ⛔ G-10 | écoute, chiffres |
| T-6.12 | Réglages appareil (ODE, tuile, KV, thermique) | tableau A/B, défauts |
| T-6.13 | E6 : décodage AR à états en Core AI (optionnel) | parité, débit vs MLX |

## 13.7 Risques

| Risque | Impact | Mitigation |
|---|---|---|
| fp16 dégrade le VAE (SnakeBeta `sin(x·α)²`, dépassements) | VAE reste MLX fp32 sur GPU iPhone (≈ 10 s / 60 s) | E1 avant E3 ; précédent Stable Audio rassurant ; `coreai.save_intermediates` + Debugger pour localiser |
| Conv transposée fausse sur CPU (FB24322424, noyaux ≥ 8) | audio faux si Core AI replie sur CPU | spécialisation GPU/ANE explicite ; test de parité obligatoire sur l'appareil ; jamais `preferredComputeUnitKind` CPU |
| Formes : L_ar (préfixe) varie de 500 à 6 000 tokens | trop de variantes énumérées pour la pile NAR | L_ar dynamique si `coreai-torch` l'accepte avec SDPA (à vérifier en E4), sinon paliers de 512 avec masque |
| int4 sur la voie NAR casse la qualité | **avéré (E2)** : NAR en qint8 (1,4 Go) ; la même règle s'applique aux assets Core AI du NAR (`coreai-opt` int8, pas `w4`) | E2 ; int4 g32 en mesure optionnelle |
| Traversée `NDArray` coûteuse | négligeable désormais (une par appel, K/V passés une fois par chunk) | mesurée en E4 |
| Thermique : l'A17 Pro réduit sa fréquence GPU après 1-2 min | temps réels 1,5-2× | pacing ; l'ANE tient ≈ 67 % de son débit de pointe contre ≈ 38 % pour le GPU sous charge soutenue (banc iPhone 17 Pro) — argument pour l'ANE sur le NAR |
| Plafond pratique de cache KV (≈ 1 024 tokens observés sur des LLM Core AI, iPhone 17 Pro) | prompts longs (ABC de 4 096 tokens) impossibles | mesurer notre plafond (E5) ; limiter `maxSemanticTokens` et la longueur d'ABC dans le profil mobile |
| Chaîne d'outils jeune (Core AI sorti en juin 2026, `coreai-torch` 0.4.2, `coreai-core` 1.0.0b2) | conversions qui échouent, régressions entre versions | versions épinglées dans `Scripts/coreai/requirements.txt` ; comparer Mac (macOS 27) et iPhone (iOS 27) sur les mêmes assets |
| Signature / provisioning | build device impossible | équipe de signature dans `Apps/YuE2Mobile/Signing.xcconfig` (gitignoré) |
| Licence CC-BY-NC-4.0 du pack int4 et des `.aimodel` dérivés | redistribution de dérivés des poids | autorisée non commercialement avec attribution ; ⛔ G-11 |

## 13.8 Décisions ancrées aux GATES

- **G-9 (fin E5)** : cible retenue (durée max, pas d'ODE), précision, preset, backends Core AI retenus et placement (GPU/ANE) pour le VAE et le NAR, lancement ou non de E6.
- **G-10 (T-6.11)** : première chanson iPhone écoutée ; réglages (T-6.12) ou arrêt.
- **G-11 (T-6.8)** : hébergement du pack int4 et des assets Core AI sur Hugging Face.

## 13.9 Création manuelle du projet Xcode (Vincent, fiche T-6.5)

1. Xcode 27 → *File › New › Project… › iOS › App*. Product Name `YuE2Mobile`, Team = ton équipe, Organization Identifier `com.vincentgourbin`, Interface SwiftUI, Language Swift, Testing System None, Storage None. Enregistrer dans `Apps/` du dépôt (le dossier `Apps/YuE2Mobile/` contiendra `YuE2Mobile.xcodeproj` et `YuE2Mobile/`).
2. *Project › Info* : iOS Deployment Target **27.0** (Core AI) ; macOS *Designed for iPad* décoché.
3. *File › Add Package Dependencies… › Add Local…* : sélectionner la racine du dépôt (le `Package.swift` de `YuE2Swift`) ; dans la cible `YuE2Mobile`, ajouter les produits **`YuE2Core`** et **`MLXProfiler`**. (mlx-swift compile sa metallib pour iOS dans le build Xcode : rien à faire.)
4. *Signing & Capabilities* : Automatically manage signing ; ajouter la capability **Increased Memory Limit** (entitlement `com.apple.developer.kernel.increased-memory-limit`) ; *Background Modes* non nécessaire.
5. *Info* : `UIFileSharingEnabled` = YES, `LSSupportsOpeningDocumentsInPlace` = YES (pour déposer le pack et récupérer les WAV via Finder), `NSMicrophoneUsageDescription` inutile.
6. *Build Settings* : `SWIFT_STRICT_CONCURRENCY = complete`, `OTHER_SWIFT_FLAGS` vide, `ENABLE_USER_SCRIPT_SANDBOXING = NO` (les scripts de mlx-swift) ; ajouter `Signing.xcconfig` (gitignoré) avec `DEVELOPMENT_TEAM = <ton id>` et l'affecter aux deux configurations dans *Project › Info › Configurations*.
7. Glisser les fichiers Swift fournis par la fiche (`Apps/YuE2Mobile/YuE2Mobile/*.swift`) dans le groupe de l'app ; glisser les assets `.aimodel` dans *Resources* quand E3/E4 les produisent (Xcode 27 les compile à l'avance : *Build Phases › Compile Sources* montre l'étape Core AI ; vérifier dans l'inspecteur du modèle les fonctions et leurs formes).
8. Sélectionner l'iPhone comme destination, *Product › Run*. Première installation : accepter le profil dans *Réglages › Général › VPN et gestion de l'appareil*.
9. En ligne de commande ensuite : `Scripts/ios-build.sh` = `xcodebuild -project Apps/YuE2Mobile/YuE2Mobile.xcodeproj -scheme YuE2Mobile -configuration Release -destination 'id=00008130-000235E436F0001C' -allowProvisioningUpdates build` puis `xcrun devicectl device install app --device <UDID> <chemin du .app>` ; verdict `IOS APP INSTALLED`. Rapatrier un fichier : `xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer --domain-identifier com.vincentgourbin.YuE2Mobile --source Documents/<fichier> --destination .local-runs/`.
