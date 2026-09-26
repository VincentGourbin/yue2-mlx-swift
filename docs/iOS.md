# iPhone : intégrer, mesurer, tenir dans 8 Go

Le moteur tourne sur iOS 27 (Metal via MLX ; Core AI en option pour le VAE). L'app de référence, `YuE2Studio`, vit dans un dépôt séparé (`yue2-ios`) et épingle ce package en version exacte (`Scripts/pin-engine.sh v1.0.0` de son côté). Ce document dit ce que l'app doit faire et ce qui a été mesuré sur un iPhone 15 Pro Max (A17 Pro, 8 Go, iOS 27.0).

## Projet Xcode

- Deployment target iOS 27 ; dépendance `YuE2Core` (et `MLXProfiler` pour les mesures) ; mlx-swift compile sa metallib pour iOS dans le build Xcode, rien à faire.
- Capability **Increased Memory Limit** (`com.apple.developer.kernel.increased-memory-limit`) : sans elle la limite jetsam tombe sous les 3,4 Go du profil `4bit-lean`.
- `UIFileSharingEnabled` pour déposer le pack via Finder et récupérer les WAV.
- Ne pas lier `CoreAI` explicitement (absent du SDK simulateur ; auto-lié là où `import CoreAI` compile).
- Le Simulator ne sert à rien pour l'inférence : MLX plante au premier tenseur (Metal logiciel) ; appareil physique obligatoire.

## Le profil iPhone

`4bit-lean` : pack `int4-mixed-head` (2,5 Go, déposé dans `Caches/models/YuE2-3B/mlx-prequantized/int4-mixed-head/`), fp16, résidence par étape, NAR packé, VAE fp16 tuile 256, limites mémoire calées sur `os_proc_available_memory` (cache ≈ 1 Go, seuil = disponible − 1,25 Go). `8bit-lean` (4,4 Go) tient aussi. Le bf16 ne tient pas.

```swift
let profile = YuE2ReferenceProfile.named("4bit-lean")!
profile.applyGlobalPolicy()
```

## Les quatre étapes, reprenables

| Étape | Appel | Résident | Artefact |
|---|---|---|---|
| Plan (ABC) | `Planner.plan(request:)` | voie AR + embeddings + head | `plan.json`, `score.abc` |
| Sémantique | `SemanticGenerator.generateSemantic(plan:)` | idem | `semantic.npy` (+ cache KV en mémoire si l'étape 3 suit) |
| NAR | `Synthesizer.synthesize(prefix:codec:seed:reuseCache:onBeforeChunk:resume:onStep:)` | voie NAR (AR libérée dès le cache prêt) | `latent.npy` ; `nar-checkpoint.json` + `nar-state.npy` à chaque pas |
| VAE | `decodeTiled(using: vae, latents, coreFrames: 256)` | rien du LM | `audio.wav` |

Le plus simple reste `YuE2Pipeline` avec les défauts du profil mobile ; persister `onNARStep` dans le dossier de la chanson, reprendre avec `narResume`, recharger un `ModelSession` par chanson.

## Perte du GPU en arrière-plan

iOS coupe le GPU dès que l'app quitte le premier plan (`kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`), sans délai, et mlx-core lève l'erreur dans le gestionnaire Metal : crash impossible à rattraper. Trois mesures dans le moteur, deux gestes dans l'app :

1. `YuE2GPUGate.shared.suspend()` sur `scenePhase == .inactive`, `resume()` sur `.active` et sur toute annulation.
2. Persister chaque `NARCheckpoint` reçu par `onNARStep` ; au retour, reprendre avec `narResume`. Un incident coûte un pas d'ODE plus le re-préfixe (≈ 4 s), pas l'étape.
3. Afficher aux utilisateurs : « Restez dans l'app pendant la synthèse : iOS coupe le GPU en arrière-plan ».

Le résidu (bascule pendant l'animation du multitâche, avant le parcage) ne se ferme qu'avec un patch de mlx-core (enregistrer l'erreur, la relever au prochain `eval`) — proposé en amont, non intégré.

## Ce qui a été mesuré (22 septembre 2026, pack int4-mixed-head, avant les correctifs de v1.0.0)

| Point | Mesure |
|---|---|
| Chargement AR seul | 5,6 s, footprint 3,0 Go (dont ≈ 1,4 Go de trop, corrigé en v1.0.0 : conversion fp16 de la voie parquée) |
| AR, 512 tokens | 25 tok/s, TTFT 0,3 s (Mac bf16 : 60-67 tok/s) |
| NAR, ms par évaluation à 256 / 512 / 1 024 frames | 665 / 1 294 / 2 776 à froid ; 3 700-4 050 à 1 024 une fois `fair` |
| Thermique | `nominal` → `fair` après ≈ 60 s de GPU soutenu, NAR −31 à −46 % ; `serious` sur une chanson de 60 s, l'app fait du pacing (2 s de repos entre pas) |
| VAE, 1 024 frames | MLX fp16/256 : 6,0 s, pic 2,2 Go ; Core AI GPU : 5,4 s, pic 2,1 Go, 2 Go résidents dès le chargement |
| Chanson de 30 s | 273 s, pic 3,6 Go (phase plan ; attendu ≈ 2,2-2,4 Go avec v1.0.0), écoute validée |
| Chanson de 60 s | ≈ 12,7 min hors chargements, pic 3,5 Go |
| NAR Core AI GPU | 4,4 s par évaluation à 256 frames (6,6× MLX), paliers ≤ 1 024 tokens de préfixe : pas retenu |

Règle de mesure sur l'appareil : tout point NAR au-delà de la première minute d'un passage est un point throttlé ; pour une valeur à froid, un point par lancement, téléphone reposé. Enregistrer `SystemMetrics.processFootprint()` avant / après / pic (`FootprintSampler`), `os_proc_available_memory()` au départ, `ProcessInfo.thermalState` avant / après, et la configuration exacte. Un point qui « disparaît » (jetsam, pas d'erreur Swift) est une mesure : c'est la limite.

## À remesurer avec v1.0.0

Pic de la phase plan (correctif fp16), effet des limites adaptatives sur le temps, bascule sur Safari pendant chaque étape (porte + reprise), VAE Core AI GPU sur l'appareil, 16 et 24 pas d'ODE à l'oreille.

## Core AI

`Scripts/coreai/export_vae.py` et `export_nar_stack.py` (`coreai-torch` 0.4.2, `coreai-core` 1.0.0b2) produisent les `.aimodel` ; `xcrun coreai-build compile --platform iOS --preferred-compute gpu` les compile. Jamais de spécialisation CPU pour le VAE (convolution transposée fausse, noyaux ≥ 8). Le Neural Engine est une impasse avec cette version de la chaîne (repli silencieux des convs transposées, crash de `dequantize`).
