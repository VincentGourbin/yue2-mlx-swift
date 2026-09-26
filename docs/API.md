# API Swift de `YuE2Core`

Tout ce qui suit est `public`, documenté dans le code (doc comments) et couvert par les tests. Package `YuE2Swift`, produit `YuE2Core` ; dépendances épinglées (`mlx-swift` 0.31.6 exact, `swift-mlx-profiler` ≥ 1.5). macOS 15+ / iOS 27+, Swift 6, concurrence stricte.

```swift
dependencies: [.package(url: "https://github.com/VincentGourbin/yue2-mlx-swift", exact: "1.0.0")]
```

## Le chemin le plus court

```swift
import YuE2Core

let profile = YuE2ReferenceProfile.named("4bit-lean")!
profile.applyGlobalPolicy()                       // profil mémoire, calcul NAR, decode compilé

let session = try await ModelSession.load(
    modelsDir: modelsDir, quant: profile.quant, quantizeHead: profile.quantizeHead,
    precision: profile.precision,
    residency: profile.releaseWeightsBetweenStages ? [.ar] : .all)
let vae = try await loadVAEBackend(.mlx, directory: modelsDir.appendingPathComponent("YuE2-Vae"),
                                   precision: profile.vaePrecision)
let pipeline = YuE2Pipeline(
    session: session, vae: vae, config: session.config,
    vaeCoreFrames: profile.vaeCoreFrames,
    releaseWeightsBetweenStages: profile.releaseWeightsBetweenStages)

let song = try await pipeline.generate(request: request) { event in /* progression */ }
try song.saveArtifacts(to: outputDirectory)
```

`SongRequest` (style, paroles, `cot`, graine, `abc` optionnel) est `Codable` et lit le JSON de la CLI. `SongResult` porte l'audio (`[1, samples, 2]` fp32), les latents, les tokens et les temps par phase.

## Modèle et résidence

```swift
public final class ModelSession {
    static func load(modelsDir:quant:quantizeHead:precision:residency:) async throws -> ModelSession
    var model: YuE2ForCausalLM; var tokenizer: YuE2Tokenizer; var config: GenerationConfig
    private(set) var resident: YuE2WeightResidency      // .ar, .nar, .all
    func loadWeights(of: YuE2WeightPath) throws           // no-op si déjà résident
    func releaseWeights(of: YuE2WeightPath) -> Int        // octets libérés
    func releaseAllWeights() -> Int
}
```

Une voie libérée est inutilisable tant qu'elle n'est pas rechargée (elle calculerait sur des zéros) : c'est le contrat de la résidence par étape. Le chargement safetensors est paresseux ; ne pas évaluer un tenseur, c'est ne pas l'allouer. `applyPrecision(_:where:)` ne convertit que les voies résidentes (convertir une voie parquée la matérialise : 1,44 Go, le pic iPhone de la 0.2.1).

`YuE2ForCausalLM.dequantizeWeights(of: .nar)` remplace les projections 8 bits par des `Linear` bf16 (profils rapides) ; la voie ne peut plus être rechargée, la session est à recréer pour une autre chanson.

## Pipeline et étapes

```swift
public final class YuE2Pipeline {
    init(session:vae: any VAEDecoding, config:, vaeCoreFrames: Int? = nil, releaseWeightsBetweenStages: Bool? = nil)
    func generate(request:abcSampling:semanticSampling:profiling:onEvent:narResume:onNARStep:cancel:) async throws -> SongResult
}
```

Les défauts `nil` viennent du profil mémoire (Mac : 1024 / tout résident ; mobile : 256 / résidence par étape). `PipelineEvent` : `.stage`, `.abcToken`, `.semanticToken`, `.narProgress`, `.vaeProgress`.

Les étapes existent séparément, chacune reprenable depuis ses artefacts :

| Étape | Type | Entrée → sortie |
|---|---|---|
| 1 | `Planner.plan(request:abcSampling:onToken:cancel:)` | requête → `SymbolicPlan` (partition, préfixe) |
| 2 | `SemanticGenerator.generateSemantic(plan:sampling:onToken:cancel:)` | plan → tokens sémantiques + cache KV |
| 3 | `Synthesizer.synthesize(prefix:codec:seed:steps:reuseCache:onBeforeChunk:resume:onStep:cancel:)` | tokens → latents `[T, 64]` |
| 4 | `decodeTiled(using: any VAEDecoding, _:coreFrames:haloFrames:)` | latents → audio |

`onBeforeChunk(needsARPath:)` est le point où la résidence libère la voie AR et charge la voie NAR ; `onStep` livre un `NARCheckpoint` après chaque pas d'ODE et `resume:` en repart (reprise bit-exacte, chansons mono-chunk).

```swift
public struct NARCheckpoint { chunkIndex, step, steps, state: MLXArray
    func save(to: URL) throws; static func load(from: URL) throws -> NARCheckpoint?; static func clear(in: URL) }
```

## Perte du GPU en arrière-plan (iOS)

iOS refuse toute soumission GPU d'une app hors premier plan, et mlx-core lève l'erreur dans le gestionnaire d'achèvement Metal, irrattrapable. Le moteur s'arrête avant :

```swift
YuE2GPUGate.shared.suspend()   // scenePhase == .inactive
YuE2GPUGate.shared.resume()    // .active, et à toute annulation
```

La porte est attendue avant chaque token, chaque couche NAR (si `YuE2ExecutionPolicy.evalPerLayer`, défaut mobile : 28 unités de 0,2-0,4 s au lieu d'un graphe de 10-20 s) et chaque tuile VAE ; `wait(cancel:)` renvoie `false` si l'annulation tombe pendant l'attente. Avec le checkpoint par pas, un résidu coûte un pas.

## Backends VAE

```swift
public protocol VAEDecoding {
    var config: VAEConfig { get }
    var enumeratedFrameCounts: [Int]? { get }        // nil = toute longueur (MLX)
    func decode(_ z: MLXArray) async throws -> MLXArray
}
func loadVAEBackend(_ kind: VAEBackendKind, directory: URL, precision: VAEPrecision) async throws -> any VAEDecoding
func planVAETiles(frames:coreFrames:haloFrames:enumerated:) -> [VAETileWindow]
```

`.mlx` (toujours), `.coreaiGPU` (macOS/iOS 27, parité 54,7 dB grâce aux fenêtres à forme exacte de `planVAETiles`), `.coreaiANE` (impasse avec `coreai-torch` 0.4.2). `loadVAEBackend` lance `.backendUnavailable`, jamais de substitution silencieuse : le repli MLX est à l'appelant, journalisé. Jamais `.cpuOnly` pour ce modèle (convolution transposée fausse sur CPU).

## Mémoire et politique d'exécution

```swift
YuE2MemoryManager.profile                  // .mac | .mobile (#if os(iOS), YUE2_MEMORY_PROFILE)
YuE2MemoryManager.configure(for: .ar | .nar | .vae | .load)
YuE2MemoryManager.mobileLimitsMB()         // cache ≈ 1 Go, seuil = disponible − 1,25 Go
YuE2ExecutionPolicy.narCompute             // .packed | .dequantized
YuE2ExecutionPolicy.evalPerLayer           // défaut mobile
YuE2ExecutionPolicy.compiledDecode         // off, mesuré sans gain
let sampler = FootprintSampler(); sampler.start(); …; sampler.stop(); sampler.peakMB; sampler.mlxActivePeakMB
```

`FootprintSampler` suit `phys_footprint` (le chiffre que juge jetsam) sur un thread dédié, y compris ce que MLX ne voit pas (Core AI, Metal).

## Profils de référence

```swift
public struct YuE2ReferenceProfile { id, bits, kind, quant, quantizeHead, precision, narCompute, compiledDecode,
    vaePrecision, vaeCoreFrames, releaseWeightsBetweenStages, memoryProfile, odeSteps, summary
    static let all: [YuE2ReferenceProfile]; static func named(_:) -> YuE2ReferenceProfile?; func applyGlobalPolicy() }
```

Voir [References.md](References.md) pour les six et leurs mesures.

## Quantification

`YuE2Quantization` : `.none`, `.qint8`, `.int4` (voie AR), `.qint8All`, `.int4All`, `.int4Mixed` (voie AR 4 bits, NAR 8 bits, embeddings 4 bits) ; `quantizeHead` ajoute `lm_head`. `LMWeightLoader.load` quantifie à la volée puis exporte `mlx-prequantized/<preset>[-head]/` ; les chargements suivants lisent l'export. Parités : AR int4 greedy 8/8, NAR 8 bits rel 0,049 sur 32 pas (le NAR 4 bits échoue : 0,15).

## Tests

`Scripts/run-tests.sh` (swift-testing, parallélisation 1 : interblocage connu dans mlx-swift). Deux niveaux : sans poids (fixtures tiny sous `parity/`, toujours vert, 149 tests) et avec les vrais poids (`YUE2_MODELS_DIR`, suites `Real*`, `Quantization`, `GenerationSmoke`). Toute modification numérique passe par `RealLMParityTests` (greedy 8/8 sur les deux phases) et `RealNARParityTests`.
