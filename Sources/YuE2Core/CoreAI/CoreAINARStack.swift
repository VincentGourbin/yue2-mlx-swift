// CoreAINARStack.swift - NAR velocity field via Core AI (GPU/Neural Engine), T-6.4 (E4)
// Copyright 2026 Vincent Gourbin

#if canImport(CoreAI)
import CoreAI
import Foundation
import MLX

/// Runs the exported `Scripts/coreai/export_nar_stack.py` asset (`.aimodel`/`.aimodelc`) through
/// Core AI instead of MLX, for one chunk's whole ODE solve.
///
/// The AR prefix's K/V is converted to two `NDArray`s **once per chunk**, at `init` (never per
/// `velocity` call) — `plan/13-ios-backend.md` §13.7 flags exactly this ("traversée `NDArray`
/// coûteuse") as measured-negligible only because of this once-per-chunk convention, not once
/// per ODE step (64 steps would multiply the conversion cost 64x for no reason: the AR cache is
/// invariant across the whole solve). The (L, L_ar) bucket is also chosen once here, from this
/// chunk's real `narLength`/`arLength` — every `velocity` call after that only pads/converts the
/// per-step `x_t`/`raw_t`, which do change every evaluation.
///
/// ## Compute-unit safety
/// Same open limitation as `CoreAIVAEDecoder` (T-6.3): `SpecializationOptions` can only *prefer*
/// a compute unit, never strictly exclude CPU fallback, and there is no API to introspect which
/// unit actually ran an operation. The backstop here is the same as T-6.3's: this fiche's own
/// numeric parity gate (`yue2 parity nar --backend coreai-gpu|coreai-ane`, rel < 5e-2 vs the same
/// real fixture the MLX path is checked against) must be green on the target device before this
/// backend ships in a preset.
@available(macOS 27.0, iOS 27.0, *)
public final class CoreAINARStack: NARVelocityBackend {
    public enum ComputeUnit: String, Sendable, CustomStringConvertible {
        case gpu
        case neuralEngine

        var kind: ComputeUnitKind { self == .gpu ? .gpu : .neuralEngine }
        public var description: String { rawValue }
    }

    /// `Scripts/coreai/export_nar_stack.py`'s `L_BUCKETS`/`L_AR_BUCKETS`/`num_layers` — kept in
    /// sync manually (no shared source of truth across the Python/Swift boundary, same as
    /// `CoreAIVAEDecoder.enumeratedFrames` mirroring `export_vae.py`'s `DECODER_FRAMES`).
    static let lBuckets = [32, 256, 512, 1024, 1536]
    static let lArBuckets = [512, 1024]
    static let numLayers = 28
    static let numKVHeads = 8
    static let headDim = 128

    public let unit: ComputeUnit
    private let function: InferenceFunction
    private let lBucket: Int
    private let lArBucket: Int
    private let arLength: Int
    private let narLength: Int
    private let kArND: NDArray
    private let vArND: NDArray
    private let maskND: NDArray

    /// `assetURL`: `.aimodel` or precompiled `.aimodelc`. `cache`/`arLength`/`narLength`: this
    /// chunk's AR prefix and NAR geometry (`CachedNAR.arPrefixCache`/`chunkARLength`/
    /// `chunkNARLength`) — the whole point of building this once per chunk rather than per step.
    public init(
        assetURL: URL, unit: ComputeUnit, cache: KVCache, arLength: Int, narLength: Int
    ) async throws {
        self.unit = unit
        self.arLength = arLength
        self.narLength = narLength

        let frames = narLength - 2
        guard let lBucket = Self.lBuckets.first(where: { $0 >= frames }) else {
            throw YuE2Error.invalidRequest(
                "CoreAI NAR stack only supports up to \(Self.lBuckets.last!) frames per chunk (got \(frames))")
        }
        guard let lArBucket = Self.lArBuckets.first(where: { $0 >= arLength }) else {
            throw YuE2Error.invalidRequest(
                "CoreAI NAR stack only supports up to \(Self.lArBuckets.last!) AR prefix tokens (got \(arLength))")
        }
        self.lBucket = lBucket
        self.lArBucket = lArBucket

        let options = SpecializationOptions(preferredComputeUnitKind: unit.kind)
        let model: AIModel
        do {
            model = try await AIModel(contentsOf: assetURL, options: options)
        } catch {
            throw YuE2Error.backendUnavailable(
                "CoreAI NAR stack: failed to load/specialize \(assetURL.lastPathComponent) for \(unit): \(error)")
        }
        let entrypoint = "L\(lBucket)_Lar\(lArBucket)"
        guard let function = try model.loadFunction(named: entrypoint) else {
            throw YuE2Error.backendUnavailable(
                "CoreAI NAR asset \(assetURL.lastPathComponent) is missing entrypoint \(entrypoint)")
        }
        self.function = function

        // Stack every layer's AR K/V once: [1, kvHeads, arLength, headDim] -> [arLength, kvHeads,
        // headDim] per layer -> [numLayers, arLength, kvHeads, headDim], then zero-pad to
        // lArBucket (piège n°1-adjacent: padding here never touches RoPE, which the asset itself
        // re-anchors from the mask's real-AR-length count -- see `export_nar_stack.py`'s
        // `forward`).
        var kLayers: [MLXArray] = []
        var vLayers: [MLXArray] = []
        for i in 0..<Self.numLayers {
            guard let (k, v) = cache.keyValue(layer: i) else {
                throw YuE2Error.backendUnavailable("CoreAI NAR backend: AR cache is missing layer \(i)")
            }
            kLayers.append(k[0].transposed(1, 0, 2))
            vLayers.append(v[0].transposed(1, 0, 2))
        }
        var kStacked = MLX.stacked(kLayers, axis: 0)
        var vStacked = MLX.stacked(vLayers, axis: 0)
        if lArBucket > arLength {
            let pad = MLXArray.zeros([Self.numLayers, lArBucket - arLength, Self.numKVHeads, Self.headDim])
                .asType(kStacked.dtype)
            kStacked = concatenated([kStacked, pad], axis: 1)
            vStacked = concatenated([vStacked, pad], axis: 1)
        }
        self.kArND = NDArrayBridge.ndArray(from: kStacked)
        self.vArND = NDArrayBridge.ndArray(from: vStacked)

        // Full key-padding mask: [real AR][AR bucket pad][real NAR (START/frames/END)][NAR
        // bucket pad] -- matches `k_cat`'s concatenation order in `export_nar_stack.py`'s
        // `forward`, and must cover the NAR side too (bidirectional attention: an unmasked
        // padded NAR frame is not garbage-free, `vae2llm(0) + time_embed + pos_emb` is nonzero).
        var mask = [Bool](repeating: true, count: arLength)
        mask += [Bool](repeating: false, count: lArBucket - arLength)
        mask += [Bool](repeating: true, count: narLength)
        mask += [Bool](repeating: false, count: lBucket - frames)
        self.maskND = NDArrayBridge.ndArray(scalars: mask, shape: [mask.count])
    }

    public func velocity(state: MLXArray, rawT: Double) async throws -> MLXArray {
        let L = state.dim(0)
        guard L + 2 == narLength else {
            throw YuE2Error.invalidRequest(
                "CoreAINARStack: state length \(L) does not match this chunk's narLength \(narLength - 2)")
        }
        var xt = state
        if lBucket > L {
            let pad = MLXArray.zeros([lBucket - L, state.dim(1)]).asType(state.dtype)
            xt = concatenated([xt, pad], axis: 0)
        }

        // Bind every NDArray to a local `let` before `.view(as:)` (T-6.3's working pattern in
        // `CoreAIVAEDecoder.decode`): a lifetime-dependent `View` borrows its source for as long
        // as the view is alive, and that borrow must be traceable to a named local binding that
        // spans this whole call, not to an unnamed temporary or a bare stored-property access —
        // either of the latter only got the compiler a borrow scoped to this one statement, too
        // short to cover `function.run(inputs:)` below ("lifetime-dependent value escapes its
        // scope").
        let xtND = NDArrayBridge.ndArray(from: xt)
        let rawTND = NDArrayBridge.ndArray(scalars: [Float(rawT)], shape: [1])
        let kArND = self.kArND
        let vArND = self.vArND
        let maskND = self.maskND

        var inputs = InferenceFunction.Inputs()
        inputs.insert(xtND.view(as: Float16.self), for: "x_t")
        inputs.insert(rawTND.view(as: Float32.self), for: "raw_t")
        inputs.insert(kArND.view(as: Float16.self), for: "k_ar")
        inputs.insert(vArND.view(as: Float16.self), for: "v_ar")
        inputs.insert(maskND.view(as: Bool.self), for: "key_padding_mask")

        var outputs: InferenceFunction.Outputs
        do {
            outputs = try await function.run(inputs: inputs)
        } catch {
            throw YuE2Error.backendUnavailable("CoreAI NAR stack run failed (\(unit)): \(error)")
        }
        guard let value = outputs.remove("v") else {
            throw YuE2Error.backendUnavailable("CoreAI NAR stack: output \"v\" missing from run() result")
        }
        guard let outNDArray = value.ndArray else {
            throw YuE2Error.backendUnavailable("CoreAI NAR stack: output \"v\" is not an NDArray")
        }
        let out = try NDArrayBridge.mlxArray(from: outNDArray) // [lBucket, latentDim] fp32
        let cropped = out[0..<L, 0...]
        eval(cropped)
        return cropped
    }
}
#endif
