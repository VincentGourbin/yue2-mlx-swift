// CoreAIVAEDecoder.swift - VAE decoder inference via Core AI (GPU/Neural Engine), T-6.3 (E3)
// Copyright 2026 Vincent Gourbin

#if canImport(CoreAI)
import CoreAI
import Foundation
import MLX

/// Runs the exported `Scripts/coreai/export_vae.py` decoder asset (`.aimodel`/`.aimodelc`)
/// through Core AI instead of MLX.
///
/// ## Compute-unit safety — read before changing `ComputeUnit`
/// `plan/13-ios-backend.md` and `CLAUDE.md` both require: never let this asset specialize on
/// CPU (the released framework has an open conv-transpose defect for kernel >= 8, and this
/// decoder's kernels are 12/10/8). This SDK's real API, verified against
/// `MacOSX27.0.sdk/.../CoreAIDelegates.swiftinterface`, offers exactly three ways to build
/// `SpecializationOptions`: `.default` (any compute unit), `.cpuOnly` (forces CPU), and
/// `init(preferredComputeUnitKind:)`. The last one only *prefers* a unit — per Apple's own
/// documentation, "fallback to other kinds in `allowedComputeUnitKinds` may still occur for
/// operations or operation patterns that are incompatible with the preferred kind" — and there
/// is **no public initializer or setter to restrict `allowedComputeUnitKinds` to a subset that
/// excludes CPU**, and **no public API to introspect, after specialization, which compute unit
/// actually ran a given operation**. A silent CPU fallback for an incompatible operation pattern
/// therefore cannot be prevented or positively detected by this decoder — that is a real,
/// documented limitation of the framework as shipped in this SDK (Xcode 27.0), not a gap in
/// this port. `tasks/JOURNAL.md` T-6.3 records this finding for Vincent.
///
/// The practical backstop is this ticket's own mandatory numeric parity gate (`snr > 40 dB` vs
/// MLX fp32, measured on real hardware before a preset ships): the known CPU conv-transpose
/// defect corrupts output rather than merely slowing it down, so a silent fallback would fail
/// that gate loudly instead of passing unnoticed. `CoreAIVAETests`/`yue2 parity vae --backend
/// coreai-*` must be green on the target device, every time this asset changes.
@available(macOS 27.0, iOS 27.0, *)
public final class CoreAIVAEDecoder: VAEDecoding {
    public enum ComputeUnit: String, Sendable, CustomStringConvertible {
        case gpu
        case neuralEngine

        var kind: ComputeUnitKind { self == .gpu ? .gpu : .neuralEngine }
        public var description: String { rawValue }
    }

    /// Latent frame counts the exported asset was specialized for (`Scripts/coreai/export_vae.py`
    /// `DECODER_FRAMES`) — matches `decodeTiled`'s default tile (1024) + halo (16) on each side,
    /// so a full tiled decode never needs a shape above the largest entrypoint.
    static let enumeratedFrames = [288, 544, 1056]

    public let config: VAEConfig
    public let unit: ComputeUnit
    private let functions: [Int: InferenceFunction]

    /// `decodeTiled(using:)` sizes its windows to these, so the zero-pad branch of `decode`
    /// below only runs for a signal shorter than 288 frames (≈ 11.5 s of audio).
    public var enumeratedFrameCounts: [Int]? { Self.enumeratedFrames }

    /// `assetURL` is `.aimodel` (specializes + caches on first load) or a pre-compiled
    /// `.aimodelc` (`Scripts/coreai/compile.sh`) — `AIModel.init(contentsOf:options:)` accepts
    /// both per its documentation.
    public init(assetURL: URL, config: VAEConfig, unit: ComputeUnit) async throws {
        self.config = config
        self.unit = unit
        let options = SpecializationOptions(preferredComputeUnitKind: unit.kind)
        let model: AIModel
        do {
            model = try await AIModel(contentsOf: assetURL, options: options)
        } catch {
            throw YuE2Error.backendUnavailable(
                "CoreAI VAE decoder: failed to load/specialize \(assetURL.lastPathComponent) for \(unit): \(error)")
        }
        var loaded: [Int: InferenceFunction] = [:]
        for frames in Self.enumeratedFrames {
            guard let function = try model.loadFunction(named: "len\(frames)") else {
                throw YuE2Error.backendUnavailable(
                    "CoreAI VAE asset \(assetURL.lastPathComponent) is missing entrypoint len\(frames)")
            }
            loaded[frames] = function
        }
        self.functions = loaded
    }

    /// `z`: `[1, T, latentDim]` NLC (same convention as `YuE2VAE.decode`), `T` at most the
    /// largest enumerated shape (1056) — use the package-level `decodeTiled(using:_:...)` for
    /// longer latents; this type only ever runs one enumerated shape per call.
    public func decode(_ z: MLXArray) async throws -> MLXArray {
        precondition(
            z.ndim == 3 && z.dim(0) == 1 && z.dim(1) >= 1 && z.dim(2) == config.decoderConfig.latentDim,
            "expected nonempty [1, T, \(config.decoderConfig.latentDim)] latents, got \(z.shape)")
        let frames = z.dim(1)
        guard let targetFrames = Self.enumeratedFrames.first(where: { $0 >= frames }) else {
            throw YuE2Error.invalidRequest(
                "CoreAI VAE decoder only supports up to \(Self.enumeratedFrames.last!) latent frames per call (got \(frames)); use decodeTiled(using:_:) for longer latents")
        }
        guard let function = functions[targetFrames] else {
            throw YuE2Error.backendUnavailable("CoreAI VAE decoder: no function loaded for len\(targetFrames)")
        }

        // Zero-pad on the right to the enumerated shape, then NLC -> NCL to match the export's
        // `z: [1, 64, T]` (pitfall n°11-adjacent: pad AFTER the NLC layout the rest of this port
        // uses, transpose only for the Core AI call itself).
        let padded: MLXArray
        if frames == targetFrames {
            padded = z
        } else {
            let pad = MLXArray.zeros([1, targetFrames - frames, config.decoderConfig.latentDim])
            padded = concatenated([z, pad], axis: 1)
        }
        let zNCL = padded.transposed(0, 2, 1)

        var inputs = InferenceFunction.Inputs()
        let inputArray = NDArrayBridge.ndArray(from: zNCL)
        inputs.insert(inputArray.view(as: Float16.self), for: "z")

        var outputs: InferenceFunction.Outputs
        do {
            outputs = try await function.run(inputs: inputs)
        } catch {
            throw YuE2Error.backendUnavailable("CoreAI VAE decoder run failed (\(unit)): \(error)")
        }
        guard let value = outputs.remove("out") else {
            throw YuE2Error.backendUnavailable("CoreAI VAE decoder: output \"out\" missing from run() result")
        }
        guard let outNDArray = value.ndArray else {
            throw YuE2Error.backendUnavailable("CoreAI VAE decoder: output \"out\" is not an NDArray")
        }
        let outNCL = try NDArrayBridge.mlxArray(from: outNDArray) // [1, outChannels, targetFrames * ratio]
        let outNLC = outNCL.transposed(0, 2, 1)

        if frames == targetFrames {
            return outNLC
        }
        let wantedSamples = naturalOutputLength(frames: frames)
        let cropped = outNLC[0..., 0..<wantedSamples, 0...]
        eval(cropped)
        return cropped
    }
}
#endif
