// WeightResidency.swift - stage-scoped weight residency: drop the MoT branch a stage no longer reads
// Copyright 2026 Vincent Gourbin
//
// Measured on the int4-mixed pack (2026-09-21): NAR path 1 428 MB, AR path 756 MB, `lm_head`
// 722 MB (bf16 unless `--quant-head`), `embed_tokens` 203 MB. Once the NAR's AR-prefix cache
// exists, nothing in the flow-matching solve reads the AR path, `embed_tokens` or `lm_head`
// (`CachedNAR.velocity` touches only `nar_*`, `vae2llm`/`llm2vae`, `time_embedder`, the shared
// norms); once the latents exist, nothing in the VAE stage reads the LM at all. On a Mac this is
// invisible (96 GB); on an 8 GB iPhone it is the difference between the AR-stage, NAR-stage and
// VAE-stage working sets *adding up* and only the largest one counting.

import MLX
import MLXNN

/// The Mixture-of-Transformers branch a checkpoint parameter belongs to.
public enum YuE2WeightPath: Sendable {
    /// `self_attn`/`mlp` on every layer, `embed_tokens`, `lm_head`.
    case ar
    /// `nar_self_attn`/`nar_mlp` on every layer, `vae2llm`/`llm2vae`, `time_embedder`,
    /// `latent_pos_embed`.
    case nar

    /// Which branch reads the flattened parameter `key` exclusively; `nil` for the shared
    /// pieces (`model.norm`, the per-layer norms — a few MB in total, never worth dropping).
    public static func of(key: String) -> YuE2WeightPath? {
        let parts = key.split(separator: ".").map(String.init)
        if parts.contains(where: { $0.hasPrefix("nar_") }) { return .nar }
        if let first = parts.first, ["vae2llm", "llm2vae", "time_embedder", "latent_pos_embed"].contains(first) {
            return .nar
        }
        if parts.contains("self_attn") || parts.contains("mlp") || parts.contains("embed_tokens")
            || parts.contains("lm_head")
        {
            return .ar
        }
        return nil
    }
}

/// Which branches of the LM are resident (loaded and evaluated) — `LMWeightLoader.load(residency:)`
/// and `ModelSession.load(residency:)`. The shared norms are always resident.
public struct YuE2WeightResidency: OptionSet, Sendable, Equatable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let ar = YuE2WeightResidency(rawValue: 1)
    public static let nar = YuE2WeightResidency(rawValue: 2)
    public static let all: YuE2WeightResidency = [.ar, .nar]

    /// True when a checkpoint parameter `key` must be loaded for this residency.
    public func retains(key: String) -> Bool {
        switch YuE2WeightPath.of(key: key) {
        case .ar: return contains(.ar)
        case .nar: return contains(.nar)
        case nil: return true
        }
    }
}

extension YuE2ForCausalLM {
    /// Replaces every parameter `predicate` selects with an *unevaluated* zero array of the same
    /// shape and dtype — the module tree keeps its structure (so `parameters()`, `update`, and a
    /// later reload keep working), but the GPU buffers behind the old arrays are released as soon
    /// as nothing else references them (`Memory.clearCache()` afterwards returns them to the
    /// system on a profile with a small cache limit). The branch is then **unusable** until
    /// reloaded: calling it would silently compute on zeros. Returns the bytes released.
    @discardableResult
    public func releaseWeights(where predicate: (String) -> Bool) -> Int {
        var replacements = [String: MLXArray]()
        var bytes = 0
        for (key, value) in parameters().flattened() where predicate(key) {
            bytes += value.nbytes
            replacements[key] = MLXArray.zeros(value.shape, dtype: value.dtype)
        }
        guard !replacements.isEmpty else { return 0 }
        update(parameters: ModuleParameters.unflattened(replacements))
        return bytes
    }

    /// Drops the parameters only `path` reads (`YuE2WeightPath.of(key:)`).
    @discardableResult
    public func releaseWeights(of path: YuE2WeightPath) -> Int {
        releaseWeights { YuE2WeightPath.of(key: $0) == path }
    }

    /// Drops every parameter of the LM (both branches and the shared norms).
    @discardableResult
    public func releaseAllWeights() -> Int {
        releaseWeights { _ in true }
    }
}
