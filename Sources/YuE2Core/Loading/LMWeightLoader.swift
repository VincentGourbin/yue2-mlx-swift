// LMWeightLoader.swift - loads YuE2-3B/model.safetensors into YuE2ForCausalLM (plan §2.6)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// Loads the real LM checkpoint (bf16, ~628 tensors), refusing any missing/unexpected key.
public enum LMWeightLoader {
    /// `quantization` (O5, T-4.5) quantizes `self_attn.*_proj`/`mlp.*` (and `lm_head` when
    /// `quantizeHead`) in place after the bf16 load. If a matching prequantized export already
    /// exists under `directory/mlx-prequantized/<preset>/`, it is loaded directly instead —
    /// skipping the full bf16 read entirely. Otherwise this call quantizes on the fly and then
    /// writes that export for next time (best-effort: a read-only models directory must not
    /// fail generation over a load-time optimization).
    ///
    /// `residency` (stage-scoped residency, `WeightResidency.swift`): which MoT branches to
    /// actually bring into memory — `.all` by default; `[.ar]` for a process that will only run
    /// the plan/semantic stages before `loadWeights(of: .nar)`. Honoured on the prequantized and
    /// the bf16 (`.none`) paths; the first-time quantize-and-export path always loads everything
    /// (it must, to write a complete export).
    public static func load(
        directory: URL, config: YuE2Config,
        quantization: YuE2Quantization = .none, quantizeHead: Bool = false,
        residency: YuE2WeightResidency = .all
    ) throws -> YuE2ForCausalLM {
        if YuE2PrequantizedCheckpoint.exists(lmDirectory: directory, quantization: quantization, quantizeHead: quantizeHead) {
            // "-all" presets (T-6.2, E2) export without `latent_pos_embed.pe` (~100 MB saved) —
            // build the model to match, so WeightLoader.apply's coverage check doesn't expect a
            // key the file never has. `latentPositions(count:)` recomputes those rows instead.
            let model = YuE2ForCausalLM(config: config, computePE: quantization.isAllPreset)
            YuE2QuantizationFilter.apply(quantization, to: model, quantizeHead: quantizeHead)
            try YuE2PrequantizedCheckpoint.load(
                into: model,
                from: YuE2PrequantizedCheckpoint.url(lmDirectory: directory, quantization: quantization, quantizeHead: quantizeHead),
                quantization: quantization, quantizeHead: quantizeHead,
                retaining: residency == .all ? nil : { residency.retains(key: $0) })
            YuE2MemoryManager.configure(for: .load)
            return model
        }
        if quantization != .none, residency != .all {
            YuE2Debug.log("residency \(residency) ignored: first-time quantization of \(quantization.rawValue) loads every branch")
        }

        // The first-time (bf16 -> quantize -> export) path always loads the full checkpoint,
        // `latent_pos_embed.pe` included — `PrequantizedCheckpoint.export` is what drops it from
        // the *exported* file for "-all" presets, not this in-memory model.
        let model = YuE2ForCausalLM(config: config)
        let weights = try WeightLoader.load(url: directory.appendingPathComponent("model.safetensors")) { key in
            // TimestepEmbedder.mlp is nn.Sequential(Linear, SiLU, Linear), keyed mlp.0/mlp.2.
            // ModuleParameters.unflattened treats an all-digit path segment as a list index, not
            // a module child, so update(parameters:) traps on those keys unmodified (found in
            // T-2.2, same fix as flux-2's WeightLoader for BFL's img_mlp.0/img_mlp.2).
            key
                .replacingOccurrences(of: "time_embedder.mlp.0.", with: "time_embedder.fc1.")
                .replacingOccurrences(of: "time_embedder.mlp.2.", with: "time_embedder.fc2.")
        }
        // Piège n°8: WeightLoader.apply evals tensor by tensor, never the whole tree at once.
        try WeightLoader.apply(
            weights, to: model, component: "YuE2ForCausalLM",
            retaining: (quantization == .none && residency != .all) ? { residency.retains(key: $0) } : nil)

        if quantization != .none {
            YuE2QuantizationFilter.apply(quantization, to: model, quantizeHead: quantizeHead)
            do {
                try YuE2PrequantizedCheckpoint.export(
                    model: model, lmDirectory: directory, quantization: quantization, quantizeHead: quantizeHead)
            } catch {
                YuE2Debug.log("prequantized export skipped: \(error)")
            }
        }

        YuE2MemoryManager.configure(for: .load)
        return model
    }

    /// Brings one branch into an already-built `model` (stage-scoped residency): re-reads the
    /// same file `load` used (prequantized export, or the bf16 checkpoint for `.none`) and
    /// applies/evaluates only `path`'s tensors. A quantized preset without its prequantized
    /// export cannot be partially loaded — `load` writes that export on its first full run.
    public static func load(
        path: YuE2WeightPath, into model: YuE2ForCausalLM, directory: URL,
        quantization: YuE2Quantization = .none, quantizeHead: Bool = false
    ) throws {
        let retaining: (String) -> Bool = { YuE2WeightPath.of(key: $0) == path }
        if quantization == .none {
            let weights = try WeightLoader.load(url: directory.appendingPathComponent("model.safetensors")) { key in
                key
                    .replacingOccurrences(of: "time_embedder.mlp.0.", with: "time_embedder.fc1.")
                    .replacingOccurrences(of: "time_embedder.mlp.2.", with: "time_embedder.fc2.")
            }
            try WeightLoader.apply(weights, to: model, component: "YuE2ForCausalLM [\(path)]", retaining: retaining)
            return
        }
        guard YuE2PrequantizedCheckpoint.exists(lmDirectory: directory, quantization: quantization, quantizeHead: quantizeHead) else {
            throw YuE2Error.missingFile(
                "no prequantized \(quantization.rawValue) export under \(directory.path)/mlx-prequantized — a full load must run once first")
        }
        try YuE2PrequantizedCheckpoint.load(
            into: model,
            from: YuE2PrequantizedCheckpoint.url(lmDirectory: directory, quantization: quantization, quantizeHead: quantizeHead),
            quantization: quantization, quantizeHead: quantizeHead, retaining: retaining)
    }
}
