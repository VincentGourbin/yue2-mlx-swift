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
    public static func load(
        directory: URL, config: YuE2Config,
        quantization: YuE2Quantization = .none, quantizeHead: Bool = false
    ) throws -> YuE2ForCausalLM {
        let model = YuE2ForCausalLM(config: config)

        if YuE2PrequantizedCheckpoint.exists(lmDirectory: directory, quantization: quantization, quantizeHead: quantizeHead) {
            YuE2QuantizationFilter.apply(quantization, to: model, quantizeHead: quantizeHead)
            try YuE2PrequantizedCheckpoint.load(
                into: model,
                from: YuE2PrequantizedCheckpoint.url(lmDirectory: directory, quantization: quantization, quantizeHead: quantizeHead),
                quantization: quantization, quantizeHead: quantizeHead)
            YuE2MemoryManager.configure(for: .load)
            return model
        }

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
        try WeightLoader.apply(weights, to: model, component: "YuE2ForCausalLM")

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
}
