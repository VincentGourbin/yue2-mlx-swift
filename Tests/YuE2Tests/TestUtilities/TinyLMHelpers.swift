// TinyLMHelpers.swift - shared tier-1 helpers for parity/tiny_lm.safetensors (T-2.x)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
@testable import YuE2Core

let tinyLMFixture = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
    .appendingPathComponent("parity/tiny_lm.safetensors")

/// Matches `tests/test_nar.py`'s tiny config (seed 173, T-1.3).
func tinyLMConfig() -> YuE2Config {
    YuE2Config(
        hiddenSize: 16, numHiddenLayers: 2, numAttentionHeads: 4, numKeyValueHeads: 2,
        headDim: 4, intermediateSize: 32, vocabSize: 32, maxPositionEmbeddings: 128,
        latentDim: 64, maxLatentFrames: 128
    )
}

private let tinyLMWeightPrefixes = [
    "model.", "lm_head.", "vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed.",
]

/// `time_embedder.mlp.{0,2}` -> `time_embedder.fc{1,2}` (see `TimeEmbedder`, T-2.2).
func remapLMWeightKey(_ key: String) -> String {
    key
        .replacingOccurrences(of: "time_embedder.mlp.0.", with: "time_embedder.fc1.")
        .replacingOccurrences(of: "time_embedder.mlp.2.", with: "time_embedder.fc2.")
}

/// Loads the tiny checkpoint's real weight tensors (dropping the fixture's auxiliary taps),
/// optionally dropping one more key to exercise `WeightLoader.apply`'s coverage check.
func loadTinyLMWeights(dropping keyToRemove: String? = nil) throws -> [String: MLXArray] {
    var weights = try WeightLoader.load(url: tinyLMFixture) { key in
        tinyLMWeightPrefixes.contains(where: key.hasPrefix) ? remapLMWeightKey(key) : nil
    }
    if let keyToRemove {
        weights.removeValue(forKey: keyToRemove)
    }
    return weights
}

/// Writes `weights` as a standalone `model.safetensors` under a fresh temp directory, so
/// `LMWeightLoader.load(directory:config:)` can be exercised end-to-end.
func makeTinyLMDirectory(dropping keyToRemove: String? = nil) throws -> URL {
    let weights = try loadTinyLMWeights(dropping: keyToRemove)
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
    return dir
}

/// Builds a fresh tiny model and applies the (filtered, remapped) checkpoint weights to it;
/// throws on any missing/unexpected key (full coverage).
func loadedTinyModel() throws -> YuE2ForCausalLM {
    let model = YuE2ForCausalLM(config: tinyLMConfig())
    try WeightLoader.apply(loadTinyLMWeights(), to: model, component: "tiny_lm")
    return model
}
