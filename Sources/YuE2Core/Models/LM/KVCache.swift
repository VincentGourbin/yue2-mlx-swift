// KVCache.swift - per-layer AR key/value cache (plan §2.2, pitfall #2)
// Copyright 2026 Vincent Gourbin

import MLX

/// Per-layer K/V cache for the AR path. K/V are cached **after** k_norm and RoPE
/// (pitfall #2): `append` receives already-normalized, already-rotated tensors.
///
/// `offset` (used by every layer to compute this step's RoPE positions) is **not** advanced by
/// `append` itself — it has no fixed layer count to detect "the last layer of this step" from.
/// The caller (`Backbone.forwardAR`) calls `advance(by:)` once, after every layer of a step has
/// appended, so all layers in that step see the same pre-step offset.
public final class KVCache {
    private var keys: [Int: MLXArray] = [:]
    private var values: [Int: MLXArray] = [:]
    private(set) var offset = 0

    public init() {}

    /// Concatenates `k`/`v` (`[B, kvHeads, T, headDim]`) onto layer `layer`'s cache along the
    /// sequence axis and returns the full cached K/V for that layer.
    func append(layer: Int, k: MLXArray, v: MLXArray) -> (MLXArray, MLXArray) {
        let newK: MLXArray
        let newV: MLXArray
        if let existingK = keys[layer], let existingV = values[layer] {
            newK = concatenated([existingK, k], axis: 2)
            newV = concatenated([existingV, v], axis: 2)
        } else {
            newK = k
            newV = v
        }
        keys[layer] = newK
        values[layer] = newV
        return (newK, newV)
    }

    /// Advances the offset used for the *next* step's RoPE positions, by this step's token count.
    func advance(by count: Int) {
        offset += count
    }

    /// Materializes every cached tensor (piège n°8: eval per step, not the whole graph at once).
    func evalAll() {
        for k in keys.values { eval(k) }
        for v in values.values { eval(v) }
    }

    /// Public: `yue2 bench-coreai-nar` (T-6.4, E4) needs one independent cache per backend under
    /// test, from outside `YuE2Core` — every existing in-module caller still uses the cache
    /// `CachedNAR`/`TokenGenerator.prefill` hand it directly, unaffected by this being public too.
    public func clone() -> KVCache {
        let copy = KVCache()
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        return copy
    }

    /// Read-only per-layer snapshot of this cache's K/V, for callers that attend over it without
    /// appending to it (NAR: every ODE step attends over `[this AR snapshot ; fresh NAR K/V]`
    /// without ever mutating the AR cache itself — T-3.2).
    func keyValue(layer: Int) -> (MLXArray, MLXArray)? {
        guard let k = keys[layer], let v = values[layer] else { return nil }
        return (k, v)
    }

    /// Drops cached positions beyond `length` (used when a speculative/NAR excursion must be
    /// rolled back to the last confirmed AR position).
    func truncate(to length: Int) {
        for layer in keys.keys {
            guard let k = keys[layer], let v = values[layer], k.dim(2) > length else { continue }
            keys[layer] = k[0..., 0..., 0..<length, 0...]
            values[layer] = v[0..., 0..., 0..<length, 0...]
        }
        offset = min(offset, length)
    }
}
