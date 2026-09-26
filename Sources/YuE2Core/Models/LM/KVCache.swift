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
    /// Growth step of the per-layer buffers: every append that overflows reallocates to the next
    /// multiple of this and copies once; every other append is a `slice_update` into spare
    /// capacity (donated in place by MLX when nothing else holds the buffer), never a full copy
    /// of the cache. The old `concatenated([existing, k])` copied all 28 × 2 × [8, offset, 128]
    /// tensors on every decoded token — ≈ 260 MB and 56 allocations per token at a 2 300-token
    /// prefix, the bulk of the AR phases' CPU/dispatch cost (2026-09-26).
    static let growthStep = 512

    private var keys: [Int: MLXArray] = [:]    // [B, kvHeads, capacity, headDim], valid up to `offset` (+ this step's tokens)
    private var values: [Int: MLXArray] = [:]
    private(set) var offset = 0

    public init() {}

    /// Writes `k`/`v` (`[B, kvHeads, T, headDim]`) at positions `offset ..< offset + T` of layer
    /// `layer`'s buffers and returns the full cached K/V for that layer (`[..., 0 ..< offset + T, ...]`).
    func append(layer: Int, k: MLXArray, v: MLXArray) -> (MLXArray, MLXArray) {
        let count = k.dim(2)
        let start = offset
        let end = start + count
        let (bufferK, bufferV) = buffers(layer: layer, k: k, v: v, start: start, end: end)
        bufferK[0..., 0..., start..<end, 0...] = k
        bufferV[0..., 0..., start..<end, 0...] = v
        keys[layer] = bufferK
        values[layer] = bufferV
        return (bufferK[0..., 0..., 0..<end, 0...], bufferV[0..., 0..., 0..<end, 0...])
    }

    /// The layer's buffers with capacity for `end` positions: the existing ones when they fit,
    /// otherwise grown (once per `growthStep` positions) by appending zeros — and **evaluated**
    /// before any subscript-set touches them (T-4.3: mutating an unevaluated `concatenated`
    /// result can alias its source under memory pressure).
    private func buffers(layer: Int, k: MLXArray, v: MLXArray, start: Int, end: Int) -> (MLXArray, MLXArray) {
        if let existingK = keys[layer], let existingV = values[layer], existingK.dim(2) >= end {
            return (existingK, existingV)
        }
        let capacity = ((end + Self.growthStep - 1) / Self.growthStep) * Self.growthStep
        func grown(_ existing: MLXArray?, like sample: MLXArray) -> MLXArray {
            let kept = existing.map { $0[0..., 0..., 0..<start, 0...] }
            let padCount = capacity - (kept?.dim(2) ?? 0)
            let pad = MLXArray.zeros([sample.dim(0), sample.dim(1), padCount, sample.dim(3)], dtype: sample.dtype)
            let buffer = kept.map { concatenated([$0, pad], axis: 2) } ?? pad
            eval(buffer)
            return buffer
        }
        return (grown(keys[layer], like: k), grown(values[layer], like: v))
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
    /// The buffers are shared until either side appends (MLX copies on write instead of donating
    /// a buffer that is still referenced).
    public func clone() -> KVCache {
        let copy = KVCache()
        copy.keys = keys
        copy.values = values
        copy.offset = offset
        return copy
    }

    /// Read-only per-layer snapshot of this cache's K/V (`[..., 0 ..< offset, ...]`), for callers
    /// that attend over it without appending to it (NAR: every ODE step attends over `[this AR
    /// snapshot ; fresh NAR K/V]` without ever mutating the AR cache itself — T-3.2).
    func keyValue(layer: Int) -> (MLXArray, MLXArray)? {
        guard let k = keys[layer], let v = values[layer] else { return nil }
        return (k[0..., 0..., 0..<offset, 0...], v[0..., 0..., 0..<offset, 0...])
    }

    /// Drops cached positions beyond `length` (used when a speculative/NAR excursion must be
    /// rolled back to the last confirmed AR position). The buffers keep their capacity; only the
    /// valid length moves.
    func truncate(to length: Int) {
        offset = min(offset, length)
    }
}
