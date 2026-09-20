// RestrictedHead.swift - lm_head restricted to a phase's legal rows (plan §2.3, O1)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// Computes `lm_head` only over the rows a phase can legally emit (§2.3's "exact
/// optimization"). Every row outside `bounds.allowedRange ∪ {bounds.endToken}` would score
/// `-inf` after `LogitsProcessor`'s mask anyway, so gathering just those rows once at `init`
/// and only ever matmul-ing against them produces the exact same finite values as computing
/// the full row and masking afterwards — it only skips the wasted compute (and the
/// `vocabSize`-wide allocation) on the rows that were always going to be discarded.
public final class RestrictedHead {
    /// `globalIDs[i]` is the vocabulary id local index `i` maps to: `bounds.allowedRange` in
    /// order, then `bounds.endToken` last (`globalIDs.count == bounds.allowedRange.count + 1`).
    public let globalIDs: [Int]
    private let localByGlobal: [Int: Int]
    private let weight: MLXArray

    /// When `lmHead` is a `QuantizedLinear` (O5's `--quant-head`, T-4.5), gathers the packed
    /// rows this phase needs and dequantizes just those — never the reverse (§9 n°10: a
    /// `QuantizedLinear`'s `weight` is packed `uint32`, so `MLX.take` must run on the packed
    /// array and its `scales`/`biases` *before* `dequantized`, not on some `asType` cast of it).
    public init(lmHead: Linear, bounds: LogitsBounds) {
        var ids = Array(bounds.allowedRange)
        ids.append(bounds.endToken)
        globalIDs = ids
        localByGlobal = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        let idsArray = MLXArray(ids.map { Int32($0) })
        if let quantized = lmHead as? QuantizedLinear {
            let rows = MLX.take(quantized.weight, idsArray, axis: 0)
            let scales = MLX.take(quantized.scales, idsArray, axis: 0)
            let biases = quantized.biases.map { MLX.take($0, idsArray, axis: 0) }
            weight = MLX.dequantized(
                rows, scales: scales, biases: biases,
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
        } else {
            weight = MLX.take(lmHead.weight, idsArray, axis: 0)
        }
        eval(weight)
    }

    /// `hidden @ weight.T`: `[..., hiddenSize] -> [..., n]` (`n = globalIDs.count`).
    public func logits(hidden: MLXArray) -> MLXArray {
        matmul(hidden, weight.T)
    }

    /// Local index (0-based, within `globalIDs`) → global vocabulary id.
    public func global(_ local: Int) -> Int {
        globalIDs[local]
    }

    /// Global vocabulary id → local index, or `nil` if this head does not carry that row.
    public func local(_ global: Int) -> Int? {
        localByGlobal[global]
    }
}
