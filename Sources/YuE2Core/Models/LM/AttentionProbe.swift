// AttentionProbe.swift - where AR heads look in the prompt while they write each semantic frame
// Copyright 2026 Vincent Gourbin

import MLX
import MLXFast
import MLXNN

extension YuE2ForCausalLM {
    /// Teacher-forced pass over `tokens` (a semantic-phase prefix of `prefixLength` tokens
    /// followed by the song's codec tokens) that returns, for each requested `(layer, head)`, the
    /// attention mass each query predicting a codec frame puts on each prompt segment:
    /// `[heads.count, frames, segmentCount]`, frame `f` being the query at position
    /// `prefixLength - 1 + f` (the one that samples frame `f`). `segments[i]` is the segment of
    /// prompt token `i` (`-1`: none). `heads == nil` probes every head of every layer, in
    /// layer-major order. Only the layers up to the deepest requested one run, `chunk` tokens at
    /// a time, so a long song never materializes a full attention matrix. `reusing`: the
    /// semantic phase's own cache for these tokens (no CFG); its keys and values are read, not
    /// rebuilt, so the pass costs no extra cache memory. Waits on `YuE2GPUGate` before every
    /// layer (one unit of GPU work each) and throws `.cancelled` when `cancel` fires.
    public func prefixAttention(
        tokens: [Int32], prefixLength: Int, segments: [Int32], segmentCount: Int,
        heads: [(layer: Int, head: Int)]? = nil, chunk: Int = 1024, reusing generated: KVCache? = nil,
        cancel: (() -> Bool)? = nil
    ) throws -> MLXArray {
        precondition(segments.count == prefixLength && tokens.count > prefixLength && prefixLength > 0)
        let backbone = model
        let headCount = backbone.layers[0].selfAttn.numHeads
        let probed = heads ?? (0..<backbone.layers.count).flatMap { l in (0..<headCount).map { (l, $0) } }
        let lastLayer = probed.map(\.layer).max() ?? 0
        var oneHot = [Float](repeating: 0, count: prefixLength * segmentCount)
        for (i, s) in segments.enumerated() where s >= 0 { oneHot[i * segmentCount + Int(s)] = 1 }
        let pool = MLXArray(oneHot, [prefixLength, segmentCount])
        let queryStart = prefixLength - 1
        // The last token is never attended from or to: the query before it samples it.
        let total = tokens.count - 1
        let shared = generated.flatMap { c in
            c.offset >= total && c.keyValue(layer: 0)?.0.dim(0) == 1 ? c : nil
        }

        let cache = KVCache()
        var outputs = [[MLXArray]](repeating: [], count: probed.count)
        var index = 0
        while index < total {
            let end = min(index + chunk, total)
            let offset = index
            var hidden = backbone.embedTokens(MLXArray(Array(tokens[index..<end])).reshaped(1, end - index))
            // Probed queries of this chunk (absolute positions), if any.
            let q0 = max(index, queryStart), q1 = end
            for (l, layer) in backbone.layers.enumerated() where l <= lastLayer {
                guard YuE2GPUGate.shared.wait(cancel: cancel), cancel?() != true else { throw YuE2Error.cancelled }
                let attention = layer.selfAttn
                let (q, k0, v0) = attention.projectQKV(layer.inputLayernorm(hidden), offset: offset)
                let k: MLXArray, v: MLXArray
                if let shared, let (ks, vs) = shared.keyValue(layer: l) {
                    (k, v) = (ks[0..., 0..., 0..<end, 0...], vs[0..., 0..., 0..<end, 0...])
                } else {
                    (k, v) = cache.append(layer: l, k: k0, v: v0)
                }
                if q1 > q0 {
                    let groups = attention.numHeads / attention.numKVHeads
                    let keyCount = k.dim(2)
                    let qPos = MLXArray(Int32(q0)..<Int32(q1)).reshaped(q1 - q0, 1)
                    let kPos = MLXArray(Int32(0)..<Int32(keyCount)).reshaped(1, keyCount)
                    let bias = MLX.where(kPos .> qPos, MLXArray(-Float.infinity), MLXArray(Float(0)))
                    for (p, target) in probed.enumerated() where target.layer == l {
                        let query = q[0, target.head, (q0 - index)..<(q1 - index), 0...].asType(.float32)
                        let keys = k[0, target.head / groups, 0..., 0...].asType(.float32)
                        let probs = softmax(matmul(query, keys.transposed()) * attention.scale + bias, axis: -1)
                        let pooled = matmul(probs[0..., 0..<prefixLength], pool)
                        eval(pooled)
                        outputs[p].append(pooled)
                    }
                }
                let mask: MLXFast.ScaledDotProductAttentionMaskMode = end - index == 1 ? .none : .causal
                var h = hidden + attention.attend(q, k, v, mask: mask)
                h = h + layer.mlp(layer.postAttentionLayernorm(h))
                hidden = h
                eval(hidden)
            }
            if shared == nil {
                cache.evalAll()
                cache.advance(by: end - index)
            }
            index = end
        }
        return stacked(outputs.map { concatenated($0, axis: 0) })
    }
}
