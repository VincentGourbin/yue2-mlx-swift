// BartDecoder.swift - transformers 4.45 BartDecoder, post-LN, cross-attention (plan/14-sheetsage2.md, fact 6)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXFast
import MLXNN

/// Per-layer decoding cache. Self-attention keys/values live in buffers preallocated by steps of
/// 512 positions and written in place (a concatenation per token leaves MLX's cache holding a
/// buffer of every size: 20 GB on a 500 s song); the cross-attention keys/values are projected
/// once from the encoder memory.
final class BartLayerCache {
    static let step = 512
    var keys: MLXArray?
    var values: MLXArray?
    var length = 0
    var crossKeys: MLXArray?
    var crossValues: MLXArray?

    /// Appends `[B, H, T, D]` keys/values; returns the valid prefix of both buffers.
    func append(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        let needed = length + k.dim(2)
        if keys == nil || needed > keys!.dim(2) {
            let capacity = (needed + Self.step - 1) / Self.step * Self.step
            var shape = k.shape
            shape[2] = capacity
            let (newK, newV) = (MLXArray.zeros(shape, dtype: k.dtype), MLXArray.zeros(shape, dtype: v.dtype))
            if let keys, let values, length > 0 {
                newK[0..., 0..., ..<length, 0...] = keys[0..., 0..., ..<length, 0...]
                newV[0..., 0..., ..<length, 0...] = values[0..., 0..., ..<length, 0...]
            }
            (keys, values) = (newK, newV)
        }
        keys![0..., 0..., length ..< needed, 0...] = k
        values![0..., 0..., length ..< needed, 0...] = v
        length = needed
        return (keys![0..., 0..., ..<length, 0...], values![0..., 0..., ..<length, 0...])
    }
}

/// Decoder cache for one sequence: one `BartLayerCache` per layer plus the number of positions seen.
public final class BartDecoderCache {
    var layers: [BartLayerCache]
    public internal(set) var offset = 0

    init(layerCount: Int) {
        layers = (0..<layerCount).map { _ in BartLayerCache() }
    }
}

/// `BartSdpaAttention`: biased projections, `head_dim^-0.5` scaling.
final class BartAttention: Module {
    let numHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(width: Int, heads: Int) {
        numHeads = heads
        headDim = width / heads
        _qProj.wrappedValue = Linear(width, width)
        _kProj.wrappedValue = Linear(width, width)
        _vProj.wrappedValue = Linear(width, width)
        _outProj.wrappedValue = Linear(width, width)
        super.init()
    }

    func heads(_ x: MLXArray) -> MLXArray {
        x.reshaped(x.dim(0), x.dim(1), numHeads, headDim).transposed(0, 2, 1, 3)
    }

    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, causal: Bool) -> MLXArray {
        let (b, t) = (q.dim(0), q.dim(2))
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(),
            mask: causal ? .causal : .none)
        return outProj(output.transposed(0, 2, 1, 3).reshaped(b, t, numHeads * headDim))
    }
}

/// `self-attn → +res → LN → cross-attn → +res → LN → fc1 → GELU → fc2 → +res → LN`.
final class BartDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: BartAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: BartAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(width: Int, heads: Int, ffn: Int) {
        _selfAttn.wrappedValue = BartAttention(width: width, heads: heads)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _encoderAttn.wrappedValue = BartAttention(width: width, heads: heads)
        _encoderAttnNorm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _fc1.wrappedValue = Linear(width, ffn)
        _fc2.wrappedValue = Linear(ffn, width)
        _finalNorm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, memory: MLXArray, cache: BartLayerCache) -> MLXArray {
        // Self-attention: causal over the new tokens when more than one arrives at once; a single
        // cached step attends to every past position, which is exactly the causal row.
        let q = selfAttn.heads(selfAttn.qProj(x))
        let (k, v) = cache.append(selfAttn.heads(selfAttn.kProj(x)), selfAttn.heads(selfAttn.vProj(x)))
        let causal = x.dim(1) > 1
        precondition(!causal || k.dim(2) == x.dim(1), "multi-token steps are only supported on an empty cache")
        var h = selfAttnNorm(x + selfAttn.attend(q, k, v, causal: causal))

        if cache.crossKeys == nil {
            cache.crossKeys = encoderAttn.heads(encoderAttn.kProj(memory))
            cache.crossValues = encoderAttn.heads(encoderAttn.vProj(memory))
        }
        let cq = encoderAttn.heads(encoderAttn.qProj(h))
        h = encoderAttnNorm(h + encoderAttn.attend(cq, cache.crossKeys!, cache.crossValues!, causal: false))
        return finalNorm(h + fc2(gelu(fc1(h))))
    }
}

/// BART decoder: shared token embedding (unscaled), learned positions offset by 2,
/// `layernorm_embedding`, post-LN layers, no final LayerNorm.
final class BartDecoder: Module {
    static let positionOffset = 2

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    @ModuleInfo(key: "layernorm_embedding") var layernormEmbedding: LayerNorm
    @ModuleInfo(key: "layers") var layers: [BartDecoderLayer]

    init(config: SheetSage2Config) {
        let width = config.hiddenSize
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: width)
        _embedPositions.wrappedValue = Embedding(
            embeddingCount: config.maxOutputSeqLen + Self.positionOffset, dimensions: width)
        _layernormEmbedding.wrappedValue = LayerNorm(dimensions: width, eps: 1e-5)
        _layers.wrappedValue = (0..<config.decoderLayers).map { _ in
            BartDecoderLayer(width: width, heads: config.numAttentionHeads, ffn: config.intermediateSize)
        }
        super.init()
    }

    /// `tokens` `[B, T]` → hidden `[B, T, width]`, advancing `cache`.
    func callAsFunction(_ tokens: MLXArray, memory: MLXArray, cache: BartDecoderCache) -> MLXArray {
        let t = tokens.dim(1)
        let positions = MLXArray(Int32(cache.offset + Self.positionOffset)..<Int32(cache.offset + Self.positionOffset + t))
        var h = layernormEmbedding(embedTokens(tokens) + embedPositions(positions))
        for (layer, layerCache) in zip(layers, cache.layers) {
            h = layer(h, memory: memory, cache: layerCache)
        }
        cache.offset += t
        return h
    }
}
