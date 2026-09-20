// YuE2Config.swift - LM hyperparameters (plan §2.2, modeling_yue2.py YuE2Config)
// Copyright 2026 Vincent Gourbin

import Foundation

/// Backbone hyperparameters, decoded from `YuE2-3B/config.json`.
public struct YuE2Config: Codable, Equatable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var tieWordEmbeddings: Bool
    public var latentType: String
    public var latentDim: Int
    public var maxLatentFrames: Int
    public var timestepShift: Float

    public init(
        hiddenSize: Int = 2048,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        intermediateSize: Int = 6144,
        vocabSize: Int = 184704,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000,
        maxPositionEmbeddings: Int = 24576,
        tieWordEmbeddings: Bool = false,
        latentType: String = "vae",
        latentDim: Int = 64,
        maxLatentFrames: Int = 24576,
        timestepShift: Float = 1.0
    ) {
        precondition(latentType == "vae", "YuE2 inference supports only latent_type=\"vae\"")
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
        self.latentType = latentType
        self.latentDim = latentDim
        self.maxLatentFrames = maxLatentFrames
        self.timestepShift = timestepShift
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case latentType = "latent_type"
        case latentDim = "latent_dim"
        case maxLatentFrames = "max_latent_frames"
        case timestepShift = "timestep_shift"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            hiddenSize: try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048,
            numHiddenLayers: try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28,
            numAttentionHeads: try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16,
            numKeyValueHeads: try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8,
            headDim: try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128,
            intermediateSize: try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144,
            vocabSize: try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 184704,
            rmsNormEps: try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            ropeTheta: try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000,
            maxPositionEmbeddings: try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 24576,
            tieWordEmbeddings: try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false,
            latentType: try c.decodeIfPresent(String.self, forKey: .latentType) ?? "vae",
            latentDim: try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? 64,
            maxLatentFrames: try c.decodeIfPresent(Int.self, forKey: .maxLatentFrames) ?? 24576,
            timestepShift: try c.decodeIfPresent(Float.self, forKey: .timestepShift) ?? 1.0
        )
    }

    /// Decodes `YuE2-3B/config.json`.
    public static func load(from url: URL) throws -> YuE2Config {
        try JSONDecoder().decode(YuE2Config.self, from: Data(contentsOf: url))
    }
}
