// SheetSage2Config.swift - merged SheetSage2 snapshot config.json (plan/14-sheetsage2.md)
// Copyright 2026 Vincent Gourbin

import Foundation

/// MERT-v2 encoder topology (`backbone_config` of the SheetSage2 snapshot, `MERT2Config` upstream).
public struct MERT2Config: Codable, Sendable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var convDepthwiseKernelSize: Int
    public var layerNormEps: Float
    public var subsamplingLayerNormEps: Float
    public var subsamplingChannels: [Int]
    public var subsamplingDepths: [Int]
    public var numMelBins: Int
    public var nFFT: Int
    public var winLength: Int
    public var hopLength: Int
    public var samplingRate: Int
    public var rotaryEmbeddingBase: Float
    public var inputsToLogitsRatio: Int
    public var minimumInputSamples: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case convDepthwiseKernelSize = "conv_depthwise_kernel_size"
        case layerNormEps = "layer_norm_eps"
        case subsamplingLayerNormEps = "subsampling_layer_norm_eps"
        case subsamplingChannels = "subsampling_channels"
        case subsamplingDepths = "subsampling_depths"
        case numMelBins = "num_mel_bins"
        case nFFT = "n_fft"
        case winLength = "win_length"
        case hopLength = "hop_length"
        case samplingRate = "sampling_rate"
        case rotaryEmbeddingBase = "rotary_embedding_base"
        case inputsToLogitsRatio = "inputs_to_logits_ratio"
        case minimumInputSamples = "minimum_input_samples"
    }
}

/// Top-level SheetSage2 config (`SheetSage2Config` upstream): encoder, BART decoder, window.
public struct SheetSage2Config: Codable, Sendable {
    public var backbone: MERT2Config
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var decoderLayers: Int
    public var numAttentionHeads: Int
    public var vocabSize: Int
    public var maxOutputSeqLen: Int
    public var inputAudioLength: Double
    public var timeHz: Int
    public var samplingRate: Int
    public var padTokenId: Int
    public var weightsFormat: String
    public var tokenizerFingerprint: String?
    public var loraAlpha: Double?
    public var loraRank: Int?
    /// Set on a prequantized pack (`weights_format: mlx-quantized`): affine quantization of the
    /// Conformer's linear layers, stored in MLX layout (`SheetSage2Pack`).
    public var quantization: Quantization?

    public struct Quantization: Codable, Sendable, Equatable {
        public var bits: Int
        public var groupSize: Int
        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    enum CodingKeys: String, CodingKey {
        case backbone = "backbone_config"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case decoderLayers = "decoder_layers"
        case numAttentionHeads = "num_attention_heads"
        case vocabSize = "vocab_size"
        case maxOutputSeqLen = "max_output_seq_len"
        case inputAudioLength = "input_audio_length"
        case timeHz = "time_hz"
        case samplingRate = "sampling_rate"
        case padTokenId = "pad_token_id"
        case weightsFormat = "weights_format"
        case tokenizerFingerprint = "tokenizer_fingerprint"
        case loraAlpha = "lora_alpha"
        case loraRank = "lora_rank"
        case quantization
    }

    /// Samples in one model window (300 s × 24 kHz on the released checkpoint), rounded up to the
    /// encoder stride exactly as `SheetSage2Model._prepare_audio` pads it.
    public var windowSamples: Int {
        let window = Int((inputAudioLength * Double(samplingRate)).rounded())
        let stride = backbone.inputsToLogitsRatio
        return window % stride == 0 ? window : window + stride - window % stride
    }

    public static func load(_ url: URL) throws -> SheetSage2Config {
        try JSONDecoder().decode(SheetSage2Config.self, from: Data(contentsOf: url))
    }
}

public enum SheetSage2Error: Error, CustomStringConvertible {
    case weightMismatch(String)
    case invalidAudio(String)
    case invalidSequence(String)

    public var description: String {
        switch self {
        case .weightMismatch(let message): "SheetSage2 weights: \(message)"
        case .invalidAudio(let message): "SheetSage2 audio: \(message)"
        case .invalidSequence(let message): "SheetSage2 tokens: \(message)"
        }
    }
}
