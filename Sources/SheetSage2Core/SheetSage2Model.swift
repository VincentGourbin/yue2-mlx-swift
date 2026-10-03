// SheetSage2Model.swift - encoder memory and decoder logits (plan/14-sheetsage2.md, facts 2 and 5)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// Intermediate encoder tensors, for parity bisection (`encodeWithTaps`).
public struct SheetSage2EncoderTaps {
    public var mel: MLXArray
    public var subsampling: [MLXArray]
    public var layers: [Int: MLXArray]
    public var mixed: MLXArray
    public var memory: MLXArray
}

/// SheetSage2: MERT-v2 encoder (LoRA merged), learned layer mix, projection, BART decoder whose
/// output projection is the token embedding.
public final class SheetSage2Model: Module {
    public let config: SheetSage2Config
    /// Chunked STFT and ConvNeXt (lean profiles); `nil` computes each stage on the whole window.
    public var chunkFrames: Int?
    /// `false` once `releaseEncoder()` ran: decoding still works, encoding throws.
    public private(set) var encoderResident = true

    @ModuleInfo(key: "encoder") var encoder: MERT2Encoder
    @ParameterInfo(key: "layer_weight") var layerWeight: MLXArray
    @ModuleInfo(key: "encoder_projection") var encoderProjection: Linear
    @ModuleInfo(key: "decoder") var decoder: BartDecoder

    public init(config: SheetSage2Config) {
        self.config = config
        _encoder.wrappedValue = MERT2Encoder(config: config.backbone)
        _layerWeight.wrappedValue = MLXArray.zeros([config.backbone.numHiddenLayers + 1])
        _encoderProjection.wrappedValue = Linear(config.backbone.hiddenSize, config.hiddenSize)
        _decoder.wrappedValue = BartDecoder(config: config)
        super.init()
    }

    /// Zero-pads `waveform` (24 kHz mono, not normalized) to the fixed model window, exactly as
    /// `_prepare_audio`: the encoder always attends to the whole window, silence included.
    public func padToWindow(_ waveform: MLXArray) throws -> MLXArray {
        let samples = waveform.dim(0)
        let window = config.windowSamples
        guard samples >= config.backbone.minimumInputSamples else {
            throw SheetSage2Error.invalidAudio("\(samples) samples, at least \(config.backbone.minimumInputSamples) required")
        }
        guard samples <= window else {
            throw SheetSage2Error.invalidAudio("\(samples) samples exceed one \(window)-sample window")
        }
        let x = waveform.asType(.float32)
        return samples == window ? x : concatenated([x, MLXArray.zeros([window - samples])], axis: 0)
    }

    /// Encoder memory `[1, frames, hidden]` for one window, plus every intermediate tensor.
    public func encodeWithTaps(_ waveform: MLXArray, tapLayers: Set<Int> = []) throws -> SheetSage2EncoderTaps {
        guard encoderResident else {
            throw SheetSage2Error.weightMismatch("the encoder was released (releaseEncoder); load the model again to encode")
        }
        let padded = try padToWindow(waveform)
        let mel = encoder.featureExtractor(padded, chunkFrames: chunkFrames)
        // Evaluate stage by stage: one lazy graph over the whole 300 s window keeps every stage's
        // transients alive at once (8 GB peak measured, 2.95 GB evaluated per stage, fp16).
        eval(mel)
        // Each stage runs in its own weights' dtype: the ConvNeXt (float32 under reduced precision,
        // see `SheetSage2Model.load`), the Conformer stack, and the layer mix + projection (the model's
        // compute dtype).
        let convDType = encoder.subsampling[0].layers[0].expand.weight.dtype
        let stackDType = encoder.layers[0].ffn1.w1.weight.dtype
        let mixDType = encoderProjection.weight.dtype
        var hidden = mel.asType(convDType)[.newAxis]
        var subsampling = [MLXArray]()
        for block in encoder.subsampling {
            hidden = block(hidden, chunkFrames: chunkFrames)
            eval(hidden)
            subsampling.append(hidden)
        }
        hidden = hidden.asType(stackDType)
        let weights = softmax(layerWeight.asType(.float32), axis: 0).asType(mixDType)
        var mixed = hidden.asType(mixDType) * weights[0]
        var layers = [Int: MLXArray]()
        for (i, layer) in encoder.layers.enumerated() {
            hidden = layer(hidden)
            mixed = mixed + hidden.asType(mixDType) * weights[i + 1]
            eval(hidden, mixed)
            if tapLayers.contains(i) { layers[i] = hidden }
        }
        let memory = encoderProjection(mixed)
        eval(memory)
        return SheetSage2EncoderTaps(mel: mel, subsampling: subsampling, layers: layers, mixed: mixed, memory: memory)
    }

    /// Encoder memory `[1, frames, hidden]` for one window.
    public func encode(_ waveform: MLXArray) throws -> MLXArray {
        try encodeWithTaps(waveform).memory
    }

    /// Drops the encoder's weights (≈ 1.2 GB in float16) once every window is encoded: the
    /// decoder only needs its own weights and the encoder memories.
    public func releaseEncoder() {
        let empty = encoder.parameters().flattened().map { ($0.0, MLXArray.zeros([0], dtype: $0.1.dtype)) }
        encoder.update(parameters: ModuleParameters.unflattened(empty))
        encoderResident = false
        Memory.clearCache()
    }

    public func makeCache() -> BartDecoderCache {
        BartDecoderCache(layerCount: config.decoderLayers)
    }

    /// Vocabulary logits `[1, T, vocab]` for `tokens` `[1, T]`, advancing `cache`.
    public func decode(_ tokens: MLXArray, memory: MLXArray, cache: BartDecoderCache) -> MLXArray {
        let hidden = decoder(tokens, memory: memory.asType(decoder.embedTokens.weight.dtype), cache: cache)
        return matmul(hidden, decoder.embedTokens.weight.T)
    }
}
