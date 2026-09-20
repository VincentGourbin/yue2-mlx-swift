// OobleckEncoder.swift - Oobleck encoder topology, weight-norm already merged (T-5.1)
// Copyright 2026 Vincent Gourbin
//
// Mirrors OobleckDecoder.swift's conventions (weights pre-resolved by VAEWeightLoader: weight_norm
// merged, torch->MLX layout already applied). Every conv here is a plain Conv1d over NLC with a
// `[out, k, in]` weight -- the encoder never uses ConvTranspose1d.

import Foundation
import MLX
import MLXNN

/// `layers = [ResidualUnit(in,1), ResidualUnit(in,3), ResidualUnit(in,9), SnakeBeta(in),
/// Conv1d(in->out, k=2*stride, stride=stride, padding=ceil(stride/2))]` -- residuals THEN
/// downsample, the reverse order of `DecoderBlock` (which upsamples first).
final class EncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    init(inChannels: Int, outChannels: Int, stride: Int) {
        let padding = (stride + 1) / 2 // ceil(stride / 2) for positive Int
        _layers.wrappedValue = [
            ResidualUnit(channels: inChannels, dilation: 1),
            ResidualUnit(channels: inChannels, dilation: 3),
            ResidualUnit(channels: inChannels, dilation: 9),
            SnakeBeta(channels: inChannels),
            VAEConv1d(inChannels: inChannels, outChannels: outChannels, kernelSize: 2 * stride, stride: stride, padding: padding),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return h
    }
}

/// `layers.0` Conv1d(inChannels -> narrowest, k7 p3, with bias); `layers.1..N-2` `EncoderBlock`s
/// widening forward (not reversed, unlike the decoder), one per stride, in config order;
/// `layers.N-1` SnakeBeta; `layers.N` Conv1d(-> latentDim, **k3 p1**, with bias -- narrower
/// kernel/padding than the decoder's final k7/p3, and no `bias: false` here).
public final class OobleckEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    public init(config: VAEEncoderConfig) {
        precondition(config.useSnake, "only the released snake-activation encoder is supported")
        let cMults = [1] + config.cMults
        let channels = config.channels

        var built: [UnaryLayer] = [
            VAEConv1d(inChannels: config.inChannels, outChannels: cMults[0] * channels, kernelSize: 7, padding: 3),
        ]
        for i in 0..<(cMults.count - 1) {
            built.append(EncoderBlock(
                inChannels: cMults[i] * channels,
                outChannels: cMults[i + 1] * channels,
                stride: config.strides[i]
            ))
        }
        built.append(SnakeBeta(channels: cMults[cMults.count - 1] * channels))
        built.append(VAEConv1d(inChannels: cMults[cMults.count - 1] * channels, outChannels: config.latentDim, kernelSize: 3, padding: 1))
        _layers.wrappedValue = built
    }

    /// `taps?(i, output)` fires after each top-level layer, for bisection against
    /// `tap_encode_T*.after_layer_i` fixtures.
    public func callAsFunction(_ x: MLXArray, taps: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        var h = x
        for (i, layer) in layers.enumerated() {
            h = layer(h)
            taps?(i, h)
        }
        return h
    }
}
