// OobleckDecoder.swift - Oobleck decoder topology, weight-norm already merged (plan §2.5)
// Copyright 2026 Vincent Gourbin
//
// Weights are pre-resolved by VAEWeightLoader (weight_norm merged, torch->MLX layout already
// applied), so every conv here is a plain conv over NLC with a `[out, k, in]` weight.

import Foundation
import MLX
import MLXNN

/// Plain Conv1d over NLC. `bias: false` for the decoder's final projection.
final class VAEConv1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?
    private let stride: Int
    private let padding: Int
    private let dilation: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, dilation: Int = 1, bias hasBias: Bool = true) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        _weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        _bias.wrappedValue = hasBias ? MLXArray.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

/// Plain ConvTranspose1d over NLC.
final class VAEConvTransposed1d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    private let stride: Int
    private let padding: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.stride = stride
        self.padding = padding
        _weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        _bias.wrappedValue = MLXArray.zeros([outChannels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.convTransposed1d(x, weight, stride: stride, padding: padding) + bias
    }
}

/// `layers = [SnakeBeta, Conv1d(k7, dilation d, padding 3d), SnakeBeta, Conv1d(k1)]`, output
/// `x + layers(x)`.
final class ResidualUnit: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    init(channels: Int, dilation: Int) {
        let padding = dilation * 3 // dilation * (kernelSize - 1) / 2, kernelSize = 7
        _layers.wrappedValue = [
            SnakeBeta(channels: channels),
            VAEConv1d(inChannels: channels, outChannels: channels, kernelSize: 7, padding: padding, dilation: dilation),
            SnakeBeta(channels: channels),
            VAEConv1d(inChannels: channels, outChannels: channels, kernelSize: 1),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return x + h
    }
}

/// `layers.0` SnakeBeta, `layers.1` ConvTranspose1d(kernel 2s, stride s, padding ceil(s/2)),
/// `layers.2..4` ResidualUnit with dilations 1, 3, 9.
final class DecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    init(inChannels: Int, outChannels: Int, stride: Int) {
        let padding = (stride + 1) / 2 // ceil(stride / 2) for positive Int
        _layers.wrappedValue = [
            SnakeBeta(channels: inChannels),
            VAEConvTransposed1d(inChannels: inChannels, outChannels: outChannels, kernelSize: 2 * stride, stride: stride, padding: padding),
            ResidualUnit(channels: outChannels, dilation: 1),
            ResidualUnit(channels: outChannels, dilation: 3),
            ResidualUnit(channels: outChannels, dilation: 9),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return h
    }
}

/// `layers.0` Conv1d(latentDim -> widest, k7 p3) ; `layers.1..N-2` `DecoderBlock`s narrowing
/// back down, one per stride, applied in reverse order ; `layers.N-1` SnakeBeta ; `layers.N`
/// Conv1d(-> outChannels, k7 p3, no bias) ; `layers.N+1` Identity (`final_tanh` is never true
/// on a released checkpoint — asserted at init, not silently ignored).
public final class OobleckDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [UnaryLayer]

    public init(config: VAEDecoderConfig) {
        precondition(config.useSnake && config.snakeType == "vanilla", "only the released vanilla-SnakeBeta decoder is supported")
        precondition(!config.finalTanh, "final_tanh decoders are not supported (no released checkpoint uses one)")
        let cMults = [1] + config.cMults
        let channels = config.channels

        var built: [UnaryLayer] = [
            VAEConv1d(inChannels: config.latentDim, outChannels: cMults[cMults.count - 1] * channels, kernelSize: 7, padding: 3),
        ]
        for i in stride(from: cMults.count - 1, through: 1, by: -1) {
            built.append(DecoderBlock(
                inChannels: cMults[i] * channels,
                outChannels: cMults[i - 1] * channels,
                stride: config.strides[i - 1]
            ))
        }
        built.append(SnakeBeta(channels: cMults[0] * channels))
        built.append(VAEConv1d(inChannels: cMults[0] * channels, outChannels: config.outChannels, kernelSize: 7, padding: 3, bias: false))
        built.append(Identity())
        _layers.wrappedValue = built
    }

    /// `taps?(i, output)` fires after each top-level layer, for bisection against
    /// `tap_T33.after_layer_i` fixtures.
    public func callAsFunction(_ x: MLXArray, taps: ((Int, MLXArray) -> Void)? = nil) -> MLXArray {
        var h = x
        for (i, layer) in layers.enumerated() {
            h = layer(h)
            taps?(i, h)
        }
        return h
    }
}
