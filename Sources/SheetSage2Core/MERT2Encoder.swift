// MERT2Encoder.swift - ConvNeXt subsampling + Conformer stack (plan/14-sheetsage2.md, facts 3-4)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXFast
import MLXNN

// Layout: everything runs channels-last `[B, T, C]`, which is MLX's native Conv1d layout, so the
// upstream `Transpose()` wrappers disappear. Conv weights are converted from PyTorch's
// `[out, in/groups, k]` to MLX's `[out, k, in/groups]` at load time (`SheetSage2Weights`).

/// ConvNeXt V2 global response normalization over the **time** axis (upstream `dim=1`).
final class GlobalResponseNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(dimensions: Int) {
        _weight.wrappedValue = MLXArray.zeros([1, 1, dimensions])
        _bias.wrappedValue = MLXArray.zeros([1, 1, dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let magnitude = MLX.sqrt(MLX.sum(x * x, axis: 1, keepDims: true))
        let normalized = magnitude / (MLX.mean(magnitude, axis: -1, keepDims: true) + 1e-6)
        return weight * (x * normalized) + bias + x
    }
}

/// `depthwise conv k7 → LayerNorm → Linear ×4 → GELU → GRN → Linear`, residual.
final class ConvNextLayer: Module {
    @ModuleInfo(key: "depthwise") var depthwise: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "expand") var expand: Linear
    @ModuleInfo(key: "grn") var grn: GlobalResponseNorm
    @ModuleInfo(key: "project") var project: Linear

    init(dimensions: Int, eps: Float) {
        _depthwise.wrappedValue = Conv1d(
            inputChannels: dimensions, outputChannels: dimensions, kernelSize: 7, padding: 3,
            groups: dimensions)
        _norm.wrappedValue = LayerNorm(dimensions: dimensions, eps: eps)
        _expand.wrappedValue = Linear(dimensions, 4 * dimensions)
        _grn.wrappedValue = GlobalResponseNorm(dimensions: 4 * dimensions)
        _project.wrappedValue = Linear(4 * dimensions, dimensions)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + project(grn(gelu(expand(norm(depthwise(x))))))
    }
}

/// Optional `LayerNorm → Conv1d k2 stride s` resampling, then `depth` ConvNeXt layers.
final class ConvNextBlock: Module {
    @ModuleInfo(key: "resample_norm") var resampleNorm: LayerNorm?
    @ModuleInfo(key: "resample_conv") var resampleConv: Conv1d?
    @ModuleInfo(key: "layers") var layers: [ConvNextLayer]

    init(inChannels: Int, outChannels: Int, stride: Int, depth: Int, eps: Float) {
        if inChannels != outChannels || stride > 1 {
            _resampleNorm.wrappedValue = LayerNorm(dimensions: inChannels, eps: eps)
            _resampleConv.wrappedValue = Conv1d(
                inputChannels: inChannels, outputChannels: outChannels, kernelSize: 2, stride: stride)
        }
        _layers.wrappedValue = (0..<depth).map { _ in ConvNextLayer(dimensions: outChannels, eps: eps) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if let resampleNorm, let resampleConv {
            h = resampleConv(resampleNorm(h))
        }
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

/// Bidirectional self-attention with biases and non-interleaved RoPE over frame positions.
final class ConformerSelfAttention: Module {
    let numHeads: Int
    let headDim: Int
    let rope: MLXNN.RoPE

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(config: MERT2Config) {
        numHeads = config.numAttentionHeads
        headDim = config.hiddenSize / config.numAttentionHeads
        rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: config.rotaryEmbeddingBase)
        let width = config.hiddenSize
        _queryProj.wrappedValue = Linear(width, width)
        _keyProj.wrappedValue = Linear(width, width)
        _valueProj.wrappedValue = Linear(width, width)
        _outProj.wrappedValue = Linear(width, width)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let q = rope(queryProj(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3))
        let k = rope(keyProj(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3))
        let v = valueProj(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped(b, t, numHeads * headDim))
    }
}

/// `Linear → GELU → Linear` (upstream `FeedForward`, keys `w_1`/`w_2`).
final class ConformerFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    init(config: MERT2Config) {
        _w1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _w2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(gelu(w1(x))) }
}

/// `LayerNorm → pointwise ×2 → GLU → depthwise k31 → LayerNorm → GELU → pointwise`, no biases.
final class ConformerConvolution: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "pointwise_in") var pointwiseIn: Conv1d
    @ModuleInfo(key: "depthwise") var depthwise: Conv1d
    @ModuleInfo(key: "depthwise_norm") var depthwiseNorm: LayerNorm
    @ModuleInfo(key: "pointwise_out") var pointwiseOut: Conv1d

    init(config: MERT2Config) {
        let width = config.hiddenSize
        let kernel = config.convDepthwiseKernelSize
        _layerNorm.wrappedValue = LayerNorm(dimensions: width, eps: config.layerNormEps)
        _pointwiseIn.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: 2 * width, kernelSize: 1, bias: false)
        _depthwise.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: width, kernelSize: kernel, padding: (kernel - 1) / 2,
            groups: width, bias: false)
        _depthwiseNorm.wrappedValue = LayerNorm(dimensions: width, eps: config.layerNormEps)
        _pointwiseOut.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: width, kernelSize: 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = glu(pointwiseIn(layerNorm(x)), axis: -1)
        return pointwiseOut(gelu(depthwiseNorm(depthwise(gated))))
    }
}

/// Macaron Conformer block: half FFN, attention, convolution, half FFN, final LayerNorm.
final class ConformerBlock: Module {
    @ModuleInfo(key: "ffn1_layer_norm") var ffn1Norm: LayerNorm
    @ModuleInfo(key: "ffn1") var ffn1: ConformerFeedForward
    @ModuleInfo(key: "attn_layer_norm") var attnNorm: LayerNorm
    @ModuleInfo(key: "attn") var attn: ConformerSelfAttention
    @ModuleInfo(key: "conv_module") var convModule: ConformerConvolution
    @ModuleInfo(key: "ffn2_layer_norm") var ffn2Norm: LayerNorm
    @ModuleInfo(key: "ffn2") var ffn2: ConformerFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(config: MERT2Config) {
        let (width, eps) = (config.hiddenSize, config.layerNormEps)
        _ffn1Norm.wrappedValue = LayerNorm(dimensions: width, eps: eps)
        _ffn1.wrappedValue = ConformerFeedForward(config: config)
        _attnNorm.wrappedValue = LayerNorm(dimensions: width, eps: eps)
        _attn.wrappedValue = ConformerSelfAttention(config: config)
        _convModule.wrappedValue = ConformerConvolution(config: config)
        _ffn2Norm.wrappedValue = LayerNorm(dimensions: width, eps: eps)
        _ffn2.wrappedValue = ConformerFeedForward(config: config)
        _finalNorm.wrappedValue = LayerNorm(dimensions: width, eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + 0.5 * ffn1(ffn1Norm(x))
        h = attn(attnNorm(h)) + h
        h = convModule(h) + h
        h = h + 0.5 * ffn2(ffn2Norm(h))
        return finalNorm(h)
    }
}

/// MERT-v2: log-mel front-end, ConvNeXt subsampling to 25 frames/s, Conformer stack.
final class MERT2Encoder: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: MelFrontend
    @ModuleInfo(key: "subsampling_module") var subsampling: [ConvNextBlock]
    @ModuleInfo(key: "layers") var layers: [ConformerBlock]

    init(config: MERT2Config) {
        _featureExtractor.wrappedValue = MelFrontend(config: config)
        let channels = [config.numMelBins] + config.subsamplingChannels
        let strides = [1, 2, 2]
        _subsampling.wrappedValue = (0..<3).map { i in
            ConvNextBlock(
                inChannels: channels[i], outChannels: channels[i + 1], stride: strides[i],
                depth: config.subsamplingDepths[i], eps: config.subsamplingLayerNormEps)
        }
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in ConformerBlock(config: config) }
        super.init()
    }
}
