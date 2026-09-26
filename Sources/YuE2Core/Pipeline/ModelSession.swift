// ModelSession.swift - loads LM + tokenizer + generation defaults once for CLI commands (T-2.10)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN

/// The checkpoint's native compute dtype (`.bf16`) or fp16 (T-6.2, E2 — a Neural Engine backend
/// needs fp16, same motivation as `VAEPrecision`, T-6.1). Only *floating* leaves are cast: a
/// quantization preset's packed `uint32` weights are untouched (casting those would numerically
/// reinterpret bit patterns as nonsense, not convert a value — piège n°10), while their
/// `scales`/`biases` (still bf16) are cast like any other float.
public enum YuE2ComputePrecision: String, CaseIterable, Sendable {
    case bf16
    case fp16

    var dtype: DType? { self == .fp16 ? .float16 : nil }
}

extension YuE2ForCausalLM {
    /// Casts every floating (non-quantized) parameter to `precision`'s dtype, tensor by tensor
    /// (piège n°8: never the whole parameter tree at once). No-op for `.bf16`. A quantization
    /// preset's packed `uint32` weights are untouched (`isFloatingPoint` excludes them); their
    /// `scales`/`biases` are cast like any other float.
    ///
    /// `where include`: restrict the cast to some keys — **required** under stage-scoped
    /// residency. A non-resident branch holds *unevaluated* init graphs (random `Linear` weights,
    /// lazily quantized); casting and evaluating their `scales`/`biases` forces the whole
    /// quantized weight to materialize, i.e. allocates the branch that residency was meant to
    /// skip (measured 2026-09-26: +1.44 GB, the NAR path, on every AR-only fp16 load — the
    /// iPhone's plan-phase peak).
    public func applyPrecision(_ precision: YuE2ComputePrecision, where include: ((String) -> Bool)? = nil) {
        guard let dtype = precision.dtype else { return }
        var casted = [String: MLXArray]()
        for (key, value) in parameters().flattened()
        where value.dtype.isFloatingPoint && value.dtype != dtype && (include?(key) ?? true) {
            let c = value.asType(dtype)
            eval(c)
            casted[key] = c
        }
        update(parameters: ModuleParameters.unflattened(casted))
    }
}

/// The LM checkpoint, tokenizer, and checkpoint-native generation defaults, loaded once and
/// shared by every CLI command that generates (`plan`, `semantic`, and future `generate`).
public final class ModelSession {
    /// AR quantization preset (O5, T-4.5): `.none` keeps the checkpoint's native bf16;
    /// `.qint8`/`.int4` quantize `self_attn.*_proj`/`mlp.*` (opt-in, `--quant`). Re-exported as a
    /// type alias of `YuE2Quantization` so existing call sites (`ModelSession.Quantization`)
    /// keep compiling unchanged.
    public typealias Quantization = YuE2Quantization

    public let model: YuE2ForCausalLM
    public let tokenizer: YuE2Tokenizer
    public let config: GenerationConfig
    /// Which LM branches are currently loaded (stage-scoped residency, `WeightResidency.swift`).
    public private(set) var resident: YuE2WeightResidency

    private let lmDirectory: URL
    private let quant: Quantization
    private let quantizeHead: Bool
    private let precision: YuE2ComputePrecision

    private init(
        model: YuE2ForCausalLM, tokenizer: YuE2Tokenizer, config: GenerationConfig,
        resident: YuE2WeightResidency, lmDirectory: URL, quant: Quantization, quantizeHead: Bool,
        precision: YuE2ComputePrecision
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.resident = resident
        self.lmDirectory = lmDirectory
        self.quant = quant
        self.quantizeHead = quantizeHead
        self.precision = precision
    }

    /// Loads `path`'s weights from the same checkpoint this session came from (no-op when
    /// already resident) and casts them to this session's precision.
    public func loadWeights(of path: YuE2WeightPath) throws {
        let flag: YuE2WeightResidency = path == .ar ? .ar : .nar
        guard !resident.contains(flag) else { return }
        try LMWeightLoader.load(path: path, into: model, directory: lmDirectory, quantization: quant, quantizeHead: quantizeHead)
        model.applyPrecision(precision) { YuE2WeightPath.of(key: $0) == path }
        resident.insert(flag)
    }

    /// Drops `path`'s weights (`YuE2ForCausalLM.releaseWeights(of:)`); `loadWeights(of:)` brings
    /// them back. Returns the bytes released.
    @discardableResult
    public func releaseWeights(of path: YuE2WeightPath) -> Int {
        let flag: YuE2WeightResidency = path == .ar ? .ar : .nar
        guard resident.contains(flag) else { return 0 }
        resident.remove(flag)
        return model.releaseWeights(of: path)
    }

    /// Drops every LM weight; `loadWeights(of:)` can still bring a branch back.
    @discardableResult
    public func releaseAllWeights() -> Int {
        resident = []
        return model.releaseAllWeights()
    }

    /// Loads `$modelsDir/YuE2-3B`: `config.json`, `model.safetensors` (628 tensors),
    /// `tokenizer.json` (T-1.5), `yue2_generation_config.json`. `quant` and `quantizeHead`
    /// select O5's AR-only or T-6.2's "-all" quantization presets; `.none` is the
    /// checkpoint-native bf16 path and is unaffected by either flag. `precision: .fp16` casts
    /// every remaining floating parameter after quantization (piège n°8: tensor by tensor).
    /// `residency`: which branches to load now (`.all` default; `[.ar]` when the NAR will be
    /// brought in later with `loadWeights(of: .nar)` — see `LMWeightLoader.load(residency:)` for
    /// the one path that ignores it).
    public static func load(
        modelsDir: URL, quant: Quantization = .none, quantizeHead: Bool = false,
        precision: YuE2ComputePrecision = .bf16, residency: YuE2WeightResidency = .all
    ) async throws -> ModelSession {
        let lmDir = modelsDir.appendingPathComponent("YuE2-3B")
        let config = try YuE2Config.load(from: lmDir.appendingPathComponent("config.json"))
        let effective: YuE2WeightResidency =
            (quant != .none && !YuE2PrequantizedCheckpoint.exists(lmDirectory: lmDir, quantization: quant, quantizeHead: quantizeHead))
            ? .all : residency
        let model = try LMWeightLoader.load(
            directory: lmDir, config: config, quantization: quant, quantizeHead: quantizeHead, residency: residency)
        model.applyPrecision(precision, where: effective == .all ? nil : { effective.retains(key: $0) })
        let tokenizer = try await YuE2Tokenizer.load(modelDir: lmDir)
        let generationConfig = try GenerationConfig.load(from: lmDir.appendingPathComponent("yue2_generation_config.json"))
        return ModelSession(
            model: model, tokenizer: tokenizer, config: generationConfig, resident: effective,
            lmDirectory: lmDir, quant: quant, quantizeHead: quantizeHead, precision: precision)
    }
}
