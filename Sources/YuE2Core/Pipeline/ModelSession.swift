// ModelSession.swift - loads LM + tokenizer + generation defaults once for CLI commands (T-2.10)
// Copyright 2026 Vincent Gourbin

import Foundation

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

    private init(model: YuE2ForCausalLM, tokenizer: YuE2Tokenizer, config: GenerationConfig) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    }

    /// Loads `$modelsDir/YuE2-3B`: `config.json`, `model.safetensors` (628 tensors),
    /// `tokenizer.json` (T-1.5), `yue2_generation_config.json`. `quant` and `quantizeHead`
    /// select O5's AR-only quantization preset (T-4.5); `.none` is the checkpoint-native bf16
    /// path and is unaffected by either flag.
    public static func load(
        modelsDir: URL, quant: Quantization = .none, quantizeHead: Bool = false
    ) async throws -> ModelSession {
        let lmDir = modelsDir.appendingPathComponent("YuE2-3B")
        let config = try YuE2Config.load(from: lmDir.appendingPathComponent("config.json"))
        let model = try LMWeightLoader.load(directory: lmDir, config: config, quantization: quant, quantizeHead: quantizeHead)
        let tokenizer = try await YuE2Tokenizer.load(modelDir: lmDir)
        let generationConfig = try GenerationConfig.load(from: lmDir.appendingPathComponent("yue2_generation_config.json"))
        return ModelSession(model: model, tokenizer: tokenizer, config: generationConfig)
    }
}
