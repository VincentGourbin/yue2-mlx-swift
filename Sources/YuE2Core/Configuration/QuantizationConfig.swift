// QuantizationConfig.swift - AR-only quantization presets (plan §7 O5, T-4.5)
// Copyright 2026 Vincent Gourbin
//
// Only the AR backbone's dense projections are eligible: `self_attn.{q,k,v,o}_proj` and
// `mlp.{gate,up,down}_proj` on every layer, plus `lm_head` when `quantizeHead` is requested.
// Everything else — `nar_*` (the NAR path is a separate MoT branch, never touched by this
// preset), `embed_tokens`, every `*norm*` (RMSNorm has no `Linear` to quantize anyway),
// `vae2llm`/`llm2vae`/`time_embedder` (NAR-only auxiliaries) and `latent_pos_embed` (not even a
// `Linear`) — stays bf16. The filter below expresses this by requiring a `self_attn`/`mlp` path
// component with no `nar_` sibling, rather than a hand-maintained exclusion list: it is
// equivalent to the spec's list but cannot drift from the module tree as layers are added.
import MLX
import MLXNN

/// AR-only quantization preset (opt-in via `--quant`; `.none` keeps everything bf16).
public enum YuE2Quantization: String, CaseIterable, Sendable, Codable {
    case none
    case qint8
    case int4

    /// (bits, groupSize) for MLXNN's affine quantizer, `nil` for `.none`. Group size 64 is the
    /// plan's explicit choice (§7 O5: "quantification 8-bit affine (group 64)"); int4 reuses it
    /// since affine mode has no format-mandated group size (unlike mxfp/nvfp4).
    public var descriptor: (bits: Int, groupSize: Int)? {
        switch self {
        case .none: return nil
        case .qint8: return (8, 64)
        case .int4: return (4, 64)
        }
    }

    public var displayName: String {
        guard let descriptor else { return "none (bf16)" }
        return "\(rawValue) (\(descriptor.bits)-bit, group \(descriptor.groupSize))"
    }
}

enum YuE2QuantizationFilter {
    /// True for a leaf module path this preset is allowed to quantize: `lm_head` (only when
    /// `quantizeHead`), or any non-`nar_*` `self_attn`/`mlp` projection on the AR backbone.
    static func isEligible(path: String, quantizeHead: Bool) -> Bool {
        if path == "lm_head" { return quantizeHead }
        let components = path.split(separator: ".").map(String.init)
        guard !components.contains(where: { $0.hasPrefix("nar_") }) else { return false }
        return components.contains("self_attn") || components.contains("mlp")
    }

    /// Applies `quantization` to `model`'s eligible `Linear` leaves in place. No-op for `.none`.
    static func apply(_ quantization: YuE2Quantization, to model: Module, quantizeHead: Bool) {
        guard let descriptor = quantization.descriptor else { return }
        quantize(model: model) { path, module in
            guard module is Linear, isEligible(path: path, quantizeHead: quantizeHead) else { return nil }
            return (groupSize: descriptor.groupSize, bits: descriptor.bits, mode: .affine)
        }
    }
}
