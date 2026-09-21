// QuantizationConfig.swift - quantization presets: AR-only (T-4.5, O5) and "-all" (T-6.2, E2)
// Copyright 2026 Vincent Gourbin
//
// AR-only presets (`.qint8`/`.int4`): only the AR backbone's dense projections are eligible —
// `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` on every layer, plus `lm_head` when
// `quantizeHead` is requested. Everything else — `nar_*` (a separate MoT branch), `embed_tokens`,
// every `*norm*` (RMSNorm has no `Linear` to quantize anyway), `vae2llm`/`llm2vae`/`time_embedder`
// (NAR-only auxiliaries) and `latent_pos_embed` (not even a `Linear`) — stays bf16.
//
// "-all" presets (`.qint8All`/`.int4All`, T-6.2): the iPhone pack needs both MoT branches shrunk,
// so `nar_self_attn`/`nar_mlp` and `embed_tokens` (via MLXNN's `QuantizedEmbedding`, the same
// `quantize(model:filter:)` call handles both module kinds) become eligible too. `vae2llm`/
// `llm2vae`/`time_embedder` still stay floating — they are small and layer-count-independent, not
// worth the dequantization cost on every NAR step.
//
// `.int4Mixed` (E2 resolution, T-6.2): `int4All`'s NAR path failed real-fixture parity
// (velocity_rel 0.152 vs a 0.03 budget) — the midpoint solver runs 64 `velocity` evaluations per
// chunk, so 4-bit NAR-projection error compounds across the whole ODE trajectory in a way the
// AR path (one pass per token) never sees. `.int4Mixed` keeps the AR path (and `embed_tokens`/
// `lm_head`) at 4 bits, matching `.int4All`'s already-verified numbers there, but reverts only
// `nar_self_attn`/`nar_mlp` to 8 bits.
import MLX
import MLXNN

/// Quantization preset (opt-in via `--quant`; `.none` keeps everything bf16).
public enum YuE2Quantization: String, CaseIterable, Sendable, Codable {
    case none
    case qint8
    case int4
    case qint8All = "qint8-all"
    case int4All = "int4-all"
    case int4Mixed = "int4-mixed"

    /// (bits, groupSize) for MLXNN's affine quantizer, `nil` for `.none`. Group size 64 is the
    /// plan's explicit choice (§7 O5: "quantification 8-bit affine (group 64)"); int4 reuses it
    /// since affine mode has no format-mandated group size (unlike mxfp/nvfp4). For `.int4Mixed`
    /// this is the AR/embedding descriptor — `YuE2QuantizationFilter.descriptor(for:quantization:)`
    /// overrides it to 8-bit for `nar_*` paths.
    public var descriptor: (bits: Int, groupSize: Int)? {
        switch self {
        case .none: return nil
        case .qint8, .qint8All: return (8, 64)
        case .int4, .int4All, .int4Mixed: return (4, 64)
        }
    }

    /// `true` for the E2 "-all"/"-mixed" presets: both MoT branches (AR *and* NAR) plus
    /// `embed_tokens` are eligible, and the exported pack omits `latent_pos_embed.pe` (recomputed
    /// instead — `NARModules.computeLatentPositions`).
    public var isAllPreset: Bool {
        self == .qint8All || self == .int4All || self == .int4Mixed
    }

    public var displayName: String {
        guard let descriptor else { return "none (bf16)" }
        if self == .int4Mixed { return "\(rawValue) (AR/embed 4-bit, NAR 8-bit, group \(descriptor.groupSize))" }
        return "\(rawValue) (\(descriptor.bits)-bit, group \(descriptor.groupSize))"
    }
}

enum YuE2QuantizationFilter {
    /// True for a leaf module path this preset is allowed to quantize.
    ///
    /// AR-only (`includeNAR == false`, T-4.5): `lm_head` (only when `quantizeHead`), or any
    /// non-`nar_*` `self_attn`/`mlp` projection.
    ///
    /// "-all"/"-mixed" (`includeNAR == true`, T-6.2): the same, plus `nar_self_attn`/`nar_mlp` and
    /// `embed_tokens` — the `nar_*` exclusion is lifted entirely (`.int4Mixed` still quantizes
    /// those paths, just at a different bit width — see `descriptor(for:quantization:)`).
    static func isEligible(path: String, quantizeHead: Bool, includeNAR: Bool) -> Bool {
        let components = path.split(separator: ".").map(String.init)
        if components.contains("lm_head") { return quantizeHead }
        if components.contains("embed_tokens") { return includeNAR }
        if !includeNAR, components.contains(where: { $0.hasPrefix("nar_") }) { return false }
        if components.contains("self_attn") || components.contains("mlp") { return true }
        return includeNAR && (components.contains("nar_self_attn") || components.contains("nar_mlp"))
    }

    /// Per-path (bits, groupSize): `quantization.descriptor` for everything, except `.int4Mixed`'s
    /// `nar_*` paths, which get 8 bits instead of 4 (E2 resolution — see the file-level comment).
    static func descriptor(for path: String, quantization: YuE2Quantization) -> (bits: Int, groupSize: Int)? {
        guard let base = quantization.descriptor else { return nil }
        guard quantization == .int4Mixed else { return base }
        let isNAR = path.split(separator: ".").map(String.init).contains(where: { $0.hasPrefix("nar_") })
        return isNAR ? (8, base.groupSize) : base
    }

    /// Applies `quantization` to `model`'s eligible `Linear`/`Embedding` leaves in place.
    /// No-op for `.none`.
    static func apply(_ quantization: YuE2Quantization, to model: Module, quantizeHead: Bool) {
        guard quantization.descriptor != nil else { return }
        let includeNAR = quantization.isAllPreset
        quantize(model: model) { path, module in
            guard module is Linear || module is Embedding,
                isEligible(path: path, quantizeHead: quantizeHead, includeNAR: includeNAR),
                let d = descriptor(for: path, quantization: quantization)
            else { return nil }
            return (groupSize: d.groupSize, bits: d.bits, mode: .affine)
        }
    }
}

/// T-6.2 (E2) sensitivity sweep only — measurement scaffolding for `yue2 parity nar --sweep`,
/// never a production preset (that's `YuE2Quantization`/`YuE2QuantizationFilter` above). Fixes
/// AR/`embed_tokens`/`lm_head` at `int4-mixed`'s already-verified int4 g64 affine, then quantizes
/// only the targeted slice of the NAR path at a caller-chosen (bits, mode, groupSize) — one
/// projection family at a time (q/k/v, `o_proj`, gate/up, `down_proj`), or every NAR projection at
/// once (`family: nil`, e.g. the mxfp8 g32 run), to see which slice actually needs more than 4
/// bits. Everything else in NAR stays int4 g64 affine, matching `int4-mixed`.
public enum YuE2NARSweep {
    public enum Family: String, CaseIterable, Sendable {
        case qkv, oProj = "o-proj", gateUp = "gate-up", downProj = "down-proj"

        fileprivate func matches(_ path: String) -> Bool {
            switch self {
            case .qkv:
                return path.contains("nar_self_attn")
                    && (path.contains("q_proj") || path.contains("k_proj") || path.contains("v_proj"))
            case .oProj:
                return path.contains("nar_self_attn") && path.contains("o_proj")
            case .gateUp:
                return path.contains("nar_mlp") && (path.contains("gate_proj") || path.contains("up_proj"))
            case .downProj:
                return path.contains("nar_mlp") && path.contains("down_proj")
            }
        }
    }

    /// `family: nil` targets every `nar_self_attn`/`nar_mlp` leaf (the mxfp8 g32 run); a specific
    /// family targets only that slice, leaving the rest of NAR at int4 g64 affine.
    public static func apply(
        to model: Module, family: Family?, bits: Int, mode: QuantizationMode, groupSize: Int
    ) {
        quantize(model: model) { path, module in
            guard module is Linear || module is Embedding else { return nil }
            let components = path.split(separator: ".").map(String.init)
            let isNAR = components.contains(where: { $0.hasPrefix("nar_") })
            if isNAR {
                if family == nil || family!.matches(path) {
                    return (groupSize: groupSize, bits: bits, mode: mode)
                }
                return (groupSize: 64, bits: 4, mode: .affine)
            }
            if components.contains("lm_head") || components.contains("embed_tokens") {
                return (groupSize: 64, bits: 4, mode: .affine)
            }
            if components.contains("self_attn") || components.contains("mlp") {
                return (groupSize: 64, bits: 4, mode: .affine)
            }
            return nil
        }
    }
}
