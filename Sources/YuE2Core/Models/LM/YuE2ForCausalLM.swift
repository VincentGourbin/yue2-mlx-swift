// YuE2ForCausalLM.swift - backbone + lm_head + NAR auxiliary modules (plan §2.2)
// Copyright 2026 Vincent Gourbin

import MLX
import MLXNN

/// Sinusoidal timestep → `mlp` → `hiddenSize` (NAR flow-matching input; math lands in Jalon 3).
///
/// The checkpoint's `TimestepEmbedder.mlp` is a `nn.Sequential(Linear, SiLU, Linear)`, keyed
/// `mlp.0`/`mlp.2`. MLX's `Sequential` doesn't reproduce those keys — and, worse, a hand-written
/// submodule keyed `"0"`/`"2"` doesn't work either: `ModuleParameters.unflattened` treats any
/// all-digit path segment as a list index rather than a named child, so it builds an *array*
/// where a *module* is expected and `update(parameters:)` traps. The loader must instead remap
/// `mlp.0` → `fc1` and `mlp.2` → `fc2` (same fix flux-2's `WeightLoader` uses for BFL's
/// `img_mlp.0`/`img_mlp.2`).
final class TimeEmbedder: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        _fc1 = ModuleInfo(wrappedValue: Linear(frequencyEmbeddingSize, hiddenSize, bias: true), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "fc2")
        super.init()
    }
}

/// Non-learnable sinusoidal position table for NAR audio latent frames, loaded verbatim from
/// the checkpoint (never recomputed — plan §2.2/§9 pitfall #3).
final class LatentPosEmbed: Module {
    @ParameterInfo(key: "pe") var pe: MLXArray

    init(maxFrames: Int, hiddenSize: Int) {
        _pe = ParameterInfo(wrappedValue: MLXArray.zeros([maxFrames, hiddenSize]), key: "pe")
        super.init()
    }
}

/// The full checkpoint: AR backbone + `lm_head`, plus the NAR auxiliary modules declared for
/// checkpoint coverage (`vae2llm`, `llm2vae`, `time_embedder`, `latent_pos_embed`) — exercised
/// starting Jalon 3.
public final class YuE2ForCausalLM: Module {
    /// Kept for NAR-side math that needs hyperparameters directly (`timestep_shift`,
    /// `max_latent_frames`) rather than through a submodule (Jalon 3, `NARModules.swift`).
    public let config: YuE2Config
    @ModuleInfo(key: "model") var model: Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "vae2llm") var vae2llm: Linear
    @ModuleInfo(key: "llm2vae") var llm2vae: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: TimeEmbedder
    @ModuleInfo(key: "latent_pos_embed") var latentPosEmbed: LatentPosEmbed

    init(config: YuE2Config) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Backbone(config: config), key: "model")
        _lmHead = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false), key: "lm_head")
        _vae2llm = ModuleInfo(wrappedValue: Linear(config.latentDim, config.hiddenSize, bias: true), key: "vae2llm")
        _llm2vae = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.latentDim, bias: true), key: "llm2vae")
        _timeEmbedder = ModuleInfo(wrappedValue: TimeEmbedder(hiddenSize: config.hiddenSize), key: "time_embedder")
        _latentPosEmbed = ModuleInfo(wrappedValue: LatentPosEmbed(maxFrames: config.maxLatentFrames, hiddenSize: config.hiddenSize), key: "latent_pos_embed")
        super.init()
    }

    /// `[B, T, V]` if `keepLast` is false, else `[B, 1, V]` (only the last position, for
    /// incremental generation — avoids running `lm_head` over the whole sequence every step).
    /// `positions`, when given, overrides cache-offset-derived RoPE positions with arbitrary
    /// absolute per-token ones (T-2.4). Full (unrestricted) `lm_head`: parity/bisection callers
    /// (`yue2 parity lm`, T-2.9) want the exact reference computation, not O1's restriction.
    public func forwardAR(
        tokens: MLXArray, cache: KVCache?, keepLast: Bool, positions: MLXArray? = nil,
        taps: ((Int, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        var hidden = model.forwardAR(tokens: tokens, cache: cache, positions: positions, taps: taps)
        if keepLast {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        return lmHead(hidden)
    }

    /// A `RestrictedHead` (O1, T-2.6) over this model's `lm_head`, for callers outside
    /// `YuE2Core` (e.g. the CLI) that cannot see `lmHead` itself.
    public func makeRestrictedHead(bounds: LogitsBounds) -> RestrictedHead {
        RestrictedHead(lmHead: lmHead, bounds: bounds)
    }
}
