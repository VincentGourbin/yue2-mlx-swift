// ModelCatalog.swift - HuggingFace repos, expected files and licensing for YuE2 checkpoints
// Copyright 2026 Vincent Gourbin

import Foundation

/// Licence attached to a downloadable checkpoint.
public struct YuE2License: Sendable, Equatable {
    public let id: String
    public let name: String
    public let url: String
    public let allowsCommercialUse: Bool
    public let summary: String

    /// Every YuE2 checkpoint (LM and both VAEs) ships under this licence.
    public static let ccByNc4 = YuE2License(
        id: "cc-by-nc-4.0",
        name: "Creative Commons Attribution-NonCommercial 4.0",
        url: "https://creativecommons.org/licenses/by-nc/4.0/",
        allowsCommercialUse: false,
        summary: "Free to use, share and adapt with attribution; commercial use is not permitted."
    )
}

/// A downloadable YuE2 checkpoint and the files it publishes on HuggingFace.
public enum YuE2Model: String, CaseIterable, Sendable {
    case lm = "YuE2-3B"
    case vae = "YuE2-Vae"
    case vaeLegacy = "YuE2-Vae-legacy"

    /// HuggingFace repository id (`m-a-p/…`).
    public var repoID: String {
        switch self {
        case .lm: return "m-a-p/YuE2-3B"
        case .vae: return "m-a-p/YuE2-Vae"
        case .vaeLegacy: return "m-a-p/YuE2-Vae-legacy"
        }
    }

    /// Files expected under `$YUE2_MODELS_DIR/<directoryName>`, relative to the repo root.
    public var files: [String] {
        switch self {
        case .lm:
            return [
                "config.json", "generation_config.json", "yue2_generation_config.json",
                "weights_manifest.json", "model.safetensors", "qwen.tiktoken",
                "examples/tonight-awake.json",
            ]
        case .vae, .vaeLegacy:
            return ["config.json", "model.safetensors", "weights_manifest.json"]
        }
    }

    public var license: YuE2License { .ccByNc4 }

    /// Destination directory name under `$YUE2_MODELS_DIR` (identical to the repo's short name).
    public var directoryName: String { rawValue }
}

/// The real byte-pair tokenizer (`tokenizer.json`) is not published by `m-a-p/YuE2-3B`;
/// it is Qwen2.5's, fetched separately and converted in T-1.5.
public enum YuE2TokenizerSource {
    public static let repoID = "Qwen/Qwen2.5-0.5B"
    public static let file = "tokenizer.json"
    /// Where the raw download lands before T-1.5 converts it in place.
    public static let destination = "YuE2-3B/tokenizer.json.qwen25"
}
