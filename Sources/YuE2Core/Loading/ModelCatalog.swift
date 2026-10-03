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
    /// SheetSage2 transcription (audio → ABC, plan/14-sheetsage2.md): adapters + decoder. Gated.
    case sheetsage2 = "SheetSage2"
    /// MERT-v2-FullSong, SheetSage2's encoder parent; the adapters are merged into it at load. Gated.
    case mert2FullSong = "MERT-v2-FullSong"

    /// HuggingFace repository id (`m-a-p/…`).
    public var repoID: String {
        switch self {
        case .lm: return "m-a-p/YuE2-3B"
        case .vae: return "m-a-p/YuE2-Vae"
        case .vaeLegacy: return "m-a-p/YuE2-Vae-legacy"
        case .sheetsage2: return "m-a-p/SheetSage2"
        case .mert2FullSong: return "m-a-p/MERT-v2-FullSong"
        }
    }

    /// Git revision files are fetched from. SheetSage2's adapters only fit the MERT parent
    /// revision its `config.json` pins (`base_model_revision`); MERT's `main` has moved since.
    public var revision: String {
        switch self {
        case .sheetsage2: return "398b22834dac7dd05e09b9c4e40a39fc479ec502"
        case .mert2FullSong: return "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
        default: return "main"
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
        case .sheetsage2, .mert2FullSong:
            return ["config.json", "model.safetensors"]
        }
    }

    public var license: YuE2License { .ccByNc4 }

    /// Destination directory name under `$YUE2_MODELS_DIR` (identical to the repo's short name).
    public var directoryName: String { rawValue }
}

/// A prequantized pack published as a derivative of `m-a-p/YuE2-3B` (docs/Weights.md): one folder
/// per pack in a single Hugging Face repository, landing locally at
/// `$YUE2_MODELS_DIR/YuE2-3B/mlx-prequantized/<pack>/model.safetensors` — exactly where
/// `LMWeightLoader.load` looks for the matching preset, so a downloaded pack replaces the
/// first-use quantization (7.3 GB of bf16 loaded, minutes, 8-14 GB peak) with a plain fetch.
public enum YuE2Pack: String, CaseIterable, Sendable {
    /// AR path, embeddings and head in 4 bits, NAR path in 8 bits — `4bit-fast`, `4bit-lean`.
    case int4MixedHead = "int4-mixed-head"
    /// Everything in 8 bits — `8bit-fast`, `8bit-lean`.
    case qint8AllHead = "qint8-all-head"
    /// AR path, embeddings and head in 4 bits, NAR bf16 — the fastest on a Mac.
    case int4Head = "int4-head"

    public static let repoID = "VincentGOURBIN/yue2-mlx-packs"

    public var quantization: YuE2Quantization {
        switch self {
        case .int4MixedHead: return .int4Mixed
        case .qint8AllHead: return .qint8All
        case .int4Head: return .int4
        }
    }

    public var quantizeHead: Bool { true }

    /// Files in the repository, relative to its root.
    public var files: [String] { ["\(rawValue)/model.safetensors", "\(rawValue)/model.safetensors.sha256"] }

    /// Destination directory, relative to `$YUE2_MODELS_DIR`.
    public var localDirectory: String { "YuE2-3B/mlx-prequantized/\(rawValue)" }

    /// Download size, for progress and free-space checks.
    public var approximateBytes: Int64 {
        switch self {
        case .int4MixedHead: return 2_700_000_000
        case .qint8AllHead: return 3_750_000_000
        case .int4Head: return 4_700_000_000
        }
    }

    public var license: YuE2License { .ccByNc4 }

    /// The published pack that serves this preset, if any (`.none` and the measurement-only
    /// presets have none).
    public static func matching(_ quantization: YuE2Quantization, quantizeHead: Bool) -> YuE2Pack? {
        guard quantizeHead else { return nil }
        return allCases.first { $0.quantization == quantization }
    }
}

/// Where checkpoints live when nothing overrides it. macOS reads `$YUE2_MODELS_DIR` (a sandboxed
/// iOS app has no shell environment to read a path from, so it always gets the app container's
/// own `Caches/models` — never a shared/system location).
public enum YuE2ModelsDirectory {
    public static func resolveDefault() -> URL? {
        #if os(iOS)
        guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            return nil
        }
        return caches.appendingPathComponent("models")
        #else
        guard let env = ProcessInfo.processInfo.environment["YUE2_MODELS_DIR"], !env.isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: env)
        #endif
    }
}

/// The real byte-pair tokenizer (`tokenizer.json`) is not published by `m-a-p/YuE2-3B`;
/// it is Qwen2.5's, fetched separately and converted in T-1.5.
public enum YuE2TokenizerSource {
    public static let repoID = "Qwen/Qwen2.5-0.5B"
    public static let file = "tokenizer.json"
    /// Where the raw download lands before T-1.5 converts it in place.
    public static let destination = "YuE2-3B/tokenizer.json.qwen25"
}
