// SongResult.swift - a completed generation: audio, latents, timing, saved artifacts
// Copyright 2026 Vincent Gourbin

import CryptoKit
import Foundation
import MLX

/// Per-stage timings for one `YuE2Pipeline.generate` call. Mirrors `pipeline.py`'s `timing`
/// dict, minus the one-time model-load timing (owned by the CLI/caller, not the pipeline).
public struct SongTiming: Codable, Equatable, Sendable {
    public let abc: GenerationTiming?
    public let semantic: GenerationTiming
    public let narSeconds: Double
    public let vaeSeconds: Double
    public let e2eSeconds: Double

    enum CodingKeys: String, CodingKey {
        case abc, semantic
        case narSeconds = "nar_seconds"
        case vaeSeconds = "vae_seconds"
        case e2eSeconds = "e2e_seconds"
    }

    public init(abc: GenerationTiming?, semantic: GenerationTiming, narSeconds: Double, vaeSeconds: Double, e2eSeconds: Double) {
        self.abc = abc
        self.semantic = semantic
        self.narSeconds = narSeconds
        self.vaeSeconds = vaeSeconds
        self.e2eSeconds = e2eSeconds
    }
}

/// One complete `YuE2Pipeline.generate` call's result. Mirrors `pipeline.py`'s `SongResult`.
public struct SongResult {
    /// `[1, N, 2]`, float32, NLC (the VAE's own output layout), unclamped.
    public let audio: MLXArray
    public let sampleRate: Int
    public let semantic: SemanticResult
    /// `[T, 64]`, float32.
    public let latents: MLXArray
    public let config: GenerationConfig
    public let timing: SongTiming

    public var truncated: (abc: Bool, semantic: Bool) {
        (semantic.plan.truncated, semantic.truncated)
    }

    /// For callers outside `YuE2Core` (e.g. `yue2 synthesize`, T-3.6) assembling a result from a
    /// resumed run rather than `YuE2Pipeline.generate`.
    public init(
        audio: MLXArray, sampleRate: Int, semantic: SemanticResult, latents: MLXArray,
        config: GenerationConfig, timing: SongTiming
    ) {
        self.audio = audio
        self.sampleRate = sampleRate
        self.semantic = semantic
        self.latents = latents
        self.config = config
        self.timing = timing
    }

    /// `audio` is NLC; `AudioExporter` wants channels-first (matches `yue2 decode`'s convention).
    public func save(to url: URL, format: AudioExporter.SampleFormat = .int16) throws {
        try AudioExporter.exportToWAV(
            audio: audio.transposed(0, 2, 1), url: url, sampleRate: sampleRate, format: format)
    }

    /// Writes every artifact needed to reproduce or resume this run: the plan (`score.abc`,
    /// `abc_tokens.npy`, `prefix.npy`, `plan.json`, `plan_manifest.json`), `audio.wav` (WAV, not
    /// FLAC — no FLAC encoder in this port, plan §1.6), `semantic.npy`, `latent.npy`,
    /// `request.json`, `config.json`, and a `result.json` summary with a SHA-256 per artifact.
    public func saveArtifacts(to directory: URL, format: AudioExporter.SampleFormat = .int16) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try semantic.plan.save(to: directory)
        try save(to: directory.appendingPathComponent("audio.wav"), format: format)
        try NumpyIO.writeInt32(semantic.tokens.map(Int32.init), url: directory.appendingPathComponent("semantic.npy"))
        try NumpyIO.writeFloat32(
            latents.asArray(Float.self), shape: latents.shape, url: directory.appendingPathComponent("latent.npy"))

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        try encoder.encode(semantic.plan.request).write(to: directory.appendingPathComponent("request.json"))
        try encoder.encode(config).write(to: directory.appendingPathComponent("config.json"))

        let manifest = ResultManifest(
            status: "complete",
            truncated: TruncatedFlags(abc: truncated.abc, semantic: truncated.semantic),
            sampleRate: sampleRate,
            audioSeconds: Double(audio.dim(1)) / Double(sampleRate),
            timing: timing,
            artifacts: try Self.collectHashes(in: directory))
        try encoder.encode(manifest).write(to: directory.appendingPathComponent("result.json"))
    }

    private static func collectHashes(in directory: URL) throws -> [String: String] {
        var hashes: [String: String] = [:]
        for name in try FileManager.default.contentsOfDirectory(atPath: directory.path) {
            let file = directory.appendingPathComponent(name)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory), !isDirectory.boolValue
            else { continue }
            let digest = SHA256.hash(data: try Data(contentsOf: file))
            hashes[name] = digest.map { String(format: "%02x", $0) }.joined()
        }
        return hashes
    }
}

private struct TruncatedFlags: Codable {
    let abc: Bool
    let semantic: Bool
}

private struct ResultManifest: Codable {
    let status: String
    let truncated: TruncatedFlags
    let sampleRate: Int
    let audioSeconds: Double
    let timing: SongTiming
    let artifacts: [String: String]

    enum CodingKeys: String, CodingKey {
        case status, truncated
        case sampleRate = "sample_rate"
        case audioSeconds = "audio_seconds"
        case timing, artifacts
    }
}
