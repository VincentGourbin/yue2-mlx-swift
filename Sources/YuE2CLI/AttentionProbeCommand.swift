// AttentionProbeCommand.swift - `yue2 attention-probe` (EXPERIMENTAL, hidden): per-head prompt attention of a generated song
// Copyright 2026 Vincent Gourbin

import ArgumentParser
import Foundation
import MLX
import YuE2Core

struct AttentionProbeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "attention-probe",
        abstract: "EXPERIMENTAL: attention mass of every AR head on prompt segments (score bars, lyric words) per semantic frame.",
        shouldDisplay: false
    )

    @OptionGroup var modelOptions: ModelOptions

    @Option(name: .long, help: "A `yue2 generate --out` directory (prefix.npy, semantic.npy).")
    var song: String

    @Option(name: .long, help: "int32 npy, one segment id per prefix token (-1: none).")
    var segments: String

    @Option(name: .long, help: "Number of segments.")
    var segmentCount: Int

    @Option(name: .long, help: "Output float32 npy [layers, heads, frames, segments].")
    var out: String

    func run() async throws {
        let modelsDir = try resolveModelsDir(modelOptions.modelsDir)
        let dir = URL(fileURLWithPath: song)
        let prefix = try NumpyIO.readInt32(dir.appendingPathComponent("prefix.npy"))
        let codes = try NumpyIO.readInt32(dir.appendingPathComponent("semantic.npy"))
        let segmentIDs = try NumpyIO.readInt32(URL(fileURLWithPath: segments))
        YuE2MemoryManager.configure(for: .ar)
        let session = try await ModelSession.load(
            modelsDir: modelsDir, quant: modelOptions.quant, quantizeHead: modelOptions.quantHead,
            precision: modelOptions.precision, residency: [.ar])
        let tokens = prefix + codes.map { $0 + Int32(YuE2Token.codecOffset) }
        let started = Date()
        let config = session.model.config
        let flat = try session.model.prefixAttention(
            tokens: tokens, prefixLength: prefix.count, segments: segmentIDs, segmentCount: segmentCount)
        let mass = flat.reshaped(config.numHiddenLayers, config.numAttentionHeads, flat.dim(1), flat.dim(2))
        try NumpyIO.writeFloat32(mass.asArray(Float.self), shape: mass.shape, url: URL(fileURLWithPath: out))
        FileHandle.standardError.write(Data(String(format: "attention probe: %@ in %.1f s\n", "\(mass.shape)", Date().timeIntervalSince(started)).utf8))
    }
}
