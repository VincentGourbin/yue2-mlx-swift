// NARCheckpoint.swift - per-step checkpoint of a NAR solve, so a lost step never costs the stage
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX

/// The ODE state after `step` of `steps` for chunk `chunkIndex` — 192 KB (1 500 × 64 fp16),
/// enough to resume `Synthesizer.synthesize(resume:)` with the same seed and settings from that
/// exact point (`CachedNAR.solve(initialState:startStep:)`). The engine keeps no history: the
/// callback receives the live `state` and whoever persists it owns the only copy.
public struct NARCheckpoint {
    public let chunkIndex: Int
    public let step: Int
    public let steps: Int
    public let state: MLXArray

    public init(chunkIndex: Int, step: Int, steps: Int, state: MLXArray) {
        self.chunkIndex = chunkIndex
        self.step = step
        self.steps = steps
        self.state = state
    }

    private struct Meta: Codable {
        var chunkIndex: Int
        var step: Int
        var steps: Int
        var frames: Int
    }

    static let metaName = "nar-checkpoint.json"
    static let stateName = "nar-state.npy"

    /// Writes `nar-state.npy` (fp32, exact for an fp16 state) then `nar-checkpoint.json`,
    /// atomically (temp file + replace), so a crash mid-write leaves the previous checkpoint.
    public func save(to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let values = state.asType(.float32).asArray(Float.self)
        let stateURL = directory.appendingPathComponent(Self.stateName)
        let temporary = directory.appendingPathComponent(".tmp-" + Self.stateName)
        try NumpyIO.writeFloat32(values, shape: state.shape, url: temporary)
        _ = try FileManager.default.replaceItemAt(stateURL, withItemAt: temporary)
        let meta = Meta(chunkIndex: chunkIndex, step: step, steps: steps, frames: state.dim(0))
        try JSONEncoder().encode(meta).write(to: directory.appendingPathComponent(Self.metaName), options: .atomic)
    }

    /// `nil` when `directory` holds no checkpoint.
    public static func load(from directory: URL) throws -> NARCheckpoint? {
        let metaURL = directory.appendingPathComponent(metaName)
        guard FileManager.default.fileExists(atPath: metaURL.path) else { return nil }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let state = try NumpyIO.readArray(directory.appendingPathComponent(stateName))
        guard state.ndim == 2, state.dim(0) == meta.frames else {
            throw YuE2Error.invalidRequest("nar-state.npy shape \(state.shape) does not match the checkpoint (\(meta.frames) frames)")
        }
        return NARCheckpoint(chunkIndex: meta.chunkIndex, step: meta.step, steps: meta.steps, state: state)
    }

    /// Removes a checkpoint once its stage completed.
    public static func clear(in directory: URL) {
        try? FileManager.default.removeItem(at: directory.appendingPathComponent(metaName))
        try? FileManager.default.removeItem(at: directory.appendingPathComponent(stateName))
    }
}
