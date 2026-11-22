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
    /// The last velocity an AB2 solve (`ODEMethod.ab2`) extrapolates from — `nil` for midpoint.
    /// Same size as `state`; resuming AB2 without it bootstraps with one midpoint step again
    /// (a valid solve, not the uninterrupted one bit for bit).
    public let previousVelocity: MLXArray?
    /// The integrator of the solve that wrote it; `nil` for checkpoints written before
    /// 2026-11-21 (midpoint), which any method may continue (a valid solve, not bit-exact).
    public let method: ODEMethod?

    public init(
        chunkIndex: Int, step: Int, steps: Int, state: MLXArray, previousVelocity: MLXArray? = nil,
        method: ODEMethod? = nil
    ) {
        self.chunkIndex = chunkIndex
        self.step = step
        self.steps = steps
        self.state = state
        self.previousVelocity = previousVelocity
        self.method = method
    }

    private struct Meta: Codable {
        var chunkIndex: Int
        var step: Int
        var steps: Int
        var frames: Int
        /// Files this checkpoint commits to (absent in checkpoints written before 2026-11-21,
        /// which used the fixed names `nar-state.npy` and no velocity).
        var stateFile: String?
        var velocityFile: String?
        var method: String?
    }

    static let metaName = "nar-checkpoint.json"
    static let legacyStateName = "nar-state.npy"
    static let legacyVelocityName = "nar-velocity.npy"

    /// Writes the save's own `nar-state-<chunk>-<step>-<id>.npy` (fp32, exact for an fp16 state) and,
    /// for AB2, `nar-velocity-<chunk>-<step>-<id>.npy`, then commits them with an atomic
    /// `nar-checkpoint.json` naming them, then removes the older step files. The JSON is the
    /// only commit point: a kill at any moment leaves either the previous checkpoint or this
    /// one, never a state and a meta (or velocity) from different steps.
    public func save(to directory: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        // Unique per save: a later solve reaching the same chunk and step in this directory
        // never overwrites the arrays the committed JSON still names.
        let tag = "\(chunkIndex)-\(step)-\(UUID().uuidString.prefix(8).lowercased())"
        let stateFile = "nar-state-\(tag).npy"
        try Self.writeAtomically(state, to: directory.appendingPathComponent(stateFile))
        var velocityFile: String?
        if let previousVelocity {
            velocityFile = "nar-velocity-\(tag).npy"
            try Self.writeAtomically(previousVelocity, to: directory.appendingPathComponent(velocityFile!))
        }
        let meta = Meta(
            chunkIndex: chunkIndex, step: step, steps: steps, frames: state.dim(0),
            stateFile: stateFile, velocityFile: velocityFile, method: method?.rawValue)
        try JSONEncoder().encode(meta).write(to: directory.appendingPathComponent(Self.metaName), options: .atomic)
        Self.removeArrays(in: directory, keeping: [stateFile, velocityFile].compactMap { $0 })
    }

    /// `nil` when `directory` holds no checkpoint.
    public static func load(from directory: URL) throws -> NARCheckpoint? {
        let metaURL = directory.appendingPathComponent(metaName)
        guard FileManager.default.fileExists(atPath: metaURL.path) else { return nil }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: metaURL))
        let stateFile = meta.stateFile ?? legacyStateName
        let state = try NumpyIO.readArray(directory.appendingPathComponent(stateFile))
        guard state.ndim == 2, state.dim(0) == meta.frames else {
            throw YuE2Error.invalidRequest("\(stateFile) shape \(state.shape) does not match the checkpoint (\(meta.frames) frames)")
        }
        var velocity: MLXArray?
        if let velocityFile = meta.velocityFile {
            let loaded = try NumpyIO.readArray(directory.appendingPathComponent(velocityFile))
            guard loaded.shape == state.shape else {
                throw YuE2Error.invalidRequest("\(velocityFile) shape \(loaded.shape) does not match \(stateFile) \(state.shape)")
            }
            velocity = loaded
        }
        return NARCheckpoint(
            chunkIndex: meta.chunkIndex, step: meta.step, steps: meta.steps, state: state, previousVelocity: velocity,
            method: meta.method.flatMap(ODEMethod.init(rawValue:)))
    }

    /// Removes a checkpoint once its stage completed.
    public static func clear(in directory: URL) {
        try? FileManager.default.removeItem(at: directory.appendingPathComponent(metaName))
        removeArrays(in: directory, keeping: [])
    }

    private static func writeAtomically(_ array: MLXArray, to url: URL) throws {
        let temporary = url.deletingLastPathComponent().appendingPathComponent(".tmp-" + url.lastPathComponent)
        try NumpyIO.writeFloat32(array.asType(.float32).asArray(Float.self), shape: array.shape, url: temporary)
        _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary)
    }

    /// Checkpoint arrays (current, legacy and leftover temporaries) other than `keeping`.
    private static func removeArrays(in directory: URL, keeping: [String]) {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        for name in names where !keeping.contains(name) {
            let bare = name.hasPrefix(".tmp-") ? String(name.dropFirst(5)) : name
            guard bare.hasSuffix(".npy"), bare.hasPrefix("nar-state") || bare.hasPrefix("nar-velocity") else { continue }
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
    }
}
