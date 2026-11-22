// ODEMethodTests.swift - Adams–Bashforth 2 NAR integrator (ODEMethod.ab2), weight-free (tiny model)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a - b)).item(Float.self)
}

@Suite("ODEMethod", .serialized)
struct ODEMethodTests {
    @Test func configAcceptsAB2AndRejectsUnknownMethods() throws {
        #expect(try GenerationConfig(odeMethod: "ab2").method == .ab2)
        #expect(try GenerationConfig().method == .midpoint)
        #expect(throws: YuE2Error.self) { try GenerationConfig(odeMethod: "euler") }
        let decoded = try JSONDecoder().decode(GenerationConfig.self, from: Data(#"{"ode_method":"ab2","ode_steps":16}"#.utf8))
        #expect(decoded.method == .ab2 && decoded.odeSteps == 16)
    }

    /// Every reference profile ships AB2 (judged by ear in bf16 and fp16 packed, 2026-11-21).
    @Test func referenceProfilesUseAB2() throws {
        for profile in YuE2ReferenceProfile.all {
            #expect(profile.odeMethod == .ab2, "\(profile.id)")
            let config = profile.generationConfig(try GenerationConfig())
            #expect(config.method == .ab2 && config.odeSteps == (profile.odeSteps ?? 32), "\(profile.id)")
        }
    }

    /// `.draft` is AB2 at 8 steps whatever the base or the profile pins; `.full` is untouched.
    @Test func draftQualityPinsAB2AtEightSteps() throws {
        let base = try GenerationConfig(odeSteps: 24)
        for profile in YuE2ReferenceProfile.all {
            let draft = profile.generationConfig(base, quality: .draft)
            #expect(draft.method == .ab2 && draft.odeSteps == YuE2ReferenceProfile.draftODESteps, "\(profile.id)")
            #expect(draft.method.evaluations(steps: draft.odeSteps) == 9)
            #expect(profile.generationConfig(base, quality: .full) == profile.generationConfig(base), "\(profile.id)")
            #expect(draft.abc == base.abc && draft.semantic == base.semantic && draft.context == base.context)
        }
    }

    @Test func evaluationCounts() {
        #expect(ODEMethod.midpoint.evaluations(steps: 32) == 64)
        #expect(ODEMethod.ab2.evaluations(steps: 32) == 33)
        #expect(ODEMethod.ab2.evaluations(steps: 32, startStep: 30) == 3)
        #expect(ODEMethod.ab2.evaluations(steps: 4, startStep: 4) == 0)
    }

    /// AB2 costs steps + 1 network calls; a `.keep` edit falls back to midpoint (2 per step).
    @Test func ab2CallsTheNetworkStepsPlusOneTimes() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let nar = CachedNAR(model: model, chunk: NARChunk(arTokens: [2, 3, 4, 5], noise: noise))
        var calls = 0
        _ = try nar.solve(steps: 6, onVelocityCall: { calls += 1 }, method: .ab2)
        #expect(calls == 7)
        calls = 0
        _ = try nar.solve(
            steps: 6, onVelocityCall: { calls += 1 },
            keep: (frames: 2, latents: noise), method: .ab2)
        #expect(calls == 12)
    }

    /// Both integrators are second order: against a fine midpoint solve, AB2's error shrinks
    /// with the step count, and at 32 steps it sits in the same range as midpoint at 16 (same
    /// network cost). Tiny model in fp32: this checks the integrator, not audio quality.
    @Test func ab2ConvergesToTheSameSolution() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let nar = CachedNAR(model: model, chunk: NARChunk(arTokens: [2, 3, 4, 5], noise: noise))
        let reference = try nar.solve(steps: 128)
        let scale = MLX.max(MLX.abs(reference)).item(Float.self)
        let ab2At8 = maxAbsDiff(try nar.solve(steps: 8, method: .ab2), reference) / scale
        let ab2At32 = maxAbsDiff(try nar.solve(steps: 32, method: .ab2), reference) / scale
        let midAt16 = maxAbsDiff(try nar.solve(steps: 16), reference) / scale
        #expect(ab2At32 < ab2At8)
        #expect(ab2At32 < 0.05)
        #expect(ab2At32 < midAt16 * 4, "ab2@32 \(ab2At32) vs midpoint@16 \(midAt16)")
    }

    /// An AB2 solve interrupted after a checkpoint resumes bit for bit when the checkpoint
    /// carries the previous velocity (saved and loaded as `nar-velocity-<chunk>-<step>.npy`).
    @Test func ab2ResumesBitExactFromCheckpoint() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let synthesizer = Synthesizer(model: model, config: try GenerationConfig(odeSteps: 6, odeMethod: "ab2"))
        let prefix = [2, 3]
        let codec = (0..<noise.dim(0)).map { $0 % 4 }
        let full = try synthesizer.synthesize(prefix: prefix, codec: codec, seed: 7, noise: noise)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        var seen: [Int] = []
        #expect(throws: YuE2Error.self) {
            try synthesizer.synthesize(
                prefix: prefix, codec: codec, seed: 7, noise: noise,
                onStep: { checkpoint in
                    seen.append(checkpoint.step)
                    if checkpoint.step == 3 { try? checkpoint.save(to: dir) }
                },
                cancel: { seen.contains(3) })
        }
        let checkpoint = try #require(try NARCheckpoint.load(from: dir))
        #expect(checkpoint.previousVelocity != nil)
        let resumed = try synthesizer.synthesize(
            prefix: prefix, codec: codec, seed: 7, noise: noise, resume: checkpoint)
        #expect(maxAbsDiff(full, resumed) == 0)

        // Clearing removes every checkpoint file.
        NARCheckpoint.clear(in: dir)
        let left = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        #expect(left.filter { $0.hasPrefix("nar-") }.isEmpty, "\(left)")
    }

    /// The JSON is the only commit point: arrays of a step whose save was cut before the JSON
    /// are ignored, older step files are removed, and pre-2026-11-21 checkpoints still load.
    @Test func checkpointCommitsOnlyThroughTheJSON() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let a = MLXArray((0..<12).map(Float.init), [6, 2])
        let v = a * 2
        try NARCheckpoint(chunkIndex: 0, step: 3, steps: 6, state: a, previousVelocity: v).save(to: dir)
        try NARCheckpoint(chunkIndex: 0, step: 4, steps: 6, state: a + 1, previousVelocity: v + 1).save(to: dir)
        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".npy") }.sorted()
        #expect(names.count == 2 && names[0].hasPrefix("nar-state-0-4-") && names[1].hasPrefix("nar-velocity-0-4-"), "\(names)")

        // A save of step 5 killed after its arrays but before its JSON.
        try NumpyIO.writeFloat32([Float](repeating: 9, count: 12), shape: [6, 2], url: dir.appendingPathComponent("nar-state-0-5.npy"))
        let loaded = try #require(try NARCheckpoint.load(from: dir))
        #expect(loaded.step == 4)
        #expect(maxAbsDiff(loaded.state, a + 1) == 0)
        #expect(maxAbsDiff(try #require(loaded.previousVelocity), v + 1) == 0)

        // Legacy layout: fixed `nar-state.npy`, meta without file names, no velocity.
        NARCheckpoint.clear(in: dir)
        try NumpyIO.writeFloat32(a.asArray(Float.self), shape: [6, 2], url: dir.appendingPathComponent("nar-state.npy"))
        try Data(#"{"chunkIndex":0,"step":2,"steps":6,"frames":6}"#.utf8).write(to: dir.appendingPathComponent("nar-checkpoint.json"))
        let legacy = try #require(try NARCheckpoint.load(from: dir))
        #expect(legacy.step == 2 && legacy.previousVelocity == nil && legacy.method == nil && maxAbsDiff(legacy.state, a) == 0)
    }

    /// A checkpoint resumes only under the integrator that wrote it (a pre-method checkpoint,
    /// `method == nil`, may be continued by either: valid, not bit-exact).
    @Test func resumeRejectsAnotherIntegrator() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let prefix = [2, 3]
        let codec = (0..<noise.dim(0)).map { $0 % 4 }
        let midpoint = Synthesizer(model: model, config: try GenerationConfig(odeSteps: 6))
        let ab2 = Synthesizer(model: model, config: try GenerationConfig(odeSteps: 6, odeMethod: "ab2"))
        var saved: NARCheckpoint?
        _ = try? ab2.synthesize(
            prefix: prefix, codec: codec, seed: 7, noise: noise,
            onStep: { if $0.step == 3 { saved = $0 } }, cancel: { saved != nil })
        let checkpoint = try #require(saved)
        #expect(checkpoint.method == .ab2)
        #expect(throws: YuE2Error.self) {
            try midpoint.synthesize(prefix: prefix, codec: codec, seed: 7, noise: noise, resume: checkpoint)
        }
        // A draft checkpoint (other steps, same integrator) is refused too.
        let draft = Synthesizer(model: model, config: try GenerationConfig(odeSteps: 4, odeMethod: "ab2"))
        #expect(throws: YuE2Error.self) {
            try draft.synthesize(prefix: prefix, codec: codec, seed: 7, noise: noise, resume: checkpoint)
        }
        let legacy = NARCheckpoint(chunkIndex: 0, step: 3, steps: 6, state: checkpoint.state)
        _ = try ab2.synthesize(prefix: prefix, codec: codec, seed: 7, noise: noise, resume: legacy)
        _ = try midpoint.synthesize(prefix: prefix, codec: codec, seed: 7, noise: noise, resume: legacy)
    }
}
