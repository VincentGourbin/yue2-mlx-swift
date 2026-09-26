// GPUGateAndCheckpointTests.swift - YuE2GPUGate, per-layer eval, NARCheckpoint resume (Q7)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a - b)).item(Float.self)
}

@Suite("GPUGateAndCheckpoint", .serialized)
struct GPUGateAndCheckpointTests {
    @Test func gateBlocksWhileSuspendedAndReleasesOnResume() async {
        let gate = YuE2GPUGate()
        #expect(gate.wait())  // open by default: returns at once
        gate.suspend()
        #expect(gate.isSuspended)
        let released = Task.detached { gate.wait() }
        try? await Task.sleep(for: .milliseconds(250))
        #expect(!Task.detached { true }.isCancelled)  // the waiter is still parked (no result yet)
        gate.resume()
        #expect(await released.value)
        #expect(!gate.isSuspended)
    }

    @Test func gateWaitReturnsFalseWhenCancelledWhileSuspended() {
        let gate = YuE2GPUGate()
        gate.suspend()
        var polls = 0
        let result = gate.wait { polls += 1; return polls >= 2 }
        #expect(!result)
        #expect(polls >= 2)
        gate.resume()
    }

    /// Per-layer eval changes when things are materialized, never what is computed.
    @Test func perLayerEvalLeavesVelocityIdentical() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let nar = CachedNAR(model: model, chunk: NARChunk(arTokens: [2, 3, 4, 5], noise: noise))
        let saved = YuE2ExecutionPolicy.evalPerLayer
        defer { YuE2ExecutionPolicy.evalPerLayer = saved }

        YuE2ExecutionPolicy.evalPerLayer = false
        let once = nar.velocity(state: noise, rawT: -2.3)
        YuE2ExecutionPolicy.evalPerLayer = true
        let perLayer = nar.velocity(state: noise, rawT: -2.3)
        #expect(maxAbsDiff(once, perLayer) == 0)
        let expected = try #require(raw["velocity_raw-2.3"])
        #expect(maxAbsDiff(perLayer, expected) <= 1e-4)
    }

    /// Solving 4 steps, or 2 steps + resume from the step-2 checkpoint, is the same trajectory
    /// bit for bit — through `Synthesizer` (the API the app uses), with a save/load round trip.
    @Test func resumeFromCheckpointMatchesUninterruptedSolve() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let config = try GenerationConfig(odeSteps: 4)
        let synthesizer = Synthesizer(model: model, config: config)
        let prefix = [2, 3]
        let codec = (0..<noise.dim(0)).map { $0 % 4 }  // injected noise must be [codec.count, 64]

        let full = try synthesizer.synthesize(prefix: prefix, codec: codec, seed: 7, noise: noise)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        var seen: [Int] = []
        // "Crash" right after step 2 was checkpointed: the cancel closure fires at the start of
        // step 3 and the solve ends with `.cancelled`, exactly like an interrupted stage.
        #expect(throws: YuE2Error.self) {
            try synthesizer.synthesize(
                prefix: prefix, codec: codec, seed: 7, noise: noise,
                onStep: { checkpoint in
                    seen.append(checkpoint.step)
                    if checkpoint.step == 2 { try? checkpoint.save(to: dir) }
                },
                cancel: { seen.contains(2) })
        }
        #expect(seen == [1, 2])

        let checkpoint = try #require(try NARCheckpoint.load(from: dir))
        #expect(checkpoint.step == 2 && checkpoint.steps == 4 && checkpoint.chunkIndex == 0)
        var resumedSteps: [Int] = []
        let resumed = try synthesizer.synthesize(
            prefix: prefix, codec: codec, seed: 7, noise: noise,
            resume: checkpoint, onStep: { resumedSteps.append($0.step) })
        #expect(resumedSteps == [3, 4])
        #expect(maxAbsDiff(full, resumed) == 0)

        NARCheckpoint.clear(in: dir)
        #expect(try NARCheckpoint.load(from: dir) == nil)
    }

    @Test func resumeRejectsAMismatchedCheckpoint() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let synthesizer = Synthesizer(model: model, config: try GenerationConfig(odeSteps: 4))
        let codec = (0..<noise.dim(0)).map { $0 % 4 }
        let wrongSteps = NARCheckpoint(chunkIndex: 0, step: 1, steps: 8, state: noise)
        #expect(throws: YuE2Error.self) {
            try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise, resume: wrongSteps)
        }
        let wrongShape = NARCheckpoint(chunkIndex: 0, step: 1, steps: 4, state: MLXArray.zeros([noise.dim(0) + 1, 64]))
        #expect(throws: YuE2Error.self) {
            try synthesizer.synthesize(prefix: [2, 3], codec: codec, seed: 7, noise: noise, resume: wrongShape)
        }
    }
}
