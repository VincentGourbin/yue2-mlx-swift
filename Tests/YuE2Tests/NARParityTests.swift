// NARParityTests.swift - NAR-side modules parity against parity/tiny_lm.safetensors (T-3.x)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.max(MLX.abs(a - b)).item(Float.self)
}

@Suite("NARParity")
struct NARParityTests {
    @Test func timeEmbeddingMatchesFixture() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        for value in [20.0, 0.0, -2.3] {
            let shifted = model.shiftT(raw: value)
            let embedding = model.timeEmbedding(tShifted: shifted)
            let expected = try #require(raw["time_emb_raw\(value)"])
            #expect(maxAbsDiff(embedding, expected) <= 1e-4)
        }
    }

    /// T-6.2 (E2): the recomputed positional table (used when `pe` is omitted from an "-all"
    /// iPhone pack) must match the checkpoint's loaded buffer exactly, not just approximately.
    @Test func computedLatentPositionsMatchLoadedBuffer() throws {
        let model = try loadedTinyModel()
        let count = 5
        let indices = MLXArray((0..<count).map { Int32($0) })
        let fromBuffer = model.latentPositions(count: count)
        let computed = YuE2ForCausalLM.computeLatentPositions(
            indices: indices, hiddenSize: model.config.hiddenSize, dtype: model.vae2llm.weight.dtype)
        #expect(maxAbsDiff(fromBuffer, computed) <= 1e-5) // fp32 tiny model, pitfall #5 convention
    }

    @Test func songChunksSplitsAndOffsetsCodec() throws {
        let prefix = [2, 3]
        let codec = Array(0..<11)
        let noise = MLXArray((0..<(11 * 64)).map { Float($0) }, [11, 64])
        let chunks = try NARChunk.songChunks(prefix: prefix, codec: codec, seed: 0, context: 15, noise: noise)

        #expect(chunks.count == 3)
        #expect(chunks.map { $0.noise.dim(0) } == [5, 5, 1])

        let expectedChunk1Tokens = [2, 3] + (5..<10).map { $0 + YuE2Token.codecOffset } + [YuE2Token.musicEnd]
        #expect(chunks[1].arTokens == expectedChunk1Tokens)
        #expect(chunks[1].noise.asArray(Float.self) == noise[5..<10].asArray(Float.self))

        let expectedChunk2Tokens = [2, 3, 10 + YuE2Token.codecOffset, YuE2Token.musicEnd]
        #expect(chunks[2].arTokens == expectedChunk2Tokens)
    }

    @Test func songChunksRejectsCodecOutOfVocabulary() throws {
        #expect(throws: YuE2Error.self) {
            try NARChunk.songChunks(prefix: [2, 3], codec: [YuE2Token.codecSize], seed: 0)
        }
    }

    @Test func songChunksRejectsMismatchedInjectedNoise() throws {
        #expect(throws: YuE2Error.self) {
            try NARChunk.songChunks(prefix: [2, 3], codec: [0, 1, 2], seed: 0, noise: MLXArray.zeros([2, 64]))
        }
    }

    @Test func cachedNARComputesPrefixAndChunkLengths() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let chunk = NARChunk(arTokens: [2, 3, 4, 5], noise: noise)
        let nar = CachedNAR(model: model, chunk: chunk)
        #expect(nar.arLength == 4)
        #expect(nar.narLength == noise.dim(0) + 2)
    }

    @Test func velocityMatchesFixtureForThreeTimesteps() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let chunk = NARChunk(arTokens: [2, 3, 4, 5], noise: noise)
        let nar = CachedNAR(model: model, chunk: chunk)
        for value in [20.0, 0.0, -2.3] {
            let velocity = nar.velocity(state: noise, rawT: value)
            let expected = try #require(raw["velocity_raw\(value)"])
            #expect(maxAbsDiff(velocity, expected) <= 1e-4)
        }
    }

    @Test func velocityBisectionMatchesFixture() throws {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["nar_noise"])
        let chunk = NARChunk(arTokens: [2, 3, 4, 5], noise: noise)
        let nar = CachedNAR(model: model, chunk: chunk)
        var taps: [String: MLXArray] = [:]
        _ = nar.velocity(state: noise, rawT: 20.0) { name, value in taps[name] = value }
        for name in ["after_vae2llm", "after_time", "after_pos", "after_layer_0", "after_layer_1"] {
            let got = try #require(taps[name])
            let expected = try #require(raw["nar.\(name)"])
            #expect(maxAbsDiff(got, expected) <= 1e-4)
        }
    }

    private func makeSolve4NAR() throws -> (nar: CachedNAR, fixture: [String: MLXArray]) {
        let raw = try loadArrays(url: tinyLMFixture)
        let model = try loadedTinyModel()
        let noise = try #require(raw["solve4_noise"])
        let chunk = NARChunk(arTokens: [2, 3, 4, 5], noise: noise)
        return (CachedNAR(model: model, chunk: chunk), raw)
    }

    @Test func solveFourStepsMatchesFixture() throws {
        let (nar, raw) = try makeSolve4NAR()
        let result = try nar.solve(steps: 4)
        let expected = try #require(raw["solve4_expected"])
        #expect(maxAbsDiff(result, expected) <= 1e-4)
    }

    @Test func solveCallsVelocityExactlyTwicePerStep() throws {
        let (nar, _) = try makeSolve4NAR()
        var count = 0
        _ = try nar.solve(steps: 4, onVelocityCall: { count += 1 })
        #expect(count == 8)
    }

    @Test func solveReportsProgressPerStep() throws {
        let (nar, _) = try makeSolve4NAR()
        var progress: [(done: Int, total: Int)] = []
        _ = try nar.solve(steps: 4, onProgress: { done, total in progress.append((done, total)) })
        #expect(progress.map(\.done) == [1, 2, 3, 4])
        #expect(progress.allSatisfy { $0.total == 4 })
    }

    @Test func solveCancelsPartwayThrough() throws {
        let (nar, _) = try makeSolve4NAR()
        var checks = 0
        #expect(throws: YuE2Error.self) {
            _ = try nar.solve(steps: 4, cancel: {
                checks += 1
                return checks > 2
            })
        }
        #expect(checks == 3)
    }

    private func makeSongSetup() throws -> (Synthesizer, YuE2ForCausalLM, [Int], [Int]) {
        let model = try loadedTinyModel()
        let synthesizer = Synthesizer(model: model, config: try GenerationConfig())
        return (synthesizer, model, [2, 3], Array(repeating: 1, count: 11))
    }

    @Test func synthesizeMatchesPerChunkSolve() throws {
        let noise = MLXArray((0..<(11 * 64)).map { Float($0 % 7) }, [11, 64])
        let (synthesizer, model, prefix, codec) = try makeSongSetup()
        let combined = try synthesizer.synthesize(
            prefix: prefix, codec: codec, seed: 42, steps: 2, context: 15, noise: noise
        )

        let chunks = try NARChunk.songChunks(prefix: prefix, codec: codec, seed: 42, context: 15, noise: noise)
        let expectedParts = try chunks.map { try CachedNAR(model: model, chunk: $0).solve(steps: 2) }
        let expected = concatenated(expectedParts, axis: 0)
        #expect(maxAbsDiff(combined, expected) <= 1e-4)
    }

    @Test func synthesizeReportsGlobalProgress() throws {
        let noise = MLXArray.zeros([11, 64])
        let (synthesizer, _, prefix, codec) = try makeSongSetup()
        var progress: [(done: Int, total: Int)] = []
        _ = try synthesizer.synthesize(
            prefix: prefix, codec: codec, seed: 42, steps: 2, context: 15, noise: noise,
            onProgress: { done, total in progress.append((done, total)) }
        )
        #expect(progress.map(\.done) == Array(1...6))
        #expect(progress.allSatisfy { $0.total == 6 })
    }

    @Test func synthesizeRejectsNonFiniteLatents() throws {
        var values = [Float](repeating: 0, count: 11 * 64)
        values[0] = .nan
        let noise = MLXArray(values, [11, 64])
        let (synthesizer, _, prefix, codec) = try makeSongSetup()
        #expect(throws: YuE2Error.self) {
            _ = try synthesizer.synthesize(prefix: prefix, codec: codec, seed: 42, steps: 2, context: 15, noise: noise)
        }
    }
}
