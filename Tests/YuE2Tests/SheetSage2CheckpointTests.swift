// SheetSage2CheckpointTests.swift - GPU-gate checkpoints, cancellation and progress (ASK Q11), tiny model
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import SheetSage2Core

private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

private func tinyTranscriber(chunked: Bool) throws -> (SheetSage2Transcriber, MLXArray) {
    let (arrays, metadata) = try loadArraysAndMetadata(url: repoRoot.appendingPathComponent("parity/tiny_sheetsage2.safetensors"))
    let config = try JSONDecoder().decode(SheetSage2Config.self, from: Data(try #require(metadata["config"]).utf8))
    let model = SheetSage2Model(config: config)
    var weights = [String: MLXArray]()
    for (key, value) in arrays where key.hasPrefix("weights.") { weights[String(key.dropFirst(8))] = value }
    try SheetSage2Weights.apply(SheetSage2Weights.sanitize(weights), to: model, dtype: .float32)
    if chunked { model.chunkFrames = 16 }  // several chunks even on the tiny 48-frame window
    return (SheetSage2Transcriber(model: model), try #require(arrays["input_waveform"]))
}

private struct Stop: Error {}

@Suite("SheetSage2Checkpoints")
struct SheetSage2CheckpointTests {
    @Test(arguments: [false, true]) func encoderChecksBeforeEveryStageWithMonotonicFraction(chunked: Bool) throws {
        let (transcriber, waveform) = try tinyTranscriber(chunked: chunked)
        var fractions = [Double]()
        let memory = try transcriber.model.encode(waveform) { fractions.append($0) }
        #expect(MLX.arrayEqual(memory, try transcriber.model.encode(waveform)).item(Bool.self))
        // STFT + 3 ConvNeXt blocks + 2 Conformer blocks + projection; more calls when chunked.
        #expect(fractions.count >= 7)
        if chunked { #expect(fractions.count > 7) }
        #expect(zip(fractions, fractions.dropFirst()).allSatisfy { $0 <= $1 })
        #expect(fractions.allSatisfy { $0 >= 0 && $0 < 1 })
    }

    @Test func decoderChecksBeforeEveryStepAndPausingChangesNothing() throws {
        let (transcriber, waveform) = try tinyTranscriber(chunked: false)
        let memory = try transcriber.model.encode(waveform)
        var generator = SheetSage2Generator(model: transcriber.model)
        let reference = try generator.generate(memory: memory, stopTimeSeconds: nil)
        var steps = [Int]()
        let paused = try generator.generate(memory: memory, stopTimeSeconds: nil, checkpoint: { tokens in
            steps.append(tokens)
            if tokens % 7 == 0 { Thread.sleep(forTimeInterval: 0.01) }  // the host's gate held closed
        })
        #expect(paused == reference)
        // One call per step, before it: prefix length, then +1 per generated token.
        #expect(steps == Array(8 ..< 8 + steps.count))
        #expect(steps.count >= reference.count - 9 && steps.count <= reference.count - 8)
    }

    @Test func throwingStopsTheDecoderAtTheNextStep() throws {
        let (transcriber, waveform) = try tinyTranscriber(chunked: false)
        let memory = try transcriber.model.encode(waveform)
        var generator = SheetSage2Generator(model: transcriber.model)
        var last = 0
        #expect(throws: Stop.self) {
            try generator.generate(memory: memory, stopTimeSeconds: nil, checkpoint: { tokens in
                last = tokens
                if tokens == 12 { throw Stop() }
            })
        }
        #expect(last == 12)
    }

    @Test func transcriberForwardsProgressThenStops() throws {
        var (transcriber, waveform) = try tinyTranscriber(chunked: true)
        var seen = [SheetSage2Progress]()
        transcriber.checkpoint = { progress in
            seen.append(progress)
            if progress.stage == .decoding && progress.tokens == 10 { throw Stop() }
        }
        #expect(throws: Stop.self) { try transcriber.transcribe(waveform) }
        #expect(seen.first?.stage == .encoding && seen.last?.stage == .decoding && seen.last?.tokens == 10)
        #expect(zip(seen, seen.dropFirst()).allSatisfy { $0.overallFraction <= $1.overallFraction + 1e-12 })
    }

    @Test func cancelledTaskStops() async throws {
        let (transcriber, waveform) = try tinyTranscriber(chunked: false)
        let task = Task { () throws -> SheetSage2Transcription in
            var t = transcriber
            t.checkpoint = { _ in Thread.sleep(forTimeInterval: 0.002) }
            return try t.transcribe(waveform)
        }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
    }
}
