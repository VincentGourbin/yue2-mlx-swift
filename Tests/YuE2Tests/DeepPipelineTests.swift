// DeepPipelineTests.swift - the deep AR pipeline keeps tokens and cache bit-identical (checkpoint-backed)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import YuE2Core

private func eightBitPackReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    return YuE2PrequantizedCheckpoint.exists(
        lmDirectory: dir.appendingPathComponent("YuE2-3B"), quantization: .qint8All, quantizeHead: true)
}

/// The deep pipeline only moves *when* step t+1's network is submitted: the ABC phase and the
/// semantic phase under CFG must give the same tokens, and leave the cache at the same offset.
@Suite("DeepPipelineReal", .enabled(if: eightBitPackReady()))
struct DeepPipelineRealTests {
    @Test func deepPipelineKeepsTokensAndCache() async throws {
        let dir = try #require(modelsDir())
        let session = try await ModelSession.load(modelsDir: dir, quant: .qint8All, quantizeHead: true, residency: [.ar])
        let request = try SongRequest(
            style: "English, warm piano pop, expressive female voice, 88 BPM",
            lyrics: "[Verse]\nNeon fades along the lane\nFootsteps keep the time of rain",
            cot: .full, seed: 831_001, id: "deep_pipeline")
        let saved = YuE2ExecutionPolicy.deepPipeline
        defer { YuE2ExecutionPolicy.deepPipeline = saved }

        func run(deep: Bool) throws -> (abc: [Int], semantic: [Int], offsets: [Int]) {
            YuE2ExecutionPolicy.deepPipeline = deep
            var abcSampling = session.config.abc
            abcSampling.minTokens = 0
            abcSampling.maxTokens = 48
            let abcPrefix = try Prefixes.tokenPrefixes(request: request, tokenizer: session.tokenizer)
            let abcCache = KVCache()
            let abc = try TokenGenerator.generate(
                model: session.model, head: RestrictedHead(lmHead: session.model.lmHead, bounds: .abc),
                prefix: abcPrefix, sampling: abcSampling, seed: 7, bounds: .abc, cache: abcCache)

            var semanticSampling = session.config.semantic
            semanticSampling.minTokens = 0
            semanticSampling.maxTokens = 64
            let ids = abc.tokens.filter { $0 < YuE2Token.eod }
            let semanticCache = KVCache()
            let semantic = try TokenGenerator.generate(
                model: session.model, head: RestrictedHead(lmHead: session.model.lmHead, bounds: .semantic),
                prefix: try Prefixes.tokenPrefixes(request: request, tokenizer: session.tokenizer, abcIDs: ids),
                sampling: semanticSampling, seed: 11, bounds: .semantic,
                negative: try Prefixes.negativePrefix(request: request, tokenizer: session.tokenizer, abcIDs: ids),
                cfgScale: 1.5, cache: semanticCache)
            return (abc.tokens, semantic.tokens, [abcCache.offset, semanticCache.offset])
        }
        let base = try run(deep: false)
        let deep = try run(deep: true)
        #expect(base.abc.count > 8)
        #expect(base.semantic.count > 8)
        #expect(deep.abc == base.abc)
        #expect(deep.semantic == base.semantic)
        #expect(deep.offsets == base.offsets)
    }
}
