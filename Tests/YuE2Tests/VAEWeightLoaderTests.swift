// VAEWeightLoaderTests.swift - weight-norm merge parity + strict apply() (T-1.7)
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import MLXNN
import Testing
@testable import YuE2Core

private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
private let tinyVAEFixture = repoRoot.appendingPathComponent("parity/tiny_vae.safetensors")

@Suite("VAEWeightLoader")
struct VAEWeightLoaderTests {
    @Test func weightNormMergeMatchesFixtureBeforeLayoutConversion() throws {
        let raw = try loadArrays(url: tinyVAEFixture)
        var checked = 0
        for (key, v) in raw where key.hasSuffix(".weight_v") {
            let base = String(key.dropLast(".weight_v".count))
            let g = try #require(raw[base + ".weight_g"])
            let expected = try #require(raw["expected_merged.\(base).weight"])
            let merged = VAEWeightLoader.mergeWeightNorm(g: g, v: v)
            let maxAbsDiff = MLX.max(MLX.abs(merged - expected)).item(Float.self)
            #expect(maxAbsDiff < 1e-6, "\(base): max abs diff \(maxAbsDiff)")
            checked += 1
        }
        // 44 decoder + 44 encoder weight-normed convs since T-5.1 (was 44, decoder-only, before).
        #expect(checked == 88)
    }

    @Test func loadDecoderWeightsProducesExactlyOneWeightPerNormedPair() throws {
        let raw = try loadArrays(url: tinyVAEFixture)
        let decoderKeys = raw.keys.filter { $0.hasPrefix("decoder.") }
        let pairs = decoderKeys.filter { $0.hasSuffix(".weight_v") }
        let expectedCount = decoderKeys.count - pairs.count

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        try FileManager.default.copyItem(at: tinyVAEFixture, to: tmpDir.appendingPathComponent("model.safetensors"))

        let loaded = try VAEWeightLoader.loadDecoderWeights(directory: tmpDir)
        #expect(loaded.count == expectedCount)
        #expect(loaded.count == 173)
        for value in loaded.values {
            #expect(value.dtype == .float32)
        }
    }

    @Test func applyRejectsUnexpectedKey() throws {
        let linear = Linear(4, 4)
        var weights = Dictionary(uniqueKeysWithValues: linear.parameters().flattened())
        weights["bogus.extra"] = MLXArray.zeros([1])
        #expect(throws: YuE2Error.self) {
            try WeightLoader.apply(weights, to: linear, component: "fake-linear")
        }
    }

    @Test func applyRejectsMissingKey() throws {
        let linear = Linear(4, 4)
        var weights = Dictionary(uniqueKeysWithValues: linear.parameters().flattened())
        let anyKey = try #require(weights.keys.first)
        weights.removeValue(forKey: anyKey)
        #expect(throws: YuE2Error.self) {
            try WeightLoader.apply(weights, to: linear, component: "fake-linear")
        }
    }

    @Test func applyAcceptsExactMatch() throws {
        let linear = Linear(4, 4)
        let weights = Dictionary(uniqueKeysWithValues: linear.parameters().flattened())
        try WeightLoader.apply(weights, to: linear, component: "fake-linear")
    }
}
