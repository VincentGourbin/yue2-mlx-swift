// SheetSage2ParityTests.swift - encoder and decoder parity, tiny (tier 1) and real (tier 2), plan/14 S-2/S-3
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import SheetSage2Core

private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

/// Tiny model built from `parity/tiny_sheetsage2.safetensors` (weights under `weights.`, config in
/// the metadata), plus the fixture's reference tensors.
private func tinyModel() throws -> (SheetSage2Model, [String: MLXArray]) {
    let (arrays, metadata) = try loadArraysAndMetadata(
        url: repoRoot.appendingPathComponent("parity/tiny_sheetsage2.safetensors"))
    let configJSON = try #require(metadata["config"])
    let config = try JSONDecoder().decode(SheetSage2Config.self, from: Data(configJSON.utf8))
    let model = SheetSage2Model(config: config)
    var weights = [String: MLXArray]()
    for (key, value) in arrays where key.hasPrefix("weights.") {
        weights[String(key.dropFirst("weights.".count))] = value
    }
    try SheetSage2Weights.apply(SheetSage2Weights.sanitize(weights), to: model, dtype: .float32)
    return (model, arrays)
}

private func expectClose(_ ours: MLXArray, _ reference: MLXArray, rel: Float, _ label: String) {
    let r = relativeError(ours, reference)
    #expect(ours.shape == reference.shape, "\(label) shape \(ours.shape) vs \(reference.shape)")
    #expect(r <= rel, "\(label) rel error \(r)")
}

@Suite("SheetSage2TinyParity")
struct SheetSage2TinyParityTests {
    @Test func encoderMatchesTorchAtEveryTap() throws {
        let (model, fixture) = try tinyModel()
        let waveform = try #require(fixture["input_waveform"])
        let taps = try model.encodeWithTaps(waveform, tapLayers: [0, 1])
        expectClose(taps.mel, try #require(fixture["encoder.mel"]), rel: 1e-5, "mel")
        for i in 0..<3 {
            expectClose(taps.subsampling[i][0], try #require(fixture["encoder.subsampling_\(i)"]), rel: 1e-4, "subsampling \(i)")
        }
        for i in 0..<2 {
            expectClose(try #require(taps.layers[i])[0], try #require(fixture["encoder.layer_\(i)"]), rel: 1e-4, "layer \(i)")
        }
        expectClose(taps.mixed[0], try #require(fixture["encoder.mixed"]), rel: 1e-4, "mixed")
        expectClose(taps.memory[0], try #require(fixture["encoder.memory"]), rel: 1e-4, "memory")
    }

    @Test func decoderTeacherForcedAndCachedMatchTorch() throws {
        let (model, fixture) = try tinyModel()
        let memory = try #require(fixture["encoder.memory"])[.newAxis]
        let tokens = try #require(fixture["decoder.tokens"]).asType(.int32)
        let reference = try #require(fixture["decoder.teacher_logits"])

        let full = model.decode(tokens[.newAxis], memory: memory, cache: model.makeCache())[0]
        expectClose(full, reference, rel: 1e-4, "teacher-forced logits")

        // Same logits one cached step at a time: positions, causal rows and cross-attention cache.
        let cache = model.makeCache()
        var steps = [MLXArray]()
        for i in 0..<tokens.dim(0) {
            steps.append(model.decode(tokens[i ..< i + 1][.newAxis], memory: memory, cache: cache)[0])
        }
        expectClose(concatenated(steps, axis: 0), reference, rel: 1e-4, "cached logits")
    }
}

private func realReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    let fm = FileManager.default
    return fm.fileExists(atPath: dir.appendingPathComponent("SheetSage2-merged/model.safetensors").path)
        && fm.fileExists(atPath: dir.appendingPathComponent("parity/sheetsage2_beethoven.safetensors").path)
}

@Suite("SheetSage2RealParity", .enabled(if: realReady()))
struct SheetSage2RealParityTests {
    private func run(fixtureName: String, dtype: DType, rel: Float) throws {
        let dir = try #require(modelsDir())
        let fixture = try loadFixture(name: fixtureName)
        let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-merged"), dtype: dtype, subsamplingFloat32: dtype != .float32)
        let waveform = try #require(fixture["input_waveform"])
        let taps = try model.encodeWithTaps(waveform, tapLayers: [0, 11, 23])
        expectClose(taps.mel, try #require(fixture["encoder.mel"]), rel: 1e-4, "\(fixtureName) mel")
        for i in 0..<3 {
            expectClose(taps.subsampling[i][0], try #require(fixture["encoder.subsampling_\(i)"]), rel: rel, "\(fixtureName) subsampling \(i)")
        }
        for i in [0, 11, 23] {
            expectClose(try #require(taps.layers[i])[0], try #require(fixture["encoder.layer_\(i)"]), rel: rel, "\(fixtureName) layer \(i)")
        }
        expectClose(taps.memory[0], try #require(fixture["encoder.memory"]), rel: rel, "\(fixtureName) memory")

        let tokens = try #require(fixture["decoder.tokens"]).asType(.int32)
        let positions = try #require(fixture["decoder.teacher_positions"]).asArray(Int32.self).map(Int.init)
        let logits = model.decode(tokens[.newAxis], memory: taps.memory, cache: model.makeCache())[0]
        let picked = stacked(positions.map { logits[$0] }, axis: 0)
        expectClose(picked, try #require(fixture["decoder.teacher_logits"]), rel: rel, "\(fixtureName) teacher logits")
    }

    @Test func beethovenFloat32() throws { try run(fixtureName: "sheetsage2_beethoven", dtype: .float32, rel: 1e-3) }
    @Test func beethovenFloat16() throws { try run(fixtureName: "sheetsage2_beethoven", dtype: .float16, rel: 1e-2) }
}

private func upstreamReady() -> Bool {
    guard let dir = modelsDir() else { return false }
    let fm = FileManager.default
    return realReady() && fm.fileExists(atPath: dir.appendingPathComponent("SheetSage2/model.safetensors").path)
        && fm.fileExists(atPath: dir.appendingPathComponent("MERT-v2-FullSong/model.safetensors").path)
}

/// `yue2 download --model sheetsage2` layout: adapters merged in Swift, against upstream's merge.
@Suite("SheetSage2AdapterMerge", .enabled(if: upstreamReady()))
struct SheetSage2AdapterMergeTests {
    @Test func swiftMergeMatchesUpstreamMergeAndTokens() throws {
        let dir = try #require(modelsDir())
        let config = try SheetSage2Config.load(dir.appendingPathComponent("SheetSage2/config.json"))
        let merged = try SheetSage2Weights.mergeAdapters(
            adapter: try loadArrays(url: dir.appendingPathComponent("SheetSage2/model.safetensors")),
            parent: try loadArrays(url: dir.appendingPathComponent("MERT-v2-FullSong/model.safetensors")), config: config)
        let reference = try loadArrays(url: dir.appendingPathComponent("SheetSage2-merged/model.safetensors"))
        var worst: Float = 0
        for layer in [0, 11, 23] {
            for projection in SheetSage2Weights.projections {
                let key = "encoder.layers.\(layer).attn.\(projection).weight"
                worst = max(worst, maxAbs(try #require(merged[key]), try #require(reference[key])))
            }
        }
        #expect(worst <= 1e-6, "merged weights max|Δ| \(worst)")

        let model = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2"), dtype: .float32, subsamplingFloat32: false)
        let fixture = try loadFixture(name: "sheetsage2_magic")
        let result = try SheetSage2Transcriber(model: model).transcribe(try #require(fixture["input_waveform"]))
        let expected = try #require(fixture["decoder.tokens"]).asArray(Int64.self).map(Int.init)
        #expect(result.tokens.first == expected)
    }
}

/// The published merged fp16 pack loads into exactly the model the 16-bit profiles build from the
/// upstream release: same tokens on both parity excerpts.
@Suite("SheetSage2FP16Pack", .enabled(if: upstreamReady() && modelsDir().map {
    FileManager.default.fileExists(atPath: $0.appendingPathComponent("SheetSage2-fp16/model.safetensors").path)
} == true))
struct SheetSage2FP16PackTests {
    @Test func packMatchesTheUpstreamReleaseUnderTheSixteenBitProfile() throws {
        let dir = try #require(modelsDir())
        let profile = SheetSage2Profile.named("16bit-fast")!
        let pack = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-fp16"), profile: profile)
        let upstream = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2"), profile: profile)
        for name in ["sheetsage2_beethoven", "sheetsage2_magic"] {
            let waveform = try #require(try loadFixture(name: name)["input_waveform"])
            let a = try SheetSage2Transcriber(model: pack, profile: profile).transcribe(waveform)
            let b = try SheetSage2Transcriber(model: upstream, profile: profile).transcribe(waveform)
            #expect(a.tokens == b.tokens, "\(name)")
            #expect(a.abc == b.abc, "\(name)")
        }
    }
}
