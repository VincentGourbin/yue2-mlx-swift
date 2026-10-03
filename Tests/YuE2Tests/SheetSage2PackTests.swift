// SheetSage2PackTests.swift - prequantized packs: save/load round trip (tiny, CPU) and real packs
// identical to the on-the-fly quantization (CPU, weights only), ASK Q10
// Copyright 2026 Vincent Gourbin

import Foundation
import MLX
import Testing
@testable import SheetSage2Core

private let repoRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

/// Every parameter equal, bit for bit (packed weights, scales, biases, float tensors).
private func expectIdenticalParameters(_ a: SheetSage2Model, _ b: SheetSage2Model, _ label: String) {
    let pa = Dictionary(uniqueKeysWithValues: a.parameters().flattened())
    let pb = Dictionary(uniqueKeysWithValues: b.parameters().flattened())
    #expect(Set(pa.keys) == Set(pb.keys), "\(label): parameter names differ")
    var differing = [String]()
    for (key, value) in pa {
        guard let other = pb[key] else { continue }
        if value.dtype != other.dtype || !MLX.arrayEqual(value, other).item(Bool.self) { differing.append(key) }
    }
    #expect(differing.isEmpty, "\(label): \(differing.count) tensors differ, e.g. \(differing.sorted().prefix(3))")
}

@Suite("SheetSage2PackRoundTrip")
struct SheetSage2PackRoundTripTests {
    @Test func tinyPackLoadsBackIdentically() throws {
        try Device.withDefaultDevice(.cpu) {
            let (arrays, metadata) = try loadArraysAndMetadata(
                url: repoRoot.appendingPathComponent("parity/tiny_sheetsage2.safetensors"))
            let config = try JSONDecoder().decode(SheetSage2Config.self, from: Data(try #require(metadata["config"]).utf8))
            let model = SheetSage2Model(config: config)
            var weights = [String: MLXArray]()
            for (key, value) in arrays where key.hasPrefix("weights.") { weights[String(key.dropFirst(8))] = value }
            try SheetSage2Weights.apply(SheetSage2Weights.sanitize(weights), to: model, dtype: .float16, subsamplingFloat32: true)
            SheetSage2Pack.quantizeEncoder(model, bits: 8, groupSize: 32)
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: dir) }
            try SheetSage2Pack.save(model, bits: 8, groupSize: 32, to: dir)
            let reloaded = try SheetSage2Model.load(directory: dir)
            #expect(reloaded.config.quantization == .init(bits: 8, groupSize: 32))
            expectIdenticalParameters(model, reloaded, "tiny 8-bit")
            let waveform = try #require(arrays["input_waveform"])
            #expect(MLX.arrayEqual(try model.encode(waveform), try reloaded.encode(waveform)).item(Bool.self))
        }
    }
}

private func packReady(_ name: String) -> Bool {
    guard let dir = modelsDir() else { return false }
    let fm = FileManager.default
    return fm.fileExists(atPath: dir.appendingPathComponent("\(name)/model.safetensors").path)
        && fm.fileExists(atPath: dir.appendingPathComponent("SheetSage2-fp16/model.safetensors").path)
}

/// A published pack is exactly what the matching profile builds on the fly from the fp16 pack.
@Suite("SheetSage2QuantizedPacks")
struct SheetSage2QuantizedPackTests {
    /// On the default (GPU) device, where the profiles quantize on the fly.
    private func check(_ name: String, bits: Int) throws {
        do {
            let dir = try #require(modelsDir())
            let profile = try #require(SheetSage2Profile.named("\(bits)bit-fast"))
            let pack = try SheetSage2Model.load(directory: dir.appendingPathComponent(name), profile: profile)
            let onTheFly = try SheetSage2Model.load(directory: dir.appendingPathComponent("SheetSage2-fp16"), profile: profile)
            expectIdenticalParameters(pack, onTheFly, name)
            #expect(throws: SheetSage2Error.self) {
                try SheetSage2Model.load(directory: dir.appendingPathComponent(name), profile: SheetSage2Profile.default)
            }
        }
    }

    @Test(.enabled(if: packReady("SheetSage2-q8"))) func q8() throws { try check("SheetSage2-q8", bits: 8) }
    @Test(.enabled(if: packReady("SheetSage2-q4"))) func q4() throws { try check("SheetSage2-q4", bits: 4) }
}
