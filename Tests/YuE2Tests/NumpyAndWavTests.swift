// NumpyAndWavTests.swift - .npy round-trip and WAV export invariants (T-1.6)
// Copyright 2026 Vincent Gourbin

import AVFoundation
import Foundation
import MLX
import Testing
@testable import YuE2Core

@Suite("NumpyAndWav")
struct NumpyAndWavTests {
    private func tempURL(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString)-\(name)")
    }

    @Test func int32RoundTripIsByteIdentical() throws {
        let url = tempURL("semantic.npy")
        defer { try? FileManager.default.removeItem(at: url) }
        let values: [Int32] = [0, -1, 32767, -32768, 151_853, 5]
        try NumpyIO.writeInt32(values, url: url)
        let written = try Data(contentsOf: url)
        #expect(try NumpyIO.readInt32(url) == values)
        try NumpyIO.writeInt32(values, url: url)
        #expect(try Data(contentsOf: url) == written)
    }

    @Test func float32RoundTripIsByteIdentical() throws {
        let url = tempURL("latent.npy")
        defer { try? FileManager.default.removeItem(at: url) }
        let shape = [7, 64]
        let values = (0..<(7 * 64)).map { Float($0) * 0.5 - 3.25 }
        try NumpyIO.writeFloat32(values, shape: shape, url: url)
        let written = try Data(contentsOf: url)
        let (readShape, readValues) = try NumpyIO.readFloat32(url)
        #expect(readShape == shape)
        #expect(readValues == values)
        try NumpyIO.writeFloat32(values, shape: shape, url: url)
        #expect(try Data(contentsOf: url) == written)
    }

    @Test func readArrayProducesMLXArrayOfStoredShape() throws {
        let url = tempURL("latent.npy")
        defer { try? FileManager.default.removeItem(at: url) }
        try NumpyIO.writeFloat32([1, 2, 3, 4, 5, 6], shape: [2, 3], url: url)
        let array = try NumpyIO.readArray(url)
        #expect(array.shape == [2, 3])
        #expect(array.asArray(Float.self) == [1, 2, 3, 4, 5, 6])
    }

    @Test func wavInt16HasExpectedSizeAndIsReadableByAVFoundation() throws {
        let url = tempURL("out.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let numSamples = 1000
        let audio = MLXArray.zeros([2, numSamples])
        try AudioExporter.exportToWAV(audio: audio, url: url, sampleRate: 48_000, format: .int16)

        let data = try Data(contentsOf: url)
        #expect(data.count == 44 + numSamples * 2 * 2)

        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.sampleRate == 48_000)
        #expect(file.fileFormat.channelCount == 2)
        #expect(file.length == AVAudioFramePosition(numSamples))
    }

    @Test func wavFloat32IsReadableAsFloatFormat() throws {
        let url = tempURL("out-float.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let audio = MLXArray.zeros([2, 500])
        try AudioExporter.exportToWAV(audio: audio, url: url, sampleRate: 48_000, format: .float32)

        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.commonFormat == .pcmFormatFloat32)
        #expect(file.fileFormat.sampleRate == 48_000)
        #expect(file.fileFormat.channelCount == 2)
    }

    @Test func wavClampsOutOfRangeSamplesAtExportOnly() throws {
        let url = tempURL("clamped.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let audio = MLXArray([Float(2.0), Float(-2.0), Float(2.0), Float(-2.0)], [2, 2])
        try AudioExporter.exportToWAV(audio: audio, url: url, sampleRate: 48_000, format: .int16)
        let data = try Data(contentsOf: url)
        let samples = data.suffix(from: 44)
        let first = Int16(littleEndian: samples.withUnsafeBytes { $0.load(as: Int16.self) })
        #expect(first == 32767)
    }
}
